"""
Find open-access DES papers in PMC and download their full text as JATS XML.

PMC, and nothing else, because it is the only free source of machine-readable full
text. All four routes measured 2026-09-08:

    NCBI efetch (db=pmc)    WORKS -- the primary route. 25/25 random open-access DES
                            papers came back as full text; all 25 parsed as `jats`,
                            all 25 carried a DOI, 17/25 had tables.
    Europe PMC fullTextXML  works, kept as the fallback. But it 404s on anything it
                            has not ingested yet: 38 of 40 papers published in 2026,
                            which efetch served without complaint.
    Unpaywall               NO USE. 40/40 open-access DES DOIs had an OA location and
                            0/40 offered an .xml URL. It indexes PDFs and landing
                            pages, not full text.
    Crossref TDM links      NO USE. 18/60 DOIs declare an `application/xml` link and
                            0/60 actually downloaded -- 21 were api.elsevier.com
                            returning a 200-wrapped auth error, 3 Wiley 403.
    Elsevier article API    403 on every article without an institutional token; see
                            the note at the top of fetch_paper.py.

So the reachable corpus is ~13,000 papers, against the ~43,000 open-access DES works
OpenAlex knows about. The rest are Elsevier, RSC, ACS and Wiley, which never deposit
in PMC, and no amount of code reaches them.

efetch returns a <pmc-articleset> root, which `dialects.JATS.matches` already claims,
so none of this needs a new reader. This module finds, downloads and DESCRIBES. It
does not filter and it does not extract -- see the note on `manifest_row`.
"""
import re
import time
from pathlib import Path

import requests
from lxml import etree

from . import config, dialects, router, xml_utils

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EPMC_FULLTEXT = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
TOOL = "des-kg"

# The search. Deliberately just the phrase and its plural -- the whole point is to
# filter as little as possible at download time.
#
# "natural deep eutectic" is NOT in here and nothing is lost by that: NADES papers
# write "natural deep eutectic SOLVENT", which contains the phrase already. Measured:
# 3,611 open-access papers match "natural deep eutectic" and only 36 of them fall
# outside this term.
TERM = '("deep eutectic solvent" OR "deep eutectic solvents")'
# PMC's own licence filter. Without it the count rises by ~500, but those extra papers
# are readable on the website and not licensed for download.
OA_FILTER = '"open access"[filter]'
# The same phrase restricted to title and abstract, for the --counts comparison only.
# It is never used to decide what to download.
TERM_TIAB = '("deep eutectic solvent"[TIAB] OR "deep eutectic solvents"[TIAB])'

# esearch refuses retstart > 9998, so a 13,000-hit query cannot be paged through
# directly -- it has to be cut into slices each smaller than that. Publication year is
# the natural cut: it is stable, it partitions the corpus exactly (the per-year counts
# sum to the unsliced total, which `search` asserts), and the busiest single year is
# ~2,900. PMC holds no DES paper before 2009.
FIRST_YEAR = 2005
RETSTART_LIMIT = 9998
PAGE = 500                  # ids per esearch request
BATCH = 20                  # articles per efetch request; 20/20 verified to come back

_last_request = 0.0


def _throttle():
    """Keep to NCBI's rate limit: 3 requests/second, or 10 with an API key."""
    global _last_request
    interval = 0.11 if config.NCBI_API_KEY else 0.35
    wait = interval - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()


def session():
    s = requests.Session()
    s.headers["User-Agent"] = config.USER_AGENT
    return s


def _params(extra):
    """Every E-utilities call identifies itself, as NCBI's terms of use require."""
    base = {"db": "pmc", "tool": TOOL, "email": config.NCBI_EMAIL}
    if config.NCBI_API_KEY:
        base["api_key"] = config.NCBI_API_KEY
    return {**base, **extra}


def _call(s, endpoint, params, method="GET", tries=3):
    """One E-utilities request, with a backoff. Raises once it has given up.

    E-utilities reports some failures as an HTTP 200 whose body is a broken JSON error
    document -- that is how the retstart cap surfaces -- so the caller checks content,
    not just the status code.
    """
    for attempt in range(tries):
        _throttle()
        try:
            if method == "POST":
                r = s.post(f"{EUTILS}/{endpoint}", data=_params(params), timeout=180)
            else:
                r = s.get(f"{EUTILS}/{endpoint}", params=_params(params), timeout=180)
            if r.status_code == 200:
                return r
            if r.status_code not in (429, 500, 502, 503, 504):
                r.raise_for_status()
        except requests.RequestException:
            if attempt == tries - 1:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError(f"{endpoint} failed after {tries} attempts")


# ---------------------------------------------------------------- counting

def _term(oa_only=True, year=None, since=None, tiab=False):
    parts = [TERM_TIAB if tiab else TERM]
    if oa_only:
        parts.append(OA_FILTER)
    if year is not None:
        parts.append(f"{year}:{year}[pdat]")
    elif since is not None:
        parts.append(f"{since}:3000[pdat]")
    return " AND ".join(parts)


def count(s, **kwargs):
    """How many PMC records the term matches. -> int."""
    r = _call(s, "esearch.fcgi", {"term": _term(**kwargs), "retmode": "json", "retmax": 0})
    return int(r.json()["esearchresult"]["count"])


def counts(s):
    """The corpus census: how many papers exist, and how many really are about DES.

    Both columns matter. The phrase matches anywhere in the full text, and PMC indexes
    reference lists -- so most of the `anywhere` column is papers that merely CITE DES
    work. Checked by hand on 40 of them: 6 title/abstract, 28 body only, 5 reference
    list only, 1 not found, and just 1 of 40 mentioned it inside a table.

    The download uses the `anywhere` set regardless. `manifest_row` records where the
    phrase actually landed for each paper, which makes narrowing later a query over the
    manifest instead of a second download.
    """
    rows = []
    for label, since in (("all years", None), ("2015+", 2015), ("2018+", 2018),
                         ("2020+", 2020), ("2023+", 2023)):
        rows.append({
            "scope": label,
            "anywhere": count(s, since=since),
            "title_abstract": count(s, since=since, tiab=True),
        })
    return rows


# ---------------------------------------------------------------- searching

def search(s, since=None, limit=None, verbose=True):
    """Every PMCID the term matches, oldest year first. -> list[str] like "PMC12345".

    Sliced by publication year because esearch will not page past retstart 9998, and
    the corpus is bigger than that. The slices are asserted to sum to the unsliced
    count: if they ever stop agreeing, this is silently downloading a subset, which is
    exactly the kind of failure that looks like success.
    """
    total = count(s, since=since)
    this_year = time.gmtime().tm_year
    years = range(max(FIRST_YEAR, since or FIRST_YEAR), this_year + 2)

    found, seen = [], set()
    for year in years:
        n = count(s, year=year)
        if not n:
            continue
        if n > RETSTART_LIMIT:
            raise RuntimeError(
                f"{year} alone has {n} hits, past esearch's retstart cap of "
                f"{RETSTART_LIMIT}. Slice by month as well before trusting this run.")
        for start in range(0, n, PAGE):
            r = _call(s, "esearch.fcgi", {"term": _term(year=year), "retmode": "json",
                                          "retmax": PAGE, "retstart": start})
            result = r.json()["esearchresult"]
            if result.get("ERROR"):
                raise RuntimeError(f"esearch {year}@{start}: {result['ERROR']}")
            for uid in result.get("idlist", []):
                pmcid = uid if str(uid).startswith("PMC") else f"PMC{uid}"
                if pmcid not in seen:
                    seen.add(pmcid)
                    found.append(pmcid)
        if verbose:
            print(f"    {year}  {n:>5}  (running total {len(found)})")
        if limit and len(found) >= limit:
            if verbose:
                print(f"    stopping at --limit {limit}")
            return found[:limit]

    if len(found) != total:
        print(f"  WARNING: year slices found {len(found)} ids but the unsliced query "
              f"counts {total}. The difference is papers whose publication date falls "
              f"outside {years.start}-{years.stop - 1}, or records added mid-run.")
    return found


# ---------------------------------------------------------------- downloading

def _pmcid_of(article):
    for aid in article.findall(".//front/article-meta/article-id"):
        if aid.get("pub-id-type") in ("pmcid", "pmc"):
            value = (aid.text or "").strip()
            if value:
                return value if value.startswith("PMC") else f"PMC{value}"
    return ""


def split_articleset(content):
    """One efetch response -> [(pmcid, bytes)], one self-contained document each.

    Each article is re-wrapped in its own <pmc-articleset> so the file on disk has the
    same shape as a single-article efetch, and so `dialects.JATS._article` finds it
    where it expects to. Verified by round-tripping: a split article re-reads as `jats`
    with its DOI, tables and references intact.
    """
    root = etree.fromstring(content)
    out = []
    for article in list(root.findall("article")):
        pmcid = _pmcid_of(article)
        if not pmcid:
            continue
        wrapper = etree.Element("pmc-articleset")
        wrapper.append(article)          # moves it out of the batch; hence list() above
        out.append((pmcid, etree.tostring(wrapper, xml_declaration=True,
                                          encoding="utf-8")))
    return out


def path_for(pmcid, out_dir=None):
    """Where a paper's XML lives. Named by PMCID, not by author.

    The PMCID is the one identifier that is stable, unique and known before the file is
    parsed, which is what makes a 13,000-file download resumable by an `exists()` check.
    The readable `<Surname>_<Year>` name that the pipeline's per-paper directories use
    is applied by select_papers.py when a paper is actually promoted for a run.
    """
    return Path(out_dir or config.PMC_DIR) / f"{pmcid}.xml"


def _from_europe_pmc(s, pmcid):
    """The fallback route. -> bytes, or None."""
    _throttle()
    try:
        r = s.get(EPMC_FULLTEXT.format(pmcid=pmcid), timeout=120)
    except requests.RequestException:
        return None
    if r.status_code == 200 and b"<body" in r.content:
        return r.content
    return None


def download(s, pmcids, out_dir=None, verbose=True):
    """Fetch every paper that is not already on disk. -> (written, skipped, failed).

    Batched: efetch takes 20 ids per POST, so the whole corpus is ~660 requests rather
    than 13,000. Interrupting is safe -- anything already written is skipped on the next
    run, and nothing is deleted.
    """
    out_dir = Path(out_dir or config.PMC_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = [p for p in pmcids if not path_for(p, out_dir).exists()]
    skipped = len(pmcids) - len(todo)
    if verbose and skipped:
        print(f"  {skipped} already on disk")

    written, failed = [], []
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        try:
            r = _call(s, "efetch.fcgi", {"id": ",".join(batch), "rettype": "xml",
                                         "retmode": "xml"}, method="POST")
            articles = split_articleset(r.content)
        except Exception as exc:
            print(f"  batch {i // BATCH + 1}: {type(exc).__name__}: {exc}")
            articles = []

        got = set()
        for pmcid, blob in articles:
            if b"<body" not in blob:
                continue                 # metadata only; try the fallback below
            path_for(pmcid, out_dir).write_bytes(blob)
            written.append(pmcid)
            got.add(pmcid)

        # Anything the batch did not return in full, ask Europe PMC for individually.
        for pmcid in batch:
            if pmcid in got:
                continue
            blob = _from_europe_pmc(s, pmcid)
            if blob:
                path_for(pmcid, out_dir).write_bytes(blob)
                written.append(pmcid)
            else:
                failed.append(pmcid)

        if verbose and (i // BATCH) % 10 == 0:
            print(f"    {i + len(batch):>6}/{len(todo)}  written {len(written)}  "
                  f"failed {len(failed)}")
    return written, skipped, failed


# ---------------------------------------------------------------- describing

# Where the search phrase actually occurs. A paper matching only in its reference list
# cites DES work rather than reporting any.
#
# The separators have to be optional and repeatable: PMC's index normalises hyphens, so a
# paper writing "deep-eutectic-solvents" is a legitimate search hit, and XML is
# pretty-printed so the words can be split by a newline and an indent. Requiring a single
# space reported 5 of the first 60 papers as `absent` when 4 of them say it hyphenated in
# their own abstract.
_PHRASE = re.compile(r"deep[\s\-‐-―]*eutectic[\s\-‐-―]*solvents?", re.I)


def _text_of(nodes):
    """Flattened text of several elements, whitespace-normalised as xml_utils.text is."""
    joined = " ".join(etree.tostring(n, method="text", encoding="unicode") for n in nodes)
    return re.sub(r"\s+", " ", joined)


def corpus_dois():
    """DOIs already in the extracted corpus, so they are never re-selected. -> set."""
    if not config.PAPERS_CSV.exists():
        return set()
    import pandas as pd

    frame = pd.read_csv(config.PAPERS_CSV)
    column = "Paper_DOI" if "Paper_DOI" in frame.columns else frame.columns[0]
    return {str(d).strip().lower() for d in frame[column].dropna()}


def manifest_row(path, known_dois=()):
    """Describe one downloaded paper as the extractor would see it. -> dict.

    Everything here is a measurement, not a judgement: it reads the file with the
    pipeline's own `dialects.detect` and `router.route`, so `n_tables` is the number of
    tables `extract_table` would actually be handed, and a file that cannot be parsed
    says so in `error` instead of quietly not appearing.

    Nothing in this row filters anything. select_papers.py ranks on these columns; the
    download ignores them.
    """
    row = {"pmcid": path.stem, "doi": "", "slug": "", "title": "", "journal": "",
           "year": "", "article_type": "", "dialect": "", "phrase_in": "",
           "n_tables": 0, "n_figures": 0, "n_sections": 0, "n_references": 0,
           "refs_with_doi": 0, "kb": path.stat().st_size // 1024,
           "path": str(path), "already_in_corpus": "", "error": ""}
    try:
        root = xml_utils.load_root(path)
        dialect = dialects.detect(root)
        meta = dialect.paper_metadata(root)
        routed = router.route(root, dialect)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"[:160]
        return row

    bibliography = {}
    try:
        bibliography = dialect.bibliography(routed.references)
    except Exception as exc:
        row["error"] = f"bibliography: {type(exc).__name__}"

    title_abstract = _text_of(root.findall(".//front//article-title")) + " " + \
        _text_of(root.findall(".//front//abstract"))
    body = _text_of(root.findall(".//body"))
    reference_list = _text_of(root.findall(".//back//ref-list"))
    if _PHRASE.search(title_abstract):
        row["phrase_in"] = "title_abstract"
    elif _PHRASE.search(body):
        row["phrase_in"] = "body"
    elif _PHRASE.search(reference_list):
        row["phrase_in"] = "reference_list"
    else:
        row["phrase_in"] = "absent"

    doi = str(meta.get("doi") or "").strip()
    row.update({
        "doi": doi,
        # The name this paper would get if it were promoted into xml/. Recorded rather
        # than used, so the manifest can be read alongside data/papers.csv.
        "slug": proposed_slug(meta, row["pmcid"]),
        "title": str(meta.get("title") or "")[:200],
        "journal": str(meta.get("journal") or "")[:120],
        "year": str(meta.get("year") or ""),
        "article_type": dialect.article_type(root),
        "dialect": dialect.name,
        "n_tables": len(routed.tables),
        "n_figures": len(routed.figures),
        "n_sections": len(routed.sections),
        "n_references": len(routed.references),
        "refs_with_doi": sum(1 for r in bibliography.values() if r.get("doi")),
        "already_in_corpus": "yes" if doi.lower() in set(known_dois) else "",
    })
    return row


def proposed_slug(meta, pmcid):
    """'<Surname>_<Year>', the convention `paper.slug_of` produces. -> str.

    Falls back to the PMCID when the metadata is too thin to name a paper, rather than
    producing a slug like `_2024` that collides with every other nameless paper.
    """
    authors = meta.get("authors") or ""
    first = str(authors).split(";")[0].strip()
    surname = first.split()[-1] if first else ""
    year = str(meta.get("year") or "").strip()
    stem = f"{surname}_{year}" if surname and year else pmcid
    return re.sub(r"[^A-Za-z0-9_.-]", "_", stem)


def build_manifest(out_dir=None, verbose=True):
    """Describe every downloaded paper. -> list[dict], written to PMC_CORPUS_CSV."""
    out_dir = Path(out_dir or config.PMC_DIR)
    paths = sorted(out_dir.glob("*.xml"))
    known = corpus_dois()
    rows = []
    for n, path in enumerate(paths, 1):
        rows.append(manifest_row(path, known))
        if verbose and n % 500 == 0:
            print(f"    described {n}/{len(paths)}")
    rows.sort(key=lambda r: (r["year"], r["pmcid"]), reverse=True)
    xml_utils.write_csv(rows, config.PMC_CORPUS_CSV)
    return rows
