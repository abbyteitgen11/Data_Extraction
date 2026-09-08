"""
The references route: read the bibliography, then find each entry's DOI.

Two network passes, both cached in reference_map.json and both resumable:

  1. resolve  -- Crossref bibliographic search  -> doi + match_score   (flag _resolved)
  2. enrich   -- GET works/{doi}                -> authors, journal,
                                                   volume, issue, pages   (flag _enriched)

Pass 2 exists because the paper's own <volume-nr> is unreliable: for reference [1]
it holds "70-71", which is a page range, not a volume. Crossref keyed by DOI is
exact, and because it is keyed it never re-runs the expensive search from pass 1.

Enriched values are written to separate cr_* keys so the cache stays
backwards-compatible and the original XML metadata is never overwritten.
"""
import html
import json
import time

import requests

from . import config, paper as paper_mod
from .schema import ReferenceRow


# ---------- 1. read the bibliography out of the XML ----------
def parse_bibliography(reference_elements, dialect):
    """-> {reference number: metadata dict}, via the format's own reader.

    Bibliographies differ more than tables do. Elsevier labels every entry `[40]`;
    Canela-Xandri's JATS has no <label> at all and Fan's are "1." -- neither of which
    is `.isdigit()`, so the old Elsevier-only parser silently returned {} for both.
    Each dialect now reads its own, and numbering falls back to list position.
    """
    return dialect.bibliography(reference_elements)


# ---------- cache ----------
# Reference numbers are meaningful only inside one paper: Sadeghi's [40] and
# Canela-Xandri's [40] are different studies. The cache is therefore per paper, and
# `paper` is a required argument everywhere -- a default would be an invitation to
# reintroduce exactly the cross-paper bleed this replaced.
def cache_path(paper):
    return config.PAPERS_DIR / paper.slug / "reference_map.json"


def _migrate_legacy(paper, refs):
    """Adopt data/reference_map.json for the paper it actually belongs to. -> dict.

    The old global cache holds one paper's 343 resolved references and it would be
    wasteful to re-query them. But it carries no record of whose they are, so it is
    adopted only when its entries demonstrably match this paper's own bibliography:
    the `raw` citation strings have to agree. A mismatch means the file belongs to a
    different paper, and it is ignored rather than trusted.
    """
    legacy = config.LEGACY_REFERENCE_CACHE
    if not legacy.exists() or not refs:
        return {}
    try:
        cached = {int(k): v for k, v in json.loads(legacy.read_text()).items()}
    except (ValueError, TypeError):
        return {}

    shared = [n for n in cached if n in refs and cached[n].get("raw")]
    if not shared:
        return {}
    agree = sum(1 for n in shared
                if str(cached[n].get("raw", ""))[:60] == str(refs[n].get("raw", ""))[:60])
    if agree < 0.9 * len(shared):
        print(f"    legacy reference cache does not match {paper.slug} "
              f"({agree}/{len(shared)} citations agree) -- ignoring it")
        return {}
    print(f"    migrated {len(cached)} cached references from "
          f"{legacy.name} ({agree}/{len(shared)} citations agree)")
    return cached


def load_cache(paper, refs=None):
    """The paper's own resolved references, migrating the legacy file once if needed."""
    path = cache_path(paper)
    if path.exists():
        return {int(k): v for k, v in json.loads(path.read_text()).items()}
    migrated = _migrate_legacy(paper, refs or {})
    if migrated:
        save_cache(migrated, paper)
    return migrated


def save_cache(cache, paper):
    path = cache_path(paper)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))


# What the cache is allowed to contribute: the results of a lookup, never the fields we
# just read out of the XML. Letting a cached entry win over a fresh parse means an
# improvement to the reader never reaches an already-cached paper -- the readable `raw`
# citation and the new volume/fpage stayed invisible for the 78 references that happened
# to have resolved on an earlier run.
_LOOKUP_KEYS = {"doi", "doi_source", "match_score", "match_basis", "fields_agree",
                "title_agreement", "_resolved", "_enriched", "cr_authors", "cr_title",
                "cr_journal", "cr_year", "volume", "issue", "pages"}


def apply_cache(reference_map, cache):
    """Merge cached LOOKUP RESULTS into a freshly-parsed reference map."""
    for num, cached in cache.items():
        if num in reference_map:
            results = {k: v for k, v in cached.items() if k in _LOOKUP_KEYS}
            reference_map[num] = {**reference_map[num], **results}
    return reference_map


# ---------- the shared, DOI-keyed Crossref cache ----------
# Unlike a reference number, a DOI means the same thing in every paper, so this one IS
# corpus-wide. Two reviews citing the same study then cost one lookup, not two.
def load_crossref_cache():
    if not config.CROSSREF_CACHE.exists():
        return {}
    try:
        return json.loads(config.CROSSREF_CACHE.read_text())
    except (ValueError, TypeError):
        return {}


def save_crossref_cache(cache):
    config.CROSSREF_CACHE.parent.mkdir(parents=True, exist_ok=True)
    config.CROSSREF_CACHE.write_text(json.dumps(cache, indent=2, ensure_ascii=False))


# ---------- 2. Crossref: find the DOI ----------
def search_query(meta):
    """What to ask Crossref for this citation. -> (query, basis).

    A title is the strongest signal, but plenty of publishers do not supply one: RSC's
    JATS gives article-title for NONE of Canela-Xandri's 203 references, only author,
    journal, year, volume and first page. Sending "{title} {journal} {year}" for those
    means sending "Green Chem. 2017" -- a journal and a year -- which is why 125 of them
    scored under 40 and found nothing.

    Author + journal + year + volume + page identifies a paper about as precisely as a
    title does, and Crossref's bibliographic index handles it well.
    """
    title = str(meta.get("title") or "").strip()
    journal = str(meta.get("journal") or "").strip()
    year = str(meta.get("year") or "").strip()
    if title:
        return " ".join(p for p in (title, journal, year) if p), "title"

    # No title: lead with the first author's surname, which is the discriminating part.
    authors = meta.get("authors") or []
    surname = str(authors[0]).split()[-1] if authors else ""
    parts = [surname, journal, year,
             str(meta.get("volume") or ""), str(meta.get("fpage") or "")]
    query = " ".join(p for p in parts if p.strip())
    if surname and journal and (meta.get("volume") or meta.get("fpage")):
        return query, "journal_volume_page"
    # Not enough to be worth a lookup that would only invite a wrong match.
    return (query if query.strip() else ""), "insufficient"


def _fields_agree(meta, item):
    """Do the citation's year, volume and first page match the record Crossref returned?

    This exists because Crossref's `score` is a Lucene relevance score, not a
    confidence: it scales with how many terms the query had. "Sheldon Green Chem. 2017
    19 18" is a SHORTER and MORE precise query than a full title, and it scores ~35 --
    below the threshold tuned for title queries -- while returning exactly the right
    paper. Rejecting on score alone threw away correct matches for all 125 of
    Canela-Xandri's title-less citations.

    Year, volume and first page agreeing is far stronger evidence than any score. A
    journal has one article at a given volume and page.
    """
    def same(a, b):
        a, b = str(a or "").strip().lower(), str(b or "").strip().lower()
        return bool(a) and bool(b) and a == b

    year = ""
    parts = (item.get("issued") or {}).get("date-parts") or [[None]]
    if parts and parts[0] and parts[0][0]:
        year = str(parts[0][0])
    pairs = {
        "year": (meta.get("year"), year),
        "volume": (meta.get("volume"), item.get("volume")),
        "page": (meta.get("fpage"), str(item.get("page") or "").split("-")[0]),
    }
    # Compare only what BOTH sides state. Demanding all three rejects correct matches
    # for no reason: Abbott's Chem. Commun. 2001, 2010 carries no volume in the
    # citation, so requiring one threw away the right DOI.
    stated = {k: v for k, v in pairs.items() if str(v[0] or "").strip()
              and str(v[1] or "").strip()}
    if not stated or any(not same(a, b) for a, b in stated.values()):
        return False
    # Year alone is far too weak -- a journal publishes hundreds of papers a year -- so
    # a volume or a page has to be among the fields that agreed.
    return "year" in stated and bool({"volume", "page"} & set(stated))


def crossref_search(meta, session, max_retries=5):
    """Bibliographic search -> (doi, score, basis, fields_agree). Backs off on 429."""
    query, basis = search_query(meta)
    if not query or basis == "insufficient":
        return None, 0.0, basis, False
    params = {"query.bibliographic": query, "rows": 1, "mailto": config.MAILTO}
    wait = 2.0
    for attempt in range(max_retries):
        try:
            r = session.get(
                "https://api.crossref.org/works",
                params=params,
                headers={"User-Agent": config.USER_AGENT},
                timeout=30,
            )
            if r.status_code == 429:
                delay = float(r.headers.get("Retry-After", wait))
                print(f"    429 on ref {meta['num']}; waiting {delay:.0f}s")
                time.sleep(delay)
                wait *= 2
                continue
            r.raise_for_status()
            items = r.json().get("message", {}).get("items", [])
            if not items:
                return None, 0.0, basis, False
            item = items[0]
            return (item.get("DOI"), item.get("score", 0.0), basis,
                    _fields_agree(meta, item))
        except (requests.RequestException, ValueError) as exc:
            print(f"    lookup error ref {meta['num']} (attempt {attempt + 1}): {exc}")
            time.sleep(wait)
            wait *= 2
    return None, 0.0, basis, False


# ---------- 3. Crossref: full metadata for a known DOI ----------
def crossref_metadata(doi, session, max_retries=3, shared=None):
    """GET works/{doi} -> the cr_* fields. Exact lookup, so no scoring needed.

    `shared` is the corpus-wide DOI-keyed cache. Because the lookup is keyed by DOI it
    is paper-independent, so the second review to cite a study pays nothing for it.
    """
    key = paper_mod.normalise_doi(doi)
    if shared is not None and key in shared:
        return shared[key]
    wait = 2.0
    for attempt in range(max_retries):
        try:
            r = session.get(
                f"https://api.crossref.org/works/{doi}",
                headers={"User-Agent": config.USER_AGENT},
                timeout=30,
            )
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", wait)))
                wait *= 2
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            m = r.json()["message"]
        except (requests.RequestException, ValueError, KeyError) as exc:
            print(f"    metadata error for {doi} (attempt {attempt + 1}): {exc}")
            time.sleep(wait)
            wait *= 2
            continue

        parts = (m.get("published") or m.get("issued") or {}).get("date-parts") or [[None]]
        fields = {
            "cr_authors": "; ".join(
                f"{a.get('given', '')} {a.get('family', '')}".strip()
                for a in m.get("author", [])
            ),
            "cr_title": (m.get("title") or [""])[0],
            "cr_journal": (m.get("container-title") or [""])[0],
            "volume": m.get("volume") or "",
            "issue": m.get("issue") or "",
            "pages": m.get("page") or m.get("article-number") or "",
            "cr_year": str(parts[0][0] or ""),
        }
        if shared is not None:
            shared[key] = fields
        return fields
    return None


# ---------- orchestration ----------
def _title_agreement(a, b):
    """Fraction of the XML title's words that appear in the Crossref title.

    A low value means Crossref probably matched the wrong paper.
    """
    wa = {w for w in a.lower().split() if len(w) > 3}
    wb = {w for w in b.lower().split() if len(w) > 3}
    if not wa or not wb:
        return None
    return round(len(wa & wb) / len(wa), 3)


def resolve_all(reference_map, paper, cache=None, network=True):
    """Fill in DOIs. Entries already flagged _resolved are never re-requested.

    A reference that states its own DOI -- JATS `<pub-id pub-id-type="doi">`, which
    76/203 and 33/34 of the two new papers carry -- needs no search at all. The
    publisher's own assertion beats a fuzzy bibliographic match, so those are marked
    resolved with a perfect score before Crossref is ever asked.
    """
    cache = cache if cache is not None else load_cache(paper, reference_map)
    reference_map = apply_cache(reference_map, cache)

    for meta in reference_map.values():
        if meta.get("doi") and not meta.get("_resolved"):
            meta["match_score"] = meta.get("match_score", 100.0)
            meta["match_basis"] = "inline_doi"
            meta["doi_source"] = meta.get("doi_source") or "xml"
            meta["_resolved"] = True

    todo = [n for n, m in reference_map.items() if not m.get("_resolved")]

    if not network:
        print(f"  resolve: {len(reference_map) - len(todo)} from cache, "
              f"{len(todo)} unresolved (offline)")
        return reference_map
    if not todo:
        print(f"  resolve: all {len(reference_map)} references already cached")
        return reference_map

    print(f"  resolve: {len(todo)} of {len(reference_map)} references via Crossref search")
    session = requests.Session()
    for i, num in enumerate(sorted(todo), 1):
        meta = reference_map[num]
        doi, score, basis, agree = crossref_search(meta, session)
        # Accept on EITHER a strong relevance score or exact bibliographic agreement.
        # The score is tuned for title queries; year+volume+page agreeing is the better
        # evidence when there is no title to query with.
        accepted = score >= config.MIN_MATCH_SCORE or agree
        meta["doi"] = doi if accepted else None
        meta["match_score"] = score
        meta["match_basis"] = f"{basis}+fields" if (agree and doi) else basis
        meta["fields_agree"] = bool(agree)
        meta["doi_source"] = "crossref_search" if meta["doi"] else ""
        meta["_resolved"] = True
        cache[num] = meta
        time.sleep(0.05)
        if i % 25 == 0:
            save_cache(cache, paper)
            print(f"    {i}/{len(todo)} resolved (checkpointed)")

    save_cache({**cache, **reference_map}, paper)
    have = sum(1 for m in reference_map.values() if m.get("doi"))
    print(f"  resolve: {have}/{len(reference_map)} have a DOI (score >= {config.MIN_MATCH_SCORE})")
    return reference_map


def enrich_all(reference_map, paper, cache=None, network=True):
    """Add volume/issue/pages/authoritative authors for every resolved DOI."""
    cache = cache if cache is not None else load_cache(paper, reference_map)
    reference_map = apply_cache(reference_map, cache)
    todo = [n for n, m in reference_map.items() if m.get("doi") and not m.get("_enriched")]

    if not network:
        done = sum(1 for m in reference_map.values() if m.get("_enriched"))
        print(f"  enrich:  {done} from cache, {len(todo)} un-enriched (offline)")
        return reference_map
    if not todo:
        print(f"  enrich:  all resolved references already enriched")
        return reference_map

    print(f"  enrich:  {len(todo)} DOIs via Crossref works/{{doi}}")
    session = requests.Session()
    shared = load_crossref_cache()
    reused = 0
    for i, num in enumerate(sorted(todo), 1):
        meta = reference_map[num]
        before = len(shared)
        extra = crossref_metadata(meta["doi"], session, shared=shared)
        reused += int(len(shared) == before and extra is not None)
        if extra:
            meta.update(extra)
            meta["title_agreement"] = _title_agreement(meta.get("title", ""), extra["cr_title"])
        meta["_enriched"] = True
        cache[num] = meta
        time.sleep(0.05)
        if i % 25 == 0:
            save_cache(cache, paper)
            save_crossref_cache(shared)
            print(f"    {i}/{len(todo)} enriched (checkpointed)")

    save_cache({**cache, **reference_map}, paper)
    save_crossref_cache(shared)
    if reused:
        print(f"  enrich:  {reused} DOI(s) already known from another paper")
    with_vol = sum(1 for m in reference_map.values() if m.get("volume"))
    print(f"  enrich:  {with_vol}/{len(reference_map)} now have a volume")
    return reference_map


def enrich_review(review, paper, network=True, cache=None):
    """Top up the paper's own metadata from Crossref.

    Elsevier's <coredata> carries no issue number, so the paper we are extracting
    from goes through the same lookup as every paper it cites. Cached under key 0,
    which no real reference number uses.
    """
    cache = cache if cache is not None else load_cache(paper)
    cached = cache.get(0)
    if cached:
        return {**review, **cached}
    if not network or not review.get("doi"):
        return review

    shared = load_crossref_cache()
    extra = crossref_metadata(review["doi"], requests.Session(), shared=shared)
    save_crossref_cache(shared)
    if not extra:
        return review
    merged = {
        "authors": extra["cr_authors"] or review.get("authors", ""),
        "title": extra["cr_title"] or review.get("title", ""),
        "journal": extra["cr_journal"] or review.get("journal", ""),
        "volume": extra["volume"] or review.get("volume", ""),
        "issue": extra["issue"] or review.get("issue", ""),
        "pages": extra["pages"] or review.get("pages", ""),
        "year": extra["cr_year"] or review.get("year", ""),
    }
    cache[0] = merged
    save_cache(cache, paper)
    return {**review, **merged}


# ---------- the shape everything downstream reads ----------
def reference_fields(meta, owner_key):
    """Normalise one reference to flat strings, preferring Crossref over the XML.

    Crossref returns HTML-escaped text, so 34 of these journals arrive as
    "Chemical Engineering &amp; Technology". Unescaping here rather than at fetch
    time fixes what is already in reference_map.json without re-querying.
    """
    authors = meta.get("cr_authors") or "; ".join(meta.get("authors") or [])
    fields = {
        "key": paper_key(meta, owner_key),
        "doi": meta.get("doi") or "",
        "match_score": meta.get("match_score"),
        "match_basis": meta.get("match_basis") or "",
        "title_agreement": meta.get("title_agreement"),
        "raw": meta.get("raw") or "",
        "authors": authors,
        "title": meta.get("cr_title") or meta.get("title") or "",
        "journal": meta.get("cr_journal") or meta.get("journal") or "",
        "volume": meta.get("volume") or "",
        "issue": meta.get("issue") or "",
        "pages": meta.get("pages") or "",
        "year": meta.get("cr_year") or meta.get("year") or "",
    }
    for key in ("authors", "title", "journal"):
        fields[key] = html.unescape(str(fields[key]))
    return fields


def paper_key(meta, owner_key):
    """Stable node identity: the DOI when Crossref matched one, else '<review>#refN'.

    A reference Crossref could not resolve is still a real paper that real
    measurements came from, and it still needs to be exactly one node. Merging on
    `doi` cannot do that -- Neo4j uniqueness constraints ignore nulls, so every
    DOI-less reference would collapse into one another silently.
    """
    if not owner_key:
        raise ValueError("paper_key requires the owning paper's key; a reference must "
                         "never inherit another paper's identity")
    return meta.get("doi") or f"{owner_key}#ref{meta['num']}"


def numbers_for_ids(ref_ids, reference_map):
    """Resolve <xref rid="cit59"> to reference numbers. -> list[int].

    Preferred over parsing the printed text wherever the format provides it. The
    printed "59" is a convention; the rid is a link the publisher asserted, so it
    survives superscripts, ranges written with en-dashes, and bibliographies whose
    numbering does not start at one.
    """
    by_id = {m.get("id"): n for n, m in (reference_map or {}).items() if m.get("id")}
    return [by_id[rid] for rid in ref_ids if rid in by_id]


def sources(ref_numbers, reference_map, owner_key):
    """Aligned per-source metadata for a set of cited reference numbers.

    EVERY cited reference is included, whether or not Crossref matched it: `key` is
    always populated and is what the graph merges on, while `doi` may be blank. The
    lists stay positionally aligned -- the n-th key belongs to the n-th title --
    precisely because nothing is skipped.

    Shared by the table route and the prose route so both carry provenance the same way.
    """
    keys = ("key", "doi", "authors", "title", "journal", "volume", "issue", "pages", "year")
    collected = {k: [] for k in keys}
    for n in ref_numbers:
        meta = (reference_map or {}).get(n)
        if not meta:                        # a citation number with no bibliography entry
            continue
        fields = reference_fields(meta, owner_key)
        for k in keys:
            collected[k].append(str(fields[k]).replace(config.SOURCE_SEP, "/"))
    return {k: config.SOURCE_SEP.join(v) for k, v in collected.items()}


def to_rows(reference_map, owner_key):
    """-> list[ReferenceRow], ordered by reference number."""
    rows = []
    for meta in sorted(reference_map.values(), key=lambda m: m["num"]):
        f = reference_fields(meta, owner_key)
        rows.append(ReferenceRow(
            ref_number=meta["num"],
            key=f["key"],
            authors=f["authors"],
            title=f["title"],
            journal=f["journal"],
            volume=f["volume"],
            issue=f["issue"],
            pages=f["pages"],
            year=f["year"],
            doi=meta.get("doi"),
            match_score=meta.get("match_score"),
            match_basis=meta.get("match_basis") or "",
            metadata_source="crossref" if meta.get("_enriched") else "xml",
            title_agreement=meta.get("title_agreement"),
            raw=meta.get("raw", ""),
        ))
    return rows
