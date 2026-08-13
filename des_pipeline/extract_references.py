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

from . import config, xml_utils
from .schema import ReferenceRow


# ---------- 1. read the bibliography out of the XML ----------
def parse_bibliography(reference_elements):
    """-> {reference number: metadata dict}. Entries without a numeric label are skipped."""
    refs = {}
    for r in reference_elements:
        label = (r.findtext("label", "") or "").strip("[]")
        if not label.isdigit():
            continue
        authors = []
        for a in r.findall(".//author"):
            given = a.findtext("given-name", "") or ""
            surname = a.findtext("surname", "") or ""
            authors.append(f"{given} {surname}".strip())
        # Elsevier's struct-bib nests two <maintitle>s: the article, then the journal.
        titles = [xml_utils.text(t) for t in r.findall(".//maintitle")]
        refs[int(label)] = {
            "num": int(label),
            "authors": authors,
            "title": titles[0] if titles else "",
            "journal": titles[1] if len(titles) > 1 else "",
            "year": (r.findtext(".//date", "") or "").strip(),
            "raw": xml_utils.text(r),
            "doi": None,
        }
    return refs


# ---------- cache ----------
def load_cache(path=None):
    path = path or config.REFERENCE_CACHE
    if not path.exists():
        return {}
    return {int(k): v for k, v in json.loads(path.read_text()).items()}


def save_cache(cache, path=None):
    path = path or config.REFERENCE_CACHE
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))


def apply_cache(reference_map, cache):
    """Merge cached lookups into a freshly-parsed reference map."""
    for num, cached in cache.items():
        if num in reference_map:
            reference_map[num] = {**reference_map[num], **cached}
    return reference_map


# ---------- 2. Crossref: find the DOI ----------
def crossref_search(meta, session, max_retries=5):
    """Bibliographic search -> (doi, match_score). Backs off politely on 429."""
    query = f"{meta['title']} {meta['journal']} {meta['year']}".strip()
    if not query:
        return None, 0.0
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
                return None, 0.0
            return items[0].get("DOI"), items[0].get("score", 0.0)
        except (requests.RequestException, ValueError) as exc:
            print(f"    lookup error ref {meta['num']} (attempt {attempt + 1}): {exc}")
            time.sleep(wait)
            wait *= 2
    return None, 0.0


# ---------- 3. Crossref: full metadata for a known DOI ----------
def crossref_metadata(doi, session, max_retries=3):
    """GET works/{doi} -> the cr_* fields. Exact lookup, so no scoring needed."""
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
        return {
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


def resolve_all(reference_map, cache=None, network=True):
    """Fill in DOIs. Entries already flagged _resolved are never re-requested."""
    cache = cache if cache is not None else load_cache()
    reference_map = apply_cache(reference_map, cache)
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
        doi, score = crossref_search(meta, session)
        meta["doi"] = doi if score >= config.MIN_MATCH_SCORE else None
        meta["match_score"] = score
        meta["_resolved"] = True
        cache[num] = meta
        time.sleep(0.05)
        if i % 25 == 0:
            save_cache(cache)
            print(f"    {i}/{len(todo)} resolved (checkpointed)")

    save_cache({**cache, **reference_map})
    have = sum(1 for m in reference_map.values() if m.get("doi"))
    print(f"  resolve: {have}/{len(reference_map)} have a DOI (score >= {config.MIN_MATCH_SCORE})")
    return reference_map


def enrich_all(reference_map, cache=None, network=True):
    """Add volume/issue/pages/authoritative authors for every resolved DOI."""
    cache = cache if cache is not None else load_cache()
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
    for i, num in enumerate(sorted(todo), 1):
        meta = reference_map[num]
        extra = crossref_metadata(meta["doi"], session)
        if extra:
            meta.update(extra)
            meta["title_agreement"] = _title_agreement(meta.get("title", ""), extra["cr_title"])
        meta["_enriched"] = True
        cache[num] = meta
        time.sleep(0.05)
        if i % 25 == 0:
            save_cache(cache)
            print(f"    {i}/{len(todo)} enriched (checkpointed)")

    save_cache({**cache, **reference_map})
    with_vol = sum(1 for m in reference_map.values() if m.get("volume"))
    print(f"  enrich:  {with_vol}/{len(reference_map)} now have a volume")
    return reference_map


def enrich_review(review, network=True, cache=None):
    """Top up the review's own metadata from Crossref.

    Elsevier's <coredata> carries no issue number, so the paper we are extracting
    from goes through the same lookup as every paper it cites. Cached under key 0,
    which no real reference number uses.
    """
    cache = cache if cache is not None else load_cache()
    cached = cache.get(0)
    if cached:
        return {**review, **cached}
    if not network or not review.get("doi"):
        return review

    extra = crossref_metadata(review["doi"], requests.Session())
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
    save_cache(cache)
    return {**review, **merged}


# ---------- the shape everything downstream reads ----------
def reference_fields(meta):
    """Normalise one reference to flat strings, preferring Crossref over the XML.

    Crossref returns HTML-escaped text, so 34 of these journals arrive as
    "Chemical Engineering &amp; Technology". Unescaping here rather than at fetch
    time fixes what is already in reference_map.json without re-querying.
    """
    authors = meta.get("cr_authors") or "; ".join(meta.get("authors") or [])
    fields = {
        "key": paper_key(meta),
        "doi": meta.get("doi") or "",
        "match_score": meta.get("match_score"),
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


def paper_key(meta, review_doi=None):
    """Stable node identity: the DOI when Crossref matched one, else '<review>#refN'.

    A reference Crossref could not resolve is still a real paper that real
    measurements came from, and it still needs to be exactly one node. Merging on
    `doi` cannot do that -- Neo4j uniqueness constraints ignore nulls, so every
    DOI-less reference would collapse into one another silently.
    """
    return meta.get("doi") or f"{review_doi or config.REVIEW_DOI}#ref{meta['num']}"


def sources(ref_numbers, reference_map):
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
        fields = reference_fields(meta)
        for k in keys:
            collected[k].append(str(fields[k]).replace(config.SOURCE_SEP, "/"))
    return {k: config.SOURCE_SEP.join(v) for k, v in collected.items()}


def to_rows(reference_map):
    """-> list[ReferenceRow], ordered by reference number."""
    rows = []
    for meta in sorted(reference_map.values(), key=lambda m: m["num"]):
        f = reference_fields(meta)
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
            metadata_source="crossref" if meta.get("_enriched") else "xml",
            title_agreement=meta.get("title_agreement"),
            raw=meta.get("raw", ""),
        ))
    return rows
