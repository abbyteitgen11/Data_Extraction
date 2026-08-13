"""
Resolve each reference to a DOI via Crossref (polite, cached, rate-limit-safe),
then write:
  - reference_map.json      : number -> {authors, title, year, doi, match_score, ...}
  - references.csv          : one row per reference (num, authors, year, doi, ...)
  - table2_with_dois.csv    : the measurement table, each row carrying its
                              primary-source ref numbers + DOIs (what you had
                              in table2_full.csv, plus the resolved provenance)

Run on a networked machine:  python3 resolve_references.py
Re-running is cheap: already-resolved references are loaded from the cache and
never re-requested, so you can resume safely and you won't get rate-limited twice.
"""
import json
import time
from pathlib import Path

import pandas as pd
import requests

from process_paper import process_paper, expand_ref_field

XML = "SadeghiDESReview.xml"
REVIEW_DOI = "10.1016/j.molliq.2023.121899"
MAILTO = "abigail.teitgen@csic.es"          # <-- put your real email here (polite pool)
CACHE = Path("reference_map.json")
MIN_SCORE = 40                          # Crossref match confidence threshold


# ---------- one Crossref lookup, with retry/backoff on 429 ----------
def crossref_doi(meta, session, max_retries=5):
    query = f"{meta['title']} {meta['journal']} {meta['year']}".strip()
    if not query:
        return None, 0.0
    params = {"query.bibliographic": query, "rows": 1, "mailto": MAILTO}
    headers = {"User-Agent": f"DES-KG/1.0 (mailto:{MAILTO})"}
    wait = 2.0
    for attempt in range(max_retries):
        try:
            r = session.get("https://api.crossref.org/works",
                            params=params, headers=headers, timeout=30)
            if r.status_code == 429:                      # rate limited -> back off
                retry_after = float(r.headers.get("Retry-After", wait))
                print(f"    429 on ref {meta['num']}; waiting {retry_after:.0f}s")
                time.sleep(retry_after)
                wait *= 2                                  # exponential backoff
                continue
            r.raise_for_status()
            items = r.json().get("message", {}).get("items", [])
            if not items:
                return None, 0.0
            return items[0].get("DOI"), items[0].get("score", 0.0)
        except (requests.RequestException, ValueError) as exc:
            print(f"    lookup error ref {meta['num']} (attempt {attempt+1}): {exc}")
            time.sleep(wait)
            wait *= 2
    return None, 0.0                                       # gave up; leave unresolved


# ---------- resolve the whole map, using the cache ----------
def resolve_all(reference_map):
    # load anything already resolved
    cached = {}
    if CACHE.exists():
        cached = {int(k): v for k, v in json.loads(CACHE.read_text()).items()}
        print(f"loaded {len(cached)} cached references")

    session = requests.Session()
    to_do = [n for n in reference_map
             if not (cached.get(n) or {}).get("_resolved")]
    print(f"resolving {len(to_do)} of {len(reference_map)} references via Crossref")

    for i, num in enumerate(sorted(to_do), 1):
        meta = reference_map[num]
        doi, score = crossref_doi(meta, session)
        meta["doi"] = doi if score >= MIN_SCORE else None
        meta["match_score"] = score
        meta["_resolved"] = True
        cached[num] = meta
        time.sleep(0.05)                                   # polite pacing
        if i % 25 == 0:                                    # checkpoint the cache
            CACHE.write_text(json.dumps(cached, indent=2, ensure_ascii=False))
            print(f"  {i}/{len(to_do)} done (checkpointed)")

    # merge cache back into the live map and save
    for num, meta in cached.items():
        if num in reference_map:
            reference_map[num] = meta
    CACHE.write_text(json.dumps(reference_map, indent=2, ensure_ascii=False))
    resolved = sum(1 for m in reference_map.values() if m.get("doi"))
    print(f"done: {resolved}/{len(reference_map)} references have a DOI "
          f"(score >= {MIN_SCORE})")
    return reference_map


# ---------- write the CSV outputs ----------
def write_references_csv(reference_map, path="references.csv"):
    rows = [{
        "ref_number": m["num"],
        "authors": "; ".join(m["authors"]),
        "title": m["title"],
        "journal": m["journal"],
        "year": m["year"],
        "doi": m.get("doi"),
        "match_score": m.get("match_score"),
    } for m in sorted(reference_map.values(), key=lambda x: x["num"])]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"wrote {path}  ({len(rows)} references)")


def write_measurements_csv(measurements, reference_map, path="table2_with_dois.csv"):
    rows = []
    for m in measurements:
        nums = m["ref_numbers"]
        dois = [reference_map[n]["doi"] for n in nums
                if n in reference_map and reference_map[n].get("doi")]
        comps = (m["components"] + [None, None, None])[:3]
        rows.append({
            "Component_1": comps[0], "Component_2": comps[1], "Component_3": comps[2],
            "Ratio_raw": m["ratio"],
            "Property": m["property"], "Value": m["value"],
            "Unit": m["unit"], "Temperature_C": m["temperature_C"],
            "review_doi": REVIEW_DOI,             # the paper we extracted from
            "source_ref_numbers": ",".join(map(str, nums)),
            "source_dois": ";".join(dois),        # the PRIMARY papers the data came from
            "locus": m["locus"],
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"wrote {path}  ({len(rows)} measurements)")


if __name__ == "__main__":
    result = process_paper(XML)                    # offline: table + reference metadata
    reference_map = resolve_all(result["reference_map"])
    write_references_csv(reference_map)
    write_measurements_csv(result["measurements"], reference_map)