#!/usr/bin/env python3
"""
Fetch a paper's machine-readable full text into xml/.

    python fetch_paper.py 10.1016/j.rechem.2024.101378
    python fetch_paper.py S2211715624000742 --name Yeow_2024
    python fetch_paper.py --doi-file dois.txt

Routes tried, in order:

  1. Crossref  -- the publisher's own declared text-mining link. This is the general
                  mechanism: every publisher registers where the machine-readable
                  full text lives, so it works beyond Elsevier.
  2. Elsevier  -- the article API, with X-ELS-APIKey and, if present,
                  X-ELS-Insttoken.
  3. Unpaywall -- open-access locations, for anything not covered above.

A NOTE ON ELSEVIER ACCESS, measured 2026-08:

    An Elsevier API key on its own is NOT enough for full text. The article
    endpoint returns 403 AUTHENTICATION_ERROR ("Requestor configuration settings
    insufficient") for every article, including gold open-access ones, while the
    abstract and search endpoints work fine on the same key. Full text needs
    either an institutional token (ELSEVIER_INSTTOKEN in .env, obtained from
    Elsevier via your library) or a request originating from your institution's
    recognised IP range.

    Until then, download the XML by hand from ScienceDirect and save it to xml/.
    This script says so explicitly rather than failing obscurely.
"""
import argparse
import os
import re
import sys
from pathlib import Path

import requests

from des_pipeline import config

ELSEVIER_ARTICLE = "https://api.elsevier.com/content/article/{idtype}/{value}"
CROSSREF_WORK = "https://api.crossref.org/works/{doi}"
UNPAYWALL = "https://api.unpaywall.org/v2/{doi}"
PII_PATTERN = re.compile(r"^S?[0-9X\-()]{15,25}$", re.I)


def normalise_doi(value):
    """'https://doi.org/10.1016/X' -> '10.1016/x'. The one spelling used everywhere."""
    s = str(value or "").strip()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s, flags=re.I)
    s = re.sub(r"^doi:\s*", "", s, flags=re.I)
    return s.lower()


def _elsevier_headers():
    headers = {"Accept": "text/xml"}
    key = os.environ.get("ELSEVIER_API_KEY")
    token = os.environ.get("ELSEVIER_INSTTOKEN")
    if key:
        headers["X-ELS-APIKey"] = key
    if token:                       # what actually unlocks full text off-campus
        headers["X-ELS-Insttoken"] = token
    return headers


def _looks_like_full_text(content):
    """Elsevier answers an unentitled request with a ~2 kB metadata-only stub.

    It has the right root element and a <coredata> block, so status code and root
    tag both look fine -- the give-away is that there is no body.
    """
    if len(content) < 20000:
        return False
    lowered = content[:400000].lower()
    return b"<originaltext" in lowered or b"<body" in lowered or b"<ce:sections" in lowered


def _explain_elsevier(response):
    match = re.search(rb"<statusText>(.*?)</statusText>", response.content or b"")
    detail = match.group(1).decode("utf-8", "replace") if match else ""
    if response.status_code == 403 or "insufficient" in detail.lower():
        return (f"403 {detail or 'AUTHENTICATION_ERROR'}\n"
                "        Your API key has no full-text entitlement. Either set\n"
                "        ELSEVIER_INSTTOKEN in .env (ask your library to request one\n"
                "        from Elsevier), or run this from your institution's network.")
    return f"{response.status_code} {detail}"


# ---------- the routes ----------
def from_crossref(doi, session):
    """The publisher's declared text-mining link. -> (content, note)."""
    try:
        r = session.get(CROSSREF_WORK.format(doi=doi),
                        headers={"User-Agent": config.USER_AGENT}, timeout=45)
        r.raise_for_status()
        links = r.json()["message"].get("link") or []
    except Exception as exc:
        return None, f"lookup failed ({type(exc).__name__})"

    xml_links = [l for l in links
                 if "xml" in (l.get("content-type") or "")
                 and l.get("intended-application") == "text-mining"]
    if not xml_links:
        return None, f"no text-mining XML link declared ({len(links)} link(s) total)"

    for link in xml_links:
        url = link["URL"]
        headers = _elsevier_headers() if "elsevier.com" in url else {"Accept": "text/xml"}
        try:
            r = session.get(url, headers=headers, timeout=90)
        except Exception as exc:
            return None, f"{url} -> {type(exc).__name__}"
        if r.status_code == 200 and _looks_like_full_text(r.content):
            return r.content, f"via {url}"
        if r.status_code == 200:
            return None, (f"{url} returned {len(r.content)} bytes of metadata only "
                          f"(no body) -- this is the unentitled stub")
        return None, f"{url} -> {_explain_elsevier(r)}"
    return None, "no usable link"


def from_elsevier(identifier, session):
    """The Elsevier article API directly, by DOI or PII."""
    if not os.environ.get("ELSEVIER_API_KEY"):
        return None, "ELSEVIER_API_KEY not set in .env"
    idtype = "pii" if PII_PATTERN.match(identifier.replace("-", "")) else "doi"
    url = ELSEVIER_ARTICLE.format(idtype=idtype, value=identifier)
    try:
        r = session.get(url, headers=_elsevier_headers(),
                        params={"httpAccept": "text/xml", "view": "FULL"}, timeout=90)
    except Exception as exc:
        return None, f"{type(exc).__name__}"
    if r.status_code == 200 and _looks_like_full_text(r.content):
        return r.content, f"via {idtype} {identifier}"
    if r.status_code == 200:
        return None, f"{len(r.content)} bytes of metadata only (no body)"
    return None, _explain_elsevier(r)


def from_unpaywall(doi, session):
    """Open-access locations. Usually a landing page or PDF rather than XML."""
    try:
        r = session.get(UNPAYWALL.format(doi=doi), params={"email": config.MAILTO}, timeout=45)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        return None, f"lookup failed ({type(exc).__name__})"
    if not data.get("is_oa"):
        return None, "not open access"

    for location in data.get("oa_locations") or []:
        url = location.get("url_for_pdf") or location.get("url")
        if not url or not url.lower().endswith(".xml"):
            continue
        try:
            r = session.get(url, timeout=90)
        except Exception:
            continue
        if r.status_code == 200 and _looks_like_full_text(r.content):
            return r.content, f"via {url}"
    return None, (f"{data.get('oa_status')} OA, but no machine-readable XML "
                  f"({len(data.get('oa_locations') or [])} location(s), landing pages only)")


# ---------- driver ----------
def fetch(identifier, name=None, out_dir=None):
    """Try every route. -> the written Path, or None."""
    out_dir = Path(out_dir or config.XML_FILES)
    session = requests.Session()
    is_doi = "/" in identifier
    doi = normalise_doi(identifier) if is_doi else None

    print(f"fetching {identifier}")
    routes = []
    if doi:
        routes.append(("crossref ", lambda: from_crossref(doi, session)))
    routes.append(("elsevier ", lambda: from_elsevier(identifier, session)))
    if doi:
        routes.append(("unpaywall", lambda: from_unpaywall(doi, session)))

    for label, route in routes:
        content, note = route()
        if content:
            print(f"  {label}    OK, {len(content)} bytes {note}")
            slug = name or re.sub(r"[^A-Za-z0-9_.-]", "_", identifier).strip("_")
            path = out_dir / f"{slug}.xml"
            out_dir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            print(f"\nwrote {path}")
            print("run:  python run_pipeline.py --steps route")
            return path
        print(f"  {label}    {note}")

    print("\nno machine-readable full text retrieved.")
    print(f"Download it by hand from the publisher and save it as "
          f"{out_dir}/<name>.xml -- the pipeline only needs the file to exist.")
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("identifier", nargs="?", help="a DOI or an Elsevier PII")
    parser.add_argument("--name", help="filename stem to save as (default: from the identifier)")
    parser.add_argument("--doi-file", help="a file of DOIs, one per line")
    parser.add_argument("--out", default=None, help="output directory (default: xml/)")
    args = parser.parse_args(argv)

    if not args.identifier and not args.doi_file:
        parser.error("give a DOI/PII, or --doi-file")

    identifiers = []
    if args.doi_file:
        identifiers += [l.strip() for l in Path(args.doi_file).read_text().splitlines()
                        if l.strip() and not l.startswith("#")]
    if args.identifier:
        identifiers.append(args.identifier)

    ok = 0
    for identifier in identifiers:
        ok += bool(fetch(identifier, name=args.name if len(identifiers) == 1 else None,
                         out_dir=args.out))
        print()
    print(f"{ok}/{len(identifiers)} retrieved")
    return 0 if ok == len(identifiers) else 1


if __name__ == "__main__":
    sys.exit(main())
