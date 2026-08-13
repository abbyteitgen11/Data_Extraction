"""
process_paper(path): one coherent, multi-path extractor for a DES paper.

Routing is by DOCUMENT STRUCTURE (deterministic), not by asking an LLM:
  <table>        -> table parser        (reliable; never touches references)
  body <section> -> LLM extractor       (prose only; you pass the function in)
  <figure>       -> flagged for human   (plot digitization later)
  <bib-reference>-> reference metadata   -> DOI resolver (Crossref, separate step)

Every record carries provenance (source, locus) and the row's resolved references,
so a data point can be traced back to the primary paper it came from.
"""
import re
import requests
from lxml import etree

# ---------- shared helpers ----------
TEMP_MAP = {"a": 40, "b": 20, "c": 60, "d": 45, "e": 30, "f": 35, "g": 50, "h": 55}
DEFAULT_TEMP = 25
DASH = {"–", "—", "-", "−", ""}
PROP_COLUMNS = [(3, "Melting_point", "C"), (4, "Density", "g*cm^-3"),
                (5, "Viscosity", "mPa*s"), (6, "Conductivity", "mS*cm^-1"),
                (7, "Surface_tension", "mN*m^-1"), (8, "Refractive_index", "")]


def load_root(path):
    root = etree.parse(path).getroot()
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def _text(el):
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


# ---------- 1. reference list: number -> structured metadata ----------
def build_reference_map(root):
    refs = {}
    for r in root.findall(".//bib-reference"):
        label = r.findtext("label", "").strip("[]")
        if not label.isdigit():
            continue
        authors = []
        for a in r.findall(".//author"):
            gn = a.findtext("given-name", "") or ""
            sn = a.findtext("surname", "") or ""
            authors.append(f"{gn} {sn}".strip())
        titles = [_text(t) for t in r.findall(".//maintitle")]  # [article, journal]
        refs[int(label)] = {
            "num": int(label),
            "authors": authors,
            "title": titles[0] if titles else "",
            "journal": titles[1] if len(titles) > 1 else "",
            "year": r.findtext(".//date", "").strip(),
            "raw": _text(r),
            "doi": None,          # filled later by resolve_dois()
        }
    return refs


def expand_ref_field(s):
    """'1,26-28' -> [1, 26, 27, 28]"""
    out = []
    for part in str(s).replace("–", "-").replace("—", "-").split(","):
        part = part.strip()
        if "-" in part:
            try:
                lo, hi = (int(x) for x in part.split("-", 1))
                out += list(range(lo, hi + 1))
            except ValueError:
                pass
        elif part.isdigit():
            out.append(int(part))
    return out


# ---------- 2. table path (reliable; reuses the Table 2 parser) ----------
def _prop_value(entry):
    text = _text(entry)
    if text in DASH:
        return None, None
    sups = [(s.text or "").strip() for s in entry.iter() if s.tag == "sup"]
    marker = next((s for s in sups if s in TEMP_MAP), None)
    temp = TEMP_MAP[marker] if marker else DEFAULT_TEMP
    if marker and text[:1] == marker:
        text = text[1:]
    t = text.replace(",", "").replace("−", "-").replace("–", "-")
    try:
        return float(t), temp
    except ValueError:
        return None, None


def parse_table2(root, reference_map):
    table = root.findall(".//table")[1]
    rows = table.findall(".//row")
    records = []
    for i, row in enumerate(rows):
        e = row.findall("entry")
        if len(e) != 10:
            continue
        hba = _text(e[0])
        if hba in {"", "HBAs"} or _text(e[2]) == "Molar ratio":
            continue
        comps = [hba] + [c.strip() for c in _text(e[1]).split("/")]
        ref_field = _text(e[9]).strip("[]").replace("–", "-")
        ref_nums = expand_ref_field(ref_field)
        row_refs = [reference_map[n] for n in ref_nums if n in reference_map]
        for idx, prop, unit in PROP_COLUMNS:
            val, temp = _prop_value(e[idx])
            if val is None:
                continue
            records.append({
                "components": comps,
                "ratio": _text(e[2]),
                "property": prop,
                "value": val,
                "unit": unit,
                "temperature_C": temp,
                "source": "Table 2",            # provenance: where in the paper
                "locus": f"row {i}",
                "ref_numbers": ref_nums,          # provenance: which primary sources
                "references": row_refs,
            })
    return records


# ---------- 3. text path (LLM; you supply the function) ----------
def get_text_sections(root):
    """Body paragraphs, excluding tables and the bibliography."""
    body = root.find(".//body") if root.find(".//body") is not None else root
    out = []
    for sec in body.findall(".//section"):
        if sec.find(".//table") is not None:
            continue
        txt = _text(sec)
        if len(txt) > 200:
            out.append(txt)
    return out


# ---------- 4. figure path (flag for human) ----------
def get_figures(root):
    return [{"caption": _text(f), "source": f.get("id") or "figure", "status": "needs_human"}
            for f in root.findall(".//figure")]


# ---------- 5. DOI resolution (separate enrichment; needs network) ----------
def doi_from_reference(meta, mailto="abigail.teitgen@csic.es"):
    query = f"{meta['title']} {meta['journal']} {meta['year']}".strip()
    if not query:
        return None, 0.0
    try:
        r = requests.get("https://api.crossref.org/works",
                         params={"query.bibliographic": query, "rows": 1},
                         headers={"User-Agent": f"DES-KG ({mailto})"}, timeout=20)
        r.raise_for_status()                       # turn HTTP errors into exceptions
        items = r.json().get("message", {}).get("items", [])
    except (requests.RequestException, ValueError) as exc:
        print(f"  crossref lookup failed for ref {meta['num']}: {exc}")
        return None, 0.0                           # leave DOI as None, keep going
    if not items:
        return None, 0.0
    return items[0].get("DOI"), items[0].get("score", 0.0)


def resolve_dois(reference_map, min_score=40, mailto="abigail.teitgen@csic.es"):
    """Fill in DOIs for every reference (run where Crossref is reachable)."""
    for meta in reference_map.values():
        doi, score = doi_from_reference(meta, mailto)
        meta["doi"] = doi if score >= min_score else None
        meta["match_score"] = score
    return reference_map


# ---------- orchestration ----------
def process_paper(path, extract_text=None, do_resolve_dois=False):
    root = load_root(path)
    reference_map = build_reference_map(root)
    if do_resolve_dois:
        resolve_dois(reference_map)

    records = parse_table2(root, reference_map)          # reliable table path
    if extract_text is not None:                         # optional LLM prose path
        for section in get_text_sections(root):
            records += extract_text(section)             # your Ollama/Sonnet function
    figures = get_figures(root)                          # flagged for human

    return {"measurements": records, "figures": figures,
            "reference_map": reference_map}


if __name__ == "__main__":
    result = process_paper("SadeghiDESReview.xml", extract_text=None, do_resolve_dois=True)   # table + refs only (offline)
    m, refs = result["measurements"], result["reference_map"]

    print(f"references parsed: {len(refs)}")
    print(f"table measurements: {len(m)}")
    joined = sum(1 for r in m if r["references"])
    print(f"measurements linked to >=1 reference: {joined}/{len(m)}")
    print(f"figures flagged for human: {len(result['figures'])}")

    print("\n--- example measurement WITH its resolved provenance ---")
    ex = next(r for r in m if r["references"])
    print(f"  {ex['components']}  {ex['ratio']}  {ex['property']} = "
          f"{ex['value']} {ex['unit']} @ {ex['temperature_C']}C   [{ex['source']}, {ex['locus']}]")
    print(f"  cited refs: {ex['ref_numbers']}")
    for rr in ex["references"][:3]:
        auth = (rr["authors"][0] + " et al.") if rr["authors"] else "?"
        print(f"    [{rr['num']}] {auth} ({rr['year']}) — {rr['title'][:60]}  DOI={rr['doi']}")