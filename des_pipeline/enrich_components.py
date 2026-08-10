"""
Look up the individual DES components in public chemistry databases.

The review names its components in prose ("Choline chloride", "Urea") and gives no
structures at all, so identifiers and pure-component properties have to come from
elsewhere. Sources, in the order they are tried:

    PubChem (pubchempy)   CID, SMILES, InChI, formula, MW, H-bond donor/acceptor counts
    chemicals            CAS registry number
    PubChem PUG-View     melting point, boiling point, density
    NIST WebBook         same, opt-in via --nist (HTML scraping, fragile)

Lookup order matters. create_Sadeghi_graphdb.py went name -> CAS -> SMILES -> CID,
but `CAS_from_any` misses many of these names ("Choline bromide", "Choline nitrate").
A direct PubChem name search resolves those, so CAS is demoted to a best-effort extra.

Two cautions:

  * PubChem name matching is fuzzy. A near-miss returns the *wrong* compound
    silently, so `lookup_status` and `matched_synonym` are recorded for auditing.
    494 names include real typos ("1,2-proanediol", "CIEtMe3NCl") that will never
    resolve.
  * PUG-View property strings are messy -- "9 °F (NTP, 1992)", "4 - 10 °C", ranges,
    and occasionally a value for a different substance. The raw string is kept in
    `property_comments`; treat the parsed number as provisional.

Nothing here ever overwrites a value measured in the paper. Everything is cached in
data/components_cache.json, keyed by the name exactly as the paper writes it.
"""
import json
import re
import time
import warnings

import pandas as pd

from . import config
from .schema import ComponentRow

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pubchempy")

PUG_VIEW = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
HEADINGS = ["Melting Point", "Boiling Point", "Density"]
CAS_PATTERN = re.compile(r"^\d{2,7}-\d{2}-\d$")


# ---------- PubChem PUG-View JSON walking (from create_Sadeghi_graphdb.py) ----------
def find_heading(node, heading):
    """Depth-first search for a TOCHeading in PubChem's nested section tree."""
    if node.get("TOCHeading") == heading:
        return node
    for child in node.get("Section", []):
        found = find_heading(child, heading)
        if found is not None:
            return found
    return None


def extract_values(node):
    """Pull the display strings (and bare numbers) out of a PUG-View section."""
    values = []
    for info in node.get("Information", []):
        value = info.get("Value", {})
        for item in value.get("StringWithMarkup", []):
            values.append(item.get("String"))
        if "Number" in value:
            values += value["Number"]
    return [v for v in values if v is not None]


# ---------- property strings -> numbers ----------
# A number, or a range, followed by an optional unit: "133 - 135 °C", "-12.69 °C",
# "9 °F (NTP, 1992)", "1.24 g/cm3".
_RANGE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(-?\d+(?:\.\d+)?)\s*°?\s*([CFK])\b", re.I)
_SINGLE = re.compile(r"(-?\d+(?:\.\d+)?)\s*°?\s*([CFK])\b", re.I)
_DENSITY = re.compile(r"(\d+(?:\.\d+)?)\s*(?:g\s*/\s*cm3|g/cu\s*cm|g\s*cm-3|g/mL)", re.I)


def _to_celsius(value, unit):
    unit = unit.upper()
    if unit == "C":
        return value
    if unit == "F":
        return (value - 32.0) * 5.0 / 9.0
    if unit == "K":
        return value - 273.15
    return None


def parse_temperature(strings):
    """First parseable temperature in a list of PUG-View strings, in Celsius.

    Ranges are averaged, which is what the original process_property.py did.
    """
    for s in strings:
        if not isinstance(s, str):
            continue
        m = _RANGE.search(s)
        if m:
            midpoint = (float(m.group(1)) + float(m.group(2))) / 2.0
            return round(_to_celsius(midpoint, m.group(3)), 3)
        m = _SINGLE.search(s)
        if m:
            return round(_to_celsius(float(m.group(1)), m.group(2)), 3)
    return None


def parse_density(strings):
    """First density in g/cm3."""
    for s in strings:
        if isinstance(s, str):
            m = _DENSITY.search(s)
            if m:
                return float(m.group(1))
    return None


# ---------- the lookups ----------
def _pubchem_compound(name):
    import pubchempy as pcp

    matches = pcp.get_compounds(name, "name")
    return matches[0] if matches else None


def _cas_number(name, compound):
    """CAS from `chemicals`, falling back to a CAS-shaped PubChem synonym."""
    try:
        from chemicals import CAS_from_any

        return CAS_from_any(name)
    except Exception:
        pass
    try:
        for synonym in (compound.synonyms or [])[:40]:
            if CAS_PATTERN.match(synonym.strip()):
                return synonym.strip()
    except Exception:
        pass
    return None


def _pugview_properties(cid, session):
    """-> ({property: value}, [raw strings]) for melting point, boiling point, density."""
    values, comments = {}, []
    for heading in HEADINGS:
        try:
            response = session.get(PUG_VIEW.format(cid=cid),
                                   params={"heading": heading}, timeout=30)
            if response.status_code != 200:
                continue
            record = response.json()["Record"]
        except Exception:
            continue

        node = None
        for section in record.get("Section", []):
            node = find_heading(section, heading)
            if node is not None:
                break
        if node is None:
            continue

        strings = extract_values(node)
        comments += [f"{heading}: {s}" for s in strings[:2] if isinstance(s, str)]
        if heading == "Melting Point":
            values["melting_point_C"] = parse_temperature(strings)
        elif heading == "Boiling Point":
            values["boiling_point_C"] = parse_temperature(strings)
        elif heading == "Density":
            values["density_g_cm3"] = parse_density(strings)
        time.sleep(0.2)                      # PubChem asks for <= 5 requests/second
    return values, comments


def _nist_webbook(cas, session):
    """Scrape the NIST WebBook page for a CAS number. Opt-in; HTML is fragile."""
    from bs4 import BeautifulSoup

    try:
        response = session.get(
            "https://webbook.nist.gov/cgi/cbook.cgi",
            params={"ID": "C" + cas.replace("-", ""), "Units": "SI"}, timeout=30)
        text = BeautifulSoup(response.text, "html.parser").get_text(" ", strip=True)
    except Exception:
        return {}

    out = {}
    m = re.search(r"Normal melting point\s+([0-9.]+)\s*K", text)
    if m:
        out["melting_point_C"] = round(float(m.group(1)) - 273.15, 3)
    m = re.search(r"Density\s+([0-9.]+)\s*g/cm3", text)
    if m:
        out["density_g_cm3"] = float(m.group(1))
    return out


def lookup(name, session, use_nist=False):
    """Resolve one component name to a ComponentRow. Never raises."""
    try:
        compound = _pubchem_compound(name)
    except Exception as exc:
        return ComponentRow(name=name, lookup_status=f"error: {type(exc).__name__}")
    if compound is None:
        return ComponentRow(name=name, lookup_status="not_found")

    cas = _cas_number(name, compound)
    properties, comments = _pugview_properties(compound.cid, session)
    sources = ["pubchem"]

    if use_nist and cas:
        from_nist = _nist_webbook(cas, session)
        for key, value in from_nist.items():
            properties.setdefault(key, value)      # PubChem wins where both have data
        if from_nist:
            sources.append("nist")

    return ComponentRow(
        name=name,
        cid=compound.cid,
        cas=cas,
        smiles=getattr(compound, "connectivity_smiles", None) or compound.smiles,
        inchi=compound.inchi,
        inchikey=compound.inchikey,
        molecular_formula=compound.molecular_formula,
        molecular_weight=float(compound.molecular_weight) if compound.molecular_weight else None,
        h_bond_donor_count=compound.h_bond_donor_count,
        h_bond_acceptor_count=compound.h_bond_acceptor_count,
        melting_point_C=properties.get("melting_point_C"),
        boiling_point_C=properties.get("boiling_point_C"),
        density_g_cm3=properties.get("density_g_cm3"),
        property_comments=" | ".join(comments),
        sources=";".join(sources),
        lookup_status="ok",
    )


# ---------- cache + driver ----------
def distinct_components(path=None):
    """Every component name in the table, in first-appearance order."""
    df = pd.read_csv(path or config.TABLE_CSV)
    names, seen = [], set()
    for column in ("Component_1", "Component_2", "Component_3"):
        for value in df[column].dropna():
            name = str(value).strip()
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    return names


def _load_cache():
    if config.COMPONENT_CACHE.exists():
        return json.loads(config.COMPONENT_CACHE.read_text())
    return {}


def _save_cache(cache):
    config.COMPONENT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    config.COMPONENT_CACHE.write_text(json.dumps(cache, indent=2, ensure_ascii=False))


def enrich_all(names, limit=None, network=True, use_nist=False):
    """Look up every name, resuming from the cache. -> list[ComponentRow]."""
    import requests

    cache = _load_cache()
    todo = [n for n in names if n not in cache]
    if limit is not None:
        todo = todo[:limit]

    if not network:
        print(f"  {len(cache)} cached, {len(names) - len(cache)} unresolved (offline)")
    elif not todo:
        print(f"  all {len(names)} components already cached")
    else:
        print(f"  looking up {len(todo)} of {len(names)} components"
              f"{' (+NIST)' if use_nist else ''}")
        session = requests.Session()
        for i, name in enumerate(todo, 1):
            row = lookup(name, session, use_nist=use_nist)
            cache[name] = row.model_dump()
            if i % 25 == 0:
                _save_cache(cache)
                print(f"    {i}/{len(todo)} done (checkpointed)")
        _save_cache(cache)

    rows = [ComponentRow(**cache[n]) for n in names if n in cache]
    ok = sum(1 for r in rows if r.lookup_status == "ok")
    with_smiles = sum(1 for r in rows if r.smiles)
    print(f"  resolved {ok}/{len(rows)} ({with_smiles} with SMILES); "
          f"see lookup_status for the rest")
    return rows
