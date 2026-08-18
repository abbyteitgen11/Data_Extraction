"""
Look up the individual DES components in public chemistry databases.

The review names its components in prose ("Choline chloride", "Urea") and gives no
structures at all, so identifiers and pure-component properties have to come from
elsewhere. Sources, in the order they are tried:

    PubChem (pubchempy)   CID, SMILES, InChI, formula, MW, H-bond donor/acceptor counts
    chemicals            CAS registry number
    PubChem PUG-View     melting point, boiling point, density
    NIST WebBook         phase-change data, ON BY DEFAULT (--no-nist skips it)

Lookup order matters. create_Sadeghi_graphdb.py went name -> CAS -> SMILES -> CID,
but `CAS_from_any` misses many of these names ("Choline bromide", "Choline nitrate").
A direct PubChem name search resolves those, so CAS is demoted to a best-effort extra.

Two cautions:

  * PubChem name matching is fuzzy. A near-miss returns the *wrong* compound
    silently, so `lookup_status` and `matched_synonym` are recorded for auditing.
    494 names include real typos ("1,2-proanediol", "CIEtMe3NCl") that will never
    resolve.
  * PUG-View property strings are messy -- "9 °F (NTP, 1992)", "4 - 10 °C", ranges,
    and occasionally a value for a different substance. The scalar parsers here take
    the FIRST parseable number and drop the rest, which is fine as a single best
    guess but loses real data: 208 of the 256 components with text report the same
    property more than once. component_properties.py reads the full set, with the
    source PubChem attached to each; this module keeps the scalar it always had.

Nothing here ever overwrites a value measured in the paper. Everything is cached in
data/components_cache.json, keyed by the name exactly as the paper writes it.
"""
import json
import re
import time
import warnings
from collections import Counter

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


def extract_entries(node, references):
    """Same strings, but keeping the provenance extract_values() throws away.

    PubChem's Record carries a Reference list keyed by ReferenceNumber, in the very
    JSON we already fetched, so "which database said this" is structural. Nothing
    downstream ever has to guess at an attribution.
    """
    out = []
    for info in node.get("Information", []):
        data_source, source_record = references.get(info.get("ReferenceNumber"), ("", ""))
        for item in info.get("Value", {}).get("StringWithMarkup", []):
            if item.get("String"):
                out.append({"string": item["String"],
                            "data_source": data_source,
                            "source_record": source_record})
    return out


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
    """-> ({property: value}, [comment strings], [entries]).

    The first two are the original scalar path, unchanged. `entries` is the new one:
    every string with its source attached, untruncated, for the LLM pass to read.
    PubChem returns ~11.5 strings per component and the scalar path looks at the
    first parseable one, so this is where the other ~80% finally goes.
    """
    values, comments, entries = {}, [], []
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

        references = {r.get("ReferenceNumber"): (r.get("SourceName") or "",
                                                 r.get("Name") or "")
                      for r in record.get("Reference", [])}
        record_title = record.get("RecordTitle", "")
        entries += [{**e, "heading": heading, "property": config.PUGVIEW_PROPERTIES[heading],
                     "record_title": record_title, "source_db": "pubchem"}
                    for e in extract_entries(node, references)]

        strings = extract_values(node)
        comments += [f"{heading}: {s}" for s in strings[:2] if isinstance(s, str)]
        if heading == "Melting Point":
            values["melting_point_C"] = parse_temperature(strings)
        elif heading == "Boiling Point":
            values["boiling_point_C"] = parse_temperature(strings)
        elif heading == "Density":
            values["density_g_cm3"] = parse_density(strings)
        time.sleep(0.2)                      # PubChem asks for <= 5 requests/second
    return values, comments, entries


# NIST WebBook phase-change data lives on the Mask=4 sub-page and is written
# "T fus 261. +/- 2. K" / "T boil 470.5 K". The landing page carries neither, which
# is why the previous "Normal melting point ... K" regex matched on 0 of 499
# components -- the scrape has never once returned anything.
_NIST_FUS = re.compile(r"T\s*fus\s+([0-9.]+)")
_NIST_BOIL = re.compile(r"T\s*boil\s+([0-9.]+)")


def _nist_webbook(cas, session):
    """Phase-change data from the NIST WebBook. -> ({property: C}, [entries]).

    Values are returned as entries too, so a NIST/PubChem disagreement is visible in
    the long CSV rather than being silently resolved by whichever ran first.
    """
    from bs4 import BeautifulSoup

    try:
        response = session.get(
            "https://webbook.nist.gov/cgi/cbook.cgi",
            params={"ID": "C" + cas.replace("-", ""), "Units": "SI", "Mask": "4"},
            timeout=30)
        text = BeautifulSoup(response.text, "html.parser").get_text(" ", strip=True)
    except Exception:
        return {}, []

    values, entries = {}, []
    for pattern, prop, field in ((_NIST_FUS, "Melting_point", "melting_point_C"),
                                 (_NIST_BOIL, "Boiling_point", "boiling_point_C")):
        m = pattern.search(text)
        if not m:
            continue
        kelvin = float(m.group(1))
        values[field] = round(kelvin - 273.15, 3)
        entries.append({
            "string": m.group(0),
            "data_source": "NIST WebBook",
            "source_record": cas,
            "heading": prop,
            "property": prop,
            "record_title": "",
            "source_db": "nist",
            "value_as_written": kelvin,
            "unit_as_written": "K",
        })
    return values, entries


def lookup(name, session, use_nist=True):
    """Resolve one component name to a ComponentRow. Never raises.

    -> (ComponentRow, entries) where entries are the raw source lines with their
    provenance, for the LLM pass. The row itself stays scalar because it is a CSV.
    """
    try:
        compound = _pubchem_compound(name)
    except Exception as exc:
        return ComponentRow(name=name, lookup_status=f"error: {type(exc).__name__}"), []
    if compound is None:
        return ComponentRow(name=name, lookup_status="not_found"), []

    cas = _cas_number(name, compound)
    properties, comments, entries = _pugview_properties(compound.cid, session)
    sources = ["pubchem"]

    if use_nist and cas:
        from_nist, nist_entries = _nist_webbook(cas, session)
        for key, value in from_nist.items():
            properties.setdefault(key, value)      # PubChem wins where both have data
        entries += nist_entries
        if from_nist:
            sources.append("nist")

    row = ComponentRow(
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
        # Tabular descriptors, free on the Compound we already fetched. tpsa comes
        # back as an int for some compounds, hence the cast.
        tpsa=float(compound.tpsa) if compound.tpsa is not None else None,
        rotatable_bond_count=compound.rotatable_bond_count,
        formal_charge=compound.charge,
        xlogp=float(compound.xlogp) if compound.xlogp is not None else None,
        complexity=float(compound.complexity) if compound.complexity is not None else None,
        melting_point_C=properties.get("melting_point_C"),
        boiling_point_C=properties.get("boiling_point_C"),
        density_g_cm3=properties.get("density_g_cm3"),
        property_comments=" | ".join(comments),
        sources=";".join(sources),
        lookup_status="ok",
    )
    return row, entries


# ---------- cache + driver ----------
def distinct_components(path=None):
    """Every component name across every paper's tables, first-appearance order."""
    from . import store

    rows = store.read_all("mixtures")
    names, seen = [], set()
    for column in ("Component_1", "Component_2", "Component_3"):
        for row in rows:
            name = str(row.get(column) or "").strip()
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    return names


# ---------- resolving prose component names to Table 2 names ----------
#
# The prose writes "ChCl:EG(1:2)"; Table 2 writes "Choline chloride" / "Ethylene
# glycol". The paper defines none of its abbreviations, so this has to be a lookup,
# never a guess. Measured: qwen3 expands roughly half of them wrongly, and the wrong
# answers are plausible -- it reads TBAB as tetraETHYLammonium bromide when the paper
# means tetraBUTYL. Four such wrong answers are themselves real Table 2 entries, so
# matching against the table vocabulary does not catch them; it legitimises them.
#
# Hence: exact / alias / alphanumeric lookup only. No fuzzy matching and no
# initialism matching -- "ChAc" is both Chloroacetic acid and Choline acetate, "AA"
# is both Acetic acid and Aconitic acid. A name we cannot resolve is flagged and
# kept out of the graph. Missing data is visible; wrong chemistry is not.

_TRAILING_ABBR = re.compile(r"\s*\([A-Za-z0-9\[\]\-]{1,12}\)\s*$")


def _norm_key(name):
    """Match key: lowercase, whitespace-collapsed, trailing '(ABBR)' removed."""
    s = re.sub(r"\s+", " ", str(name or "")).strip()
    s = _TRAILING_ABBR.sub("", s)
    return s.lower().strip()


def _alnum_key(name):
    """Looser key ignoring spaces and punctuation: 'Diethanol amine' == 'Diethanolamine'."""
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def load_alias_records(path=None):
    """abbreviation -> the full record, for the harvester and for auditing."""
    path = path or config.COMPONENT_ALIASES
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    out = {}
    for key, value in raw.items():
        if key.startswith("__"):
            continue
        out[key.strip()] = value if isinstance(value, dict) else {"name": value}
    return out


def load_aliases(path=None):
    """abbreviation -> full name, matched case-insensitively.

    A value may be a plain string (quick hand-edit) or a record carrying its
    provenance. A record whose name is null is skipped deliberately -- that is how an
    abbreviation the paper uses for two different chemicals stays unresolved instead
    of silently picking one.
    """
    return {abbr.lower(): record["name"]
            for abbr, record in load_alias_records(path).items()
            if record.get("name")}


def component_index(path=None, alias_path=None):
    """Build the lookup used by resolve_component().

    Table 2 spells some components more than one way ('Levulinic acid' /
    'Levulinic Acid'), so the canonical form is the MOST FREQUENT spelling. That
    way a resolved prose name lands on the Component node that already has the
    most edges, rather than creating a second variant.
    """
    from . import store

    df = pd.DataFrame(store.read_all("mixtures"))
    counts = Counter()
    for column in ("Component_1", "Component_2", "Component_3"):
        counts.update(str(v).strip() for v in df[column].dropna() if str(v).strip())

    canonical = {}
    for name, n in counts.most_common():           # most frequent first wins
        canonical.setdefault(_norm_key(name), name)
    alnum = {}
    for name, n in counts.most_common():
        alnum.setdefault(_alnum_key(name), name)

    aliases = load_aliases(alias_path)
    for abbr, full in aliases.items():
        if _norm_key(full) not in canonical:
            print(f"    note: alias {abbr!r} -> {full!r} is not a Table 2 component "
                  f"(fine if it is prose-only, but check the spelling)")
    # Prose-only components PubChem confirmed on an earlier run, so a second run
    # resolves them without touching the network.
    prose_vocabulary = {}
    for name, row in _load_cache().items():
        if row.get("lookup_status") == "ok" and _norm_key(name) not in canonical:
            prose_vocabulary[_norm_key(name)] = name

    return {"canonical": canonical, "alnum": alnum, "aliases": aliases,
            "vocabulary": set(counts), "prose_vocabulary": prose_vocabulary}


# Words that mark a compound CLASS rather than a compound. PubChem would either
# miss these or, worse, match them to an unrelated single substance.
_CLASS_WORDS = re.compile(
    r"\b(acids|salts?|halides|bromides|chlorides|iodides|compounds?|derivatives|"
    r"analogues|mixtures?|species|based)\b", re.I)


def looks_like_chemical(name):
    """Cheap filter deciding whether a name is worth a PubChem lookup.

    Keeps '1,5-pentanediol'; rejects 'Amino acids', 'Choline salt',
    'Tetraalkyl ammonium halides', 'HBD' and 'RCl'.
    """
    n = str(name or "").strip()
    if len(n) < 4 or n.isupper():          # too short, or a bare abbreviation
        return False
    if _CLASS_WORDS.search(n):
        return False
    return bool(re.search(r"[a-z]{3}", n))


def resolve_component(name, index, allow_lookup=False):
    """One name -> (canonical name or None, how it was resolved).

    The alias file is consulted FIRST, on the raw string. It is human-verified and
    the model's expansion is not, so where they disagree the human wins.

    With allow_lookup, a name that matches nothing in Table 2 but that PubChem
    recognises is accepted as a prose-only component -- that is how a genuinely new
    chemical like '1,5-pentanediol' gets into the graph. PubChem is the arbiter, so
    this is still a lookup rather than a guess. Results are cached on disk.
    """
    raw = str(name or "").strip()
    if not raw:
        return None, "empty"

    def canonicalise(value):
        """Redirect an alias onto Table 2's most-frequent spelling of that name.

        Table 2 spells several components two ways -- 'Tetraethylammnoium bromide'
        (8 uses, the paper's own typo) vs 'Tetraethylammonium bromide' (3). Going
        through the index means an alias lands on the busier node whichever
        spelling was written here.
        """
        return index["canonical"].get(_norm_key(value), value)

    alias = index["aliases"].get(raw.lower())
    if alias:
        return canonicalise(alias), "alias"

    key = _norm_key(raw)
    if key in index["canonical"]:                  # also handles a trailing "(ABBR)"
        return index["canonical"][key], "table"

    # "choline acetate (ChAc)" -> try the abbreviation itself against the aliases
    trailing = _TRAILING_ABBR.search(raw)
    if trailing:
        abbr = trailing.group(0).strip().strip("()").lower()
        if abbr in index["aliases"]:
            return canonicalise(index["aliases"][abbr]), "alias"

    alnum = _alnum_key(key)
    if alnum in index["alnum"]:
        return index["alnum"][alnum], "alnum"

    # Prose-only component confirmed by PubChem on an earlier run.
    if key in index["prose_vocabulary"]:
        return index["prose_vocabulary"][key], "pubchem"

    if allow_lookup and looks_like_chemical(raw):
        cache = _load_cache()
        row = cache.get(raw)
        if row is None:
            import requests

            row, _entries = lookup(raw, requests.Session())
            row = row.model_dump()
            cache[raw] = row
            _save_cache(cache)
        if row.get("lookup_status") == "ok":
            index["prose_vocabulary"][key] = raw
            return raw, "pubchem"

    return None, "unresolved"


def resolve_phrase(phrase, index):
    """Resolve a loose phrase by trying progressively shorter word windows.

    Prose tokens arrive with connectives and trailing context attached -- "For ChCl",
    "Benzilic acid (1:1", "amino acids-based DESs". Trying every suffix and prefix,
    longest first, finds the component name inside without guessing at it.
    """
    phrase = str(phrase or "").strip(" ,.;:-")
    if not phrase or not re.search(r"[A-Za-z]{2}", phrase) or phrase.lower() == "nan":
        return None
    words = phrase.split()
    for length in range(len(words), 0, -1):
        for start in (0, len(words) - length):
            found, _how = resolve_component(" ".join(words[start:start + length]), index)
            if found:
                return found
    return None


def resolve_components(names, index, allow_lookup=False):
    """-> (resolved, unresolved, status) where status is resolved|partial|unresolved."""
    resolved, unresolved = [], []
    for name in names or []:
        canonical, _how = resolve_component(name, index, allow_lookup=allow_lookup)
        (resolved if canonical else unresolved).append(canonical or str(name).strip())
    if not resolved:
        status = "unresolved"
    elif unresolved:
        status = "partial"
    else:
        status = "resolved"
    return resolved, unresolved, status


def prose_components(path=None, index=None):
    """Component names used in the prose that Table 2 never mentions.

    These are the genuinely new chemicals the paper discusses in text only --
    '1,5-pentanediol' appears six times in the prose and nowhere in the table -- and
    they are what the PubChem step has to cover beyond the table's own vocabulary.
    """
    from . import store

    rows = store.read_all("sections_llm")
    if not rows:
        return []
    df = pd.DataFrame(rows)
    if "components_resolved" not in df.columns:
        return []
    vocabulary = (index or component_index())["vocabulary"]
    names, seen = [], set()
    for cell in df.components_resolved.dropna():
        for name in str(cell).split(";"):
            name = name.strip()
            if name and name not in vocabulary and name not in seen:
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


def _stale(row):
    """An entry cached before the tabular fields and raw source lines existed.

    Re-fetching is needed for both, so it is one pass rather than two, and it means
    an existing cache heals itself instead of needing to be deleted.
    """
    return row.get("lookup_status") == "ok" and (
        "tpsa" not in row or "property_strings" not in row)


def enrich_all(names, limit=None, network=True, use_nist=True):
    """Look up every name, resuming from the cache. -> list[ComponentRow]."""
    import requests

    cache = _load_cache()
    todo = [n for n in names if n not in cache or _stale(cache[n])]
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
            row, entries = lookup(name, session, use_nist=use_nist)
            # The raw source lines ride alongside the row dump. They are far too long
            # for a CSV column and only the LLM pass reads them; pydantic ignores the
            # extra key on the way back out.
            cache[name] = {**row.model_dump(), "property_strings": entries}
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
