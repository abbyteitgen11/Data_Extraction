"""
One chemical, several spellings: find the duplicate Component nodes and propose merges.

`Component` merges on `name`, which is whatever spelling a paper used, so the same
substance becomes several nodes and its memberships split between them:

    Water  47 mixtures   |   Ethylene glycol 119   1,2-Ethanediol 2   Glycol 2
    water   1            |   1,2-ethanediol    2   Monoethylene glycol 1
    H2O     1

403 resolved components are 312 distinct substances. Anything that counts by component
undercounts, and an ML feature table built from these nodes treats one component as five.

Nothing here merges anything. It writes `data/review/component_duplicates.csv` with a
proposal per group, and `build_graph` honours whatever that file says -- the same
hand-edited-file-is-authoritative pattern as component_aliases.json, for the same reason:
a wrong merge is invisible afterwards. Once Xylitol and Ribitol are one node, nothing
downstream remembers they were ever different.

Three kinds of group turn up and they need different answers:

    naming variants     Water / water / H2O          -> merge
    mis-resolutions     PEG600 ... -> diethylene glycol, which PubChem got wrong
                                                      -> never merge
    stereoisomers       Glucose / Galactose / Mannose -> never merge

Grouping is on the FULL InChIKey for exactly that third case. The connectivity block
ignores stereochemistry, so it puts three different sugars in one group.

    python run_pipeline.py --steps duplicates
"""
import re

import pandas as pd

from . import config
from .qm9 import contested_components

REVIEW_COLUMNS = ["inchikey", "cid", "n_names", "names", "mixtures", "canonical",
                  "merge", "note"]

# PubChem's "preferred name" is its first synonym, and that is whatever its sources
# happened to rank first. It is often the right answer and sometimes not a name at all:
# a CAS number for methyltrioctylammonium chloride, "Dichlorozinc" for zinc chloride,
# "(2S,4R)-pentane-1,2,3,4,5-pentol" for xylitol. Where it is unusable the corpus's own
# spelling wins instead.
_CAS_NUMBER = re.compile(r"^\d{2,7}-\d{2}-\d$")
_SYSTEMATIC = re.compile(
    r"^\(|\d[RSEZ][,)]"                      # (2S,4R)-... stereo-prefixed IUPAC
    r"|^[a-z]+o(zinc|iron|copper|lead|tin)"  # Dichlorozinc
    r"|pentol$|tetrol$|triol$"               # ...pentane-1,2,3,4,5-pentol
    r"|pyranose$|furanose$",                 # D-Glucopyranose, a ring form no paper writes
    re.I)


def _unusable(name):
    """Is this PubChem name something no paper would ever write? -> reason | ''."""
    if _CAS_NUMBER.match(name):
        return "a CAS number, not a name"
    if _SYSTEMATIC.search(name):
        return "a systematic form no paper writes"
    return ""

# Groups where the automatic check says "same substance" but chemistry says look again.
# Each is a real hazard found by reading the generated list, not a hypothetical.
SUSPECT = {
    "HEBKCHPVOIAQTA-NGQZWQHPSA-N":
        "xylitol and ribitol are DIFFERENT pentitols -- one of these is a "
        "mis-resolution the synonym check did not catch",
    "QWBBPBRQALCEIZ-UHFFFAOYSA-N":
        "'Xylenol' is generic -- there are six isomers and PubChem picked one",
    "NOOLISFMXDJSKH-UHFFFAOYSA-N":
        "the CID is stereo-unspecified, so merging drops the racemic vs "
        "single-enantiomer distinction",
}


def _title(name):
    """'CHOLINE CHLORIDE' -> 'Choline chloride'. Leaves mixed case alone.

    PubChem's preferred names are cased inconsistently -- CHOLINE CHLORIDE, water,
    LEVULINIC ACID -- and that casing is an artefact of their sources, not a naming
    decision worth propagating into every mixture name.
    """
    text = str(name or "").strip()
    if not text:
        return ""
    if text.isupper() or text.islower():
        return text[0].upper() + text[1:].lower() if text.isupper() else \
            text[0].upper() + text[1:]
    return text


def preferred_name(group, usage):
    """The name the merged node should carry. -> (name, source, note).

    PubChem's own preferred name where it is usable, because it is externally defined
    and does not drift as papers are added; the corpus's most-used spelling where it is
    not.
    """
    most_used = max(group, key=lambda r: usage.get(r["name"], 0))["name"]
    synonyms = [s for s in str(group[0].get("synonyms") or "").split(";") if s.strip()]
    if not synonyms:
        return most_used, "corpus", "no PubChem synonyms; used the most-common spelling"

    proposed = _title(synonyms[0])
    why_not = _unusable(proposed)
    if why_not:
        return (most_used, "corpus",
                f"PubChem prefers {proposed!r} -- {why_not}; used the most-common spelling")
    # If one of our own spellings IS the preferred name, keep our casing of it.
    for r in sorted(group, key=lambda r: -usage.get(r["name"], 0)):
        if r["name"].lower() == proposed.lower():
            return r["name"], "pubchem", ""
    # A usable name, but still one none of our papers uses -- e.g. "Guaiacol" where the
    # papers write "2-methoxyphenol". Legitimate, and worth flagging so a reviewer can
    # scan for the ones they would rather spell their own way.
    return proposed, "pubchem", f"PubChem's name; no paper in the corpus writes it"


def find_groups(rows, usage=None):
    """Components sharing a full InChIKey. -> list of proposal dicts."""
    usage = usage or {}
    contested = contested_components(rows)

    by_key = {}
    for r in rows:
        if r.get("lookup_status") == "ok" and r.get("inchikey"):
            by_key.setdefault(r["inchikey"], []).append(r)

    out = []
    for key, group in by_key.items():
        if len(group) < 2:
            continue
        group = sorted(group, key=lambda r: -usage.get(r["name"], 0))
        canonical, source, why = preferred_name(group, usage)

        if key in SUSPECT:
            decision, note = "check", SUSPECT[key]
        elif any(r["name"] in contested for r in group):
            decision = "no"
            note = ("a name here is not one PubChem acknowledges for this compound, so "
                    "the identification is in doubt")
        else:
            decision, note = "yes", why

        out.append({
            "inchikey": key,
            "cid": group[0].get("cid"),
            "n_names": len(group),
            "names": ";".join(r["name"] for r in group),
            "mixtures": ";".join(str(usage.get(r["name"], 0)) for r in group),
            "canonical": canonical,
            "merge": decision,
            "note": note if decision != "yes" or why else
                    ("PubChem's preferred name" if source == "pubchem" else ""),
        })
    return sorted(out, key=lambda g: -sum(int(x) for x in g["mixtures"].split(";")))


def usage_counts():
    """How many mixtures each component name appears in. -> {name: count}.

    Read from the graph when it is loaded, else from the CSVs, so the review file shows
    the real weight of a decision either way.
    """
    counts = {}
    try:
        from .build_graph import _driver

        driver = _driver()
        try:
            for r in driver.execute_query(
                    "MATCH (c:Component) OPTIONAL MATCH (c)-[:PART_OF]->(m:Mixture) "
                    "RETURN c.name AS n, count(DISTINCT m) AS m",
                    database_=config.NEO4J_DATABASE).records:
                counts[r["n"]] = r["m"]
            return counts
        finally:
            driver.close()
    except Exception:
        pass

    from . import store

    for source, fields in ((store.read_all("mixtures"), ("Component_1", "Component_2",
                                                         "Component_3")),
                           (store.read_all("applications"), ("Component_1", "Component_2",
                                                             "Component_3"))):
        for row in source:
            for f in fields:
                name = row.get(f)
                if isinstance(name, str) and name.strip():
                    counts[name] = counts.get(name, 0) + 1
    return counts


def load_decisions():
    """The review file as {inchikey: row}. Empty when it does not exist yet."""
    if not config.COMPONENT_DUPLICATES_CSV.exists():
        return {}
    df = pd.read_csv(config.COMPONENT_DUPLICATES_CSV)
    df = df.astype(object).where(pd.notna(df), None)
    return {r["inchikey"]: r for r in df.to_dict("records") if r.get("inchikey")}


def canonical_map():
    """{written name: canonical name} for every group marked `yes`. -> dict.

    This is what build_graph applies. A group left at `no` or `check` contributes
    nothing, so an unreviewed or doubted merge simply does not happen.
    """
    mapping = {}
    for row in load_decisions().values():
        if str(row.get("merge") or "").strip().lower() != "yes":
            continue
        canonical = str(row.get("canonical") or "").strip()
        if not canonical:
            continue
        for name in str(row.get("names") or "").split(";"):
            name = name.strip()
            if name and name != canonical:
                mapping[name] = canonical
    return mapping


def run():
    """Write the review file, keeping any decision already recorded in it."""
    if not config.COMPONENTS_CSV.exists():
        print("  no components.csv yet -- run --steps components first")
        return []

    df = pd.read_csv(config.COMPONENTS_CSV)
    rows = df.astype(object).where(pd.notna(df), None).to_dict("records")
    usage = usage_counts()
    proposals = find_groups(rows, usage)

    # Never overwrite a human decision. Only the evidence columns are refreshed.
    existing = load_decisions()
    kept = 0
    for p in proposals:
        prior = existing.get(p["inchikey"])
        if prior and str(prior.get("merge") or "").strip():
            p["merge"] = prior["merge"]
            p["canonical"] = prior.get("canonical") or p["canonical"]
            p["note"] = prior.get("note") or p["note"]
            kept += 1

    config.COMPONENT_DUPLICATES_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(proposals, columns=REVIEW_COLUMNS).to_csv(
        config.COMPONENT_DUPLICATES_CSV, index=False)

    tally = {}
    for p in proposals:
        tally[p["merge"]] = tally.get(p["merge"], 0) + 1
    surplus = sum(p["n_names"] - 1 for p in proposals if p["merge"] == "yes")
    print(f"  {len(proposals)} duplicate group(s) covering "
          f"{sum(p['n_names'] for p in proposals)} component nodes")
    print(f"  proposed: " + ", ".join(f"{v} {k}" for k, v in sorted(tally.items())))
    if kept:
        print(f"  {kept} decision(s) preserved from the existing file")
    print(f"  merging the 'yes' rows would remove {surplus} node(s)")
    print(f"  -> {config.COMPONENT_DUPLICATES_CSV}   (edit `merge` and `canonical`)")
    return proposals
