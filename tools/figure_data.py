#!/usr/bin/env python3
"""
Collect everything tools/paper_figures.py draws, from the graph loaded in Neo4j and the
run's own files, into one JSON file.

    DES_RUN_TAG=corpus_full python tools/figure_data.py     # -> runs/corpus_full/figures/figure_data.json

Runs in the GraphDataBase env (neo4j driver + RDKit); paper_figures.py runs it for you
with --refresh, so normally you never call it directly. Re-run after every
`--steps graph`, because everything is read from whatever the database holds.

Component categories and DES types are not stored in the graph. They are assigned here,
by rule, from each component's PubChem SMILES (see `component_category` and `des_type`),
the same way Peng et al. (Green Chem. Eng. 2025) derived theirs -- so they are an
approximation and the figure captions should say so.
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from des_pipeline import config, store     # noqa: E402
from graph_report import PROPS, Graph     # noqa: E402

# DOI prefix -> publisher. Imprints of one group are merged (Hindawi into Wiley, BMC and
# Nature into Springer Nature), since that is how the corpus is usually described.
PUBLISHERS = {
    "10.3390": "MDPI", "10.1016": "Elsevier", "10.1021": "ACS", "10.1039": "RSC",
    "10.1007": "Springer Nature", "10.1038": "Springer Nature", "10.1186": "Springer Nature",
    "10.1002": "Wiley", "10.1111": "Wiley", "10.1155": "Wiley", "10.3389": "Frontiers",
    "10.1080": "Taylor & Francis", "10.1371": "PLOS", "10.1515": "De Gruyter",
    "10.1098": "Royal Society", "10.1063": "AIP", "10.1088": "IOP", "10.1093": "OUP",
    "10.1126": "AAAS", "10.1073": "PNAS", "10.7717": "PeerJ", "10.1149": "ECS",
    "10.3762": "Beilstein", "10.1055": "Thieme", "10.1177": "SAGE", "10.1142": "World Scientific",
}

# Free-text application roles -> a handful of functions. First match wins, so the more
# specific words come first ("extraction solvent" is an extractant, not a solvent).
ROLE_GROUPS = [
    ("extractant", r"extract"), ("catalyst", r"catal"), ("electrolyte", r"electrolyt"),
    ("absorbent / sorbent", r"sorb"), ("reaction medium", r"medium|media"),
    ("pretreatment agent", r"pretreat|fractionat|delignif"),
    ("plasticiser / additive", r"plastici|additive|modifier|template|precursor|agent"),
    ("solvent", r"solvent|dissol|solubil"),
]
SYNONYMS = {"carbon dioxide": "CO2", "co2": "CO2", "so2": "SO2", "h2s": "H2S",
            "5-hydroxymethylfurfural (hmf)": "HMF", "hmf": "HMF"}

METALS = set("Li Be Na Mg Al K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Rb Sr Y Zr Nb Mo Tc Ru "
             "Rh Pd Ag Cd In Sn Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W "
             "Re Os Ir Pt Au Hg Tl Pb Bi U Th".split())
HALOGEN_ACIDS = {"Cl", "Br", "I", "F"}     # the neutral "HCl" fragment of a hydrochloride


# ---------------------------------------------------------------- component categories
def _organic(frag):
    """Carbon-containing, minus the few carbon species chemists call inorganic
    (CO2, carbonate, bicarbonate, cyanide, thiocyanate, cyanate)."""
    carbons = [a for a in frag.GetAtoms() if a.GetSymbol() == "C"]
    if not carbons:
        return False
    if len(carbons) == 1 and carbons[0].GetTotalNumHs() == 0:
        others = {a.GetSymbol() for a in frag.GetAtoms()} - {"C"}
        no_nh = all(a.GetTotalNumHs() == 0 for a in frag.GetAtoms() if a.GetSymbol() == "N")
        if others <= {"O", "N", "S"} and no_nh:
            return False
    return True


def component_category(smiles):
    """-> (category, is_hydrate). Categories follow Peng et al.: water, elemental
    substance, organic molecule, organic salt, inorganic salt (incl. metal halides drawn
    as covalent, e.g. Cl[Zn]Cl), inorganic non-ionic compound, ion."""
    from rdkit import Chem
    if not smiles:
        return "not identified", False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "not identified", False
    if Chem.MolToSmiles(mol) == "O":
        return "water", False
    frags = list(Chem.GetMolFrags(mol, asMols=True))
    water = [f for f in frags if Chem.MolToSmiles(f) == "O"]
    frags = [f for f in frags if Chem.MolToSmiles(f) != "O"] or frags
    hydrate = bool(water)
    elements = {a.GetSymbol() for f in frags for a in f.GetAtoms()}
    has_metal = bool(elements & METALS)
    charged = [f for f in frags if Chem.GetFormalCharge(f) != 0]
    halo_acid = [f for f in frags if f.GetNumAtoms() == 1 and
                 f.GetAtomWithIdx(0).GetSymbol() in HALOGEN_ACIDS and Chem.GetFormalCharge(f) == 0]
    if len(frags) == 1 and len(elements) == 1 and not charged:
        return ("inorganic salt" if has_metal else "elemental substance"), hydrate
    if len(charged) >= 2 or (halo_acid and len(frags) >= 2):
        cations = [f for f in frags if Chem.GetFormalCharge(f) > 0] or \
                  [f for f in frags if f not in halo_acid]
        if any(_organic(f) for f in cations) and not has_metal:
            return "organic salt", hydrate
        return "inorganic salt", hydrate
    if len(frags) == 1 and charged:
        return "ion", hydrate
    if has_metal and not any(_organic(f) for f in frags):
        return "inorganic salt", hydrate
    if has_metal:
        return "inorganic salt", hydrate      # metal carboxylates and the like
    if not any(_organic(f) for f in frags):
        return "inorganic non-ionic compound", hydrate
    return "organic molecule", hydrate


def des_type(cats):
    """DES type from its components' categories (water ignored), after Smith/Abbott and
    Peng et al. -> I/II, III, IV, V, inorganic-salt eutectic, other or unclassified."""
    cats = [c for c, _ in cats if c != "water"]
    if not cats or "not identified" in cats:
        return "unclassified"
    kinds = set(cats)
    if "organic salt" in kinds and "inorganic salt" in kinds:
        return "Type I/II"
    if kinds == {"organic salt", "organic molecule"}:
        return "Type III"
    if kinds == {"inorganic salt", "organic molecule"}:
        return "Type IV"
    if kinds == {"organic molecule"}:
        return "Type V"
    if kinds == {"inorganic salt"}:
        return "inorganic-salt eutectic"
    return "other"


# ------------------------------------------------------------------------- helpers
def publisher(doi):
    prefix = str(doi or "").split("/")[0]
    return PUBLISHERS.get(prefix, "Others")


def role_group(text):
    t = str(text or "").lower()
    for name, pattern in ROLE_GROUPS:
        if re.search(pattern, t):
            return name
    return "other role"


def _fields(source_text):
    out = {}
    for part in str(source_text or "").split(" | "):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _target(text):
    t = re.sub(r"\s+", " ", str(text or "").strip().lower())
    return SYNONYMS.get(t, t) if t and t not in ("none", "nan", "n/a", "-") else None


# ---------------------------------------------------------------------- collection
def collect(g):
    d = {"run_tag": config.RUN_TAG or "(untagged)", "date": str(date.today())}
    mprops = g.q(f"MATCH (n) WHERE labels(n)[0] IN $props AND coalesce(n.origin,'') <> 'component' "
                 "RETURN DISTINCT labels(n)[0] AS p", props=PROPS)
    mprops = [r["p"] for r in mprops]

    # --- papers
    papers = g.q("MATCH (p:Paper) RETURN p.key AS key, p.doi AS doi, p.role AS role, "
                 "p.extracted AS extracted, p.year AS year")
    extracted = [p for p in papers if p["extracted"]]
    cited_only = g.q("MATCH (p:Paper) WHERE NOT coalesce(p.extracted,false) "
                     "AND EXISTS { MATCH ()-[:REPORTED_IN]->(p) } RETURN count(p) AS n")[0]["n"]
    d["papers"] = {
        "extracted": len(extracted),
        "primary": sum(p["role"] == "primary" for p in extracted),
        "review": sum(p["role"] == "review" for p in extracted),
        "cited_only": cited_only,
        "publisher": Counter(publisher(p["doi"]) for p in extracted).most_common(),
        "year": sorted(Counter(str(p["year"])[:4] for p in extracted if p["year"]).items()),
    }

    # --- components
    comps = g.q("MATCH (c:Component) OPTIONAL MATCH (c)-[r:PART_OF]->(m:Mixture) "
                "RETURN c.name AS name, c.smiles AS smiles, c.cid AS cid, "
                "c.h_bond_acceptor_count AS hba, c.h_bond_donor_count AS hbd, c.tpsa AS tpsa, "
                "c.xlogp AS xlogp, c.molecular_weight AS mw, c.qm9_id AS qm9, "
                "collect(DISTINCT r.role) AS roles, count(DISTINCT m) AS n_des")
    with_values = {r["name"] for r in g.q(
        "MATCH (c:Component)-->(x) WHERE x.origin = 'component' RETURN DISTINCT c.name AS name")}
    cat = {}
    for c in comps:
        c["category"], c["hydrate"] = component_category(c["smiles"])
        cat[c["name"]] = (c["category"], c["hydrate"])
    for c in comps:
        c["role"] = ("both" if {"HBA", "HBD"} <= set(c["roles"]) else
                     ("HBA only" if "HBA" in c["roles"] else "HBD only"))
    roles = Counter(c["role"] for c in comps)
    d["components"] = {
        "total": len(comps), "pubchem": sum(bool(c["cid"]) for c in comps),
        "with_db_values": sum(c["name"] in with_values for c in comps),
        "qm9": sum(bool(c["qm9"]) for c in comps),
        "part_of_total": sum(c["n_des"] for c in comps),
        "roles": dict(roles),
        # role -> identified in PubChem -> category -> has database values, for the Sankey
        "flows": Counter((c["role"], bool(c["cid"]), c["category"], c["name"] in with_values)
                         for c in comps).most_common(),
        "category": Counter(c["category"] for c in comps).most_common(),
        "descriptors": [{k: c[k] for k in ("name", "hba", "hbd", "tpsa", "xlogp", "mw", "category")}
                        for c in comps if c["cid"]],
    }
    role_by_comp = g.q("MATCH (c:Component)-[r:PART_OF]->(m:Mixture) "
                       "RETURN c.name AS name, r.role AS role, count(DISTINCT m) AS n")
    by_role = defaultdict(dict)
    for r in role_by_comp:
        by_role[r["name"]][r["role"]] = r["n"]
    d["components"]["top"] = [{"name": c["name"], "n_des": c["n_des"], "category": c["category"],
                               "HBA": by_role[c["name"]].get("HBA", 0),
                               "HBD": by_role[c["name"]].get("HBD", 0)}
                              for c in sorted(comps, key=lambda c: -c["n_des"])[:40]]

    # --- mixtures (DES)
    mixes = g.q("MATCH (m:Mixture) OPTIONAL MATCH (c:Component)-[:PART_OF]->(m) "
                "WITH m, collect(c.name) AS comps "
                "OPTIONAL MATCH (m)-[*1..2]->(p:Paper) WHERE p.extracted "
                "RETURN m.key AS key, comps, collect(DISTINCT p.key) AS papers, "
                "EXISTS { MATCH (m)-[:HAS_APPLICATION]->() } AS has_app, "
                "COUNT { (m)-->(x) WHERE labels(x)[0] IN $props } AS n_values, "
                "COUNT { (m)-[:HAS_APPLICATION]->() } AS n_apps", props=PROPS)
    sets, sets_dry = Counter(), defaultdict(set)
    set_records = Counter()
    types = Counter()
    desc = {c["name"]: c for c in comps}
    avg = []
    for m in mixes:
        names = sorted(set(m["comps"]))
        dry = tuple(n for n in names if cat.get(n, ("",))[0] != "water") or tuple(names)
        sets[tuple(names)] += 1
        sets_dry[dry].update(m["papers"])
        set_records[dry] += m["n_values"] + m["n_apps"]
        types[des_type([cat.get(n, ("not identified", False)) for n in names])] += 1
        xs = [desc[n]["xlogp"] for n in dry if desc.get(n) and desc[n]["xlogp"] is not None]
        ts = [desc[n]["tpsa"] for n in dry if desc.get(n) and desc[n]["tpsa"] is not None]
        if len(xs) == len(dry) and len(ts) == len(dry):
            avg.append([sum(xs) / len(xs), sum(ts) / len(ts)])
    d["mixtures"] = {
        "total": len(mixes), "distinct_sets": len(sets), "distinct_sets_dry": len(sets_dry),
        "with_values": sum(m["n_values"] > 0 for m in mixes),
        "with_apps": sum(m["has_app"] for m in mixes),
        "with_both": sum(m["n_values"] > 0 and m["has_app"] for m in mixes),
        "type": types.most_common(),
        "top_sets": [{"components": list(k), "papers": len(v), "records": set_records[k]}
                     for k, v in sorted(sets_dry.items(), key=lambda kv: -len(kv[1]))[:30]],
        "avg_xlogp_tpsa": avg,
        "n_components": Counter(len(set(m["comps"])) for m in mixes).most_common(),
    }

    # --- DES property values
    vals = g.q("MATCH (m:Mixture)-->(x) WHERE labels(x)[0] IN $props "
               "OPTIONAL MATCH (x)-[r:REPORTED_IN]->(p:Paper) WHERE p.extracted "
               "RETURN elementId(x) AS id, labels(x)[0] AS prop, x.value AS v, x.unit AS unit, x.temperature_C AS T, "
               "x.uncertainty IS NOT NULL AS unc, x.plausible AS plausible, x.origin AS origin, "
               "size(x.member_keys) AS rows, collect(DISTINCT p.doi)[0] AS doi", props=PROPS)
    rows_files = store.read_all("measurements")
    prose_all = store.read_all("sections_llm")
    prose_ok = [r for r in prose_all if r.get("status") == "ok"]
    d["values"] = {
        "nodes": len(vals), "rows_in_graph": sum(v["rows"] or 1 for v in vals),
        "rows_extracted": len(rows_files) + len(prose_all),
        "rows_by_publisher": Counter(publisher(r.get("Paper_DOI"))
                                     for r in rows_files + prose_all).most_common(),
        # what kept a row out of the graph (it stays in the CSVs for review)
        "status_table": Counter(r.get("status") or "ok" for r in rows_files).most_common(),
        "status_prose": Counter(r.get("status") or "ok" for r in prose_all).most_common(),
        "with_uncertainty": sum(v["unc"] for v in vals),
        "with_temperature": sum(v["T"] is not None for v in vals),
        "by_property": Counter(v["prop"] for v in vals).most_common(),
        "flows": Counter((v["prop"], v["T"] is not None, bool(v["unc"])) for v in vals).most_common(),
        "list": [[v["prop"], v["v"], v["T"], v["plausible"] is not False] for v in vals],
        "units": {v["prop"]: v["unit"] for v in vals},
        "plausible_range": {k: list(r) for k, r in config.PLAUSIBLE_RANGE.items() if r},
    }

    # --- component property values
    cvals = g.q("MATCH (c:Component)-->(x) WHERE x.origin = 'component' "
                "OPTIONAL MATCH (x)-[:REPORTED_IN]->(s:Source) "
                "RETURN elementId(x) AS id, c.name AS comp, labels(x)[0] AS prop, x.value AS v, x.unit AS unit, "
                "[s IN collect(s) WHERE s.kind = 'database' | s.name][0] AS db, "
                "[s IN collect(s) WHERE s.kind = 'attribution' | s.name] AS credited")
    d["component_values"] = {
        "total": len(cvals),
        "by_property": Counter(v["prop"] for v in cvals).most_common(),
        "by_property_db": Counter(f"{v['prop']}|{v['db']}" for v in cvals).most_common(),
        "credited": Counter(s for v in cvals for s in v["credited"]).most_common(),
        "list": [[v["comp"], v["prop"], v["v"]] for v in cvals],
        "units": {v["prop"]: v["unit"] for v in cvals},
        "sources": g.q("MATCH (s:Source) RETURN count(s) AS n")[0]["n"],
    }

    # --- applications
    apps = g.q("MATCH (m:Mixture)-[:HAS_APPLICATION]->(a:ApplicationRecord) "
               "OPTIONAL MATCH (a)-[:REPORTED_IN]->(p:Paper) WHERE p.extracted "
               "RETURN elementId(a) AS id, a.domain AS domain, a.source_text AS t, m.key AS mix, "
               "size(a.member_keys) AS rows, collect(DISTINCT p.key) AS papers")
    app_rows = store.read_all("applications")
    targets, papers_dom, mix_dom = defaultdict(Counter), defaultdict(set), defaultdict(set)
    spelling = defaultdict(Counter)    # lower-cased key -> the spellings it came in
    role_counts = Counter()
    for a in apps:
        f = _fields(a["t"])
        if _target(f.get("target")):
            key = _target(f.get("target"))
            targets[a["domain"]][key] += 1
            spelling[key][f["target"].strip()] += 1
        if f.get("role"):
            role_counts[role_group(f["role"])] += 1
        papers_dom[a["domain"]].update(a["papers"])
        mix_dom[a["domain"]].add(a["mix"])
    d["applications"] = {
        "records": len(apps), "rows_in_graph": sum(a["rows"] or 1 for a in apps),
        "rows_extracted": len(app_rows),
        "status": Counter(r.get("status") or "ok" for r in app_rows).most_common(),
        "rows_by_publisher": Counter(publisher(r.get("Paper_DOI")) for r in app_rows).most_common(),
        "domains": len({a["domain"] for a in apps}),
        "by_domain": Counter(a["domain"] for a in apps).most_common(),
        "papers_by_domain": {k: len(v) for k, v in papers_dom.items()},
        "des_by_domain": {k: len(v) for k, v in mix_dom.items()},
        "roles": role_counts.most_common(),
        # counted case-insensitively, shown in the most common original spelling (ZnO, not zno)
        "targets": {k: [(key if key in SYNONYMS.values() else spelling[key].most_common(1)[0][0], n)
                        for key, n in v.most_common(12)] for k, v in targets.items()},
    }

    # --- network: every DES and the hubs it touches
    links = g.q("MATCH (m:Mixture)-[r]->(x) WHERE type(r) <> 'REPORTED_IN' "
                "RETURN m.key AS m, CASE WHEN labels(x)[0] = 'ApplicationRecord' THEN 'D:' + x.domain "
                f"ELSE 'P:' + labels(x)[0] END AS hub")
    comp_links = g.q("MATCH (c:Component)-[:PART_OF]->(m:Mixture) RETURN m.key AS m, 'C:' + c.name AS hub")
    d["network"] = sorted({(r["m"], r["hub"]) for r in links + comp_links})

    # --- pipeline counts for the workflow figure
    tables = Counter(r.get("status") for r in store.read_all("tables"))
    prose = Counter(r.get("status") for r in store.read_all("sections_llm"))
    corpus = {}
    census = config.ROOT / "eval" / "top_cited" / "corpus_citations.csv"
    if census.exists():
        c = pd.read_csv(census)
        corpus = {"downloaded": len(c), "title_abstract": int((c.phrase_in == "title_abstract").sum())}
    comp_csv = config.DATA / "components.csv"
    names = pd.read_csv(comp_csv) if comp_csv.exists() else pd.DataFrame()
    queue = config.REVIEW_QUEUE_CSV
    dups = config.DATA / "duplicate_measurements.csv"
    d["pipeline"] = {
        **corpus,
        "tables": sum(tables.values()), "table_status": dict(tables),
        "prose_rows": sum(prose.values()), "prose_ok": prose.get("ok", 0),
        "references": len(store.read_all("references")),
        "figures": len(store.read_all("figures")),
        "dictionary": len(store.read_all("des_dictionary")),
        "application_rows": d["applications"]["rows_extracted"],
        "table_values": len(rows_files),
        "component_names": len(names),
        "component_names_resolved": int(names["smiles"].notna().sum()) if "smiles" in names else None,
        "review_queue": len(pd.read_csv(queue)) if queue.exists() else None,
        "duplicate_groups": len(pd.read_csv(dups)) if dups.exists() else None,
        "graph_nodes": g.q("MATCH (n) RETURN count(n) AS n")[0]["n"],
        "graph_rels": g.q("MATCH ()-[r]->() RETURN count(r) AS n")[0]["n"],
        "papers_cited_nodes": cited_only,
    }
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=(config.RUN_DIR or config.ROOT) / "figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    g = Graph()
    try:
        data = collect(g)
    finally:
        g.close()
    path = args.out / "figure_data.json"
    path.write_text(json.dumps(data, default=str))
    print(f"wrote {path}  ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
