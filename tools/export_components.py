"""
Export every Component node in the graph to CSV.

    DES_RUN_TAG=corpus_full python tools/export_components.py [--out dir]

Writes two files to runs/<tag>/exports/ (or --out):

    components_all.csv       every Component node; SMILES etc. blank when unidentified
    components_pubchem.csv   only the components identified in PubChem (have a CID)
    components_pubchem_unique.csv   the same, one row per InChIKey (unique structure)

Read from Neo4j, so it reflects the graph as currently loaded -- spelling variants
merged per data/review/component_duplicates.csv are one row, with the merged
spellings in `merged_from`.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parent.parent),
                str(Path(__file__).resolve().parent)]
from des_pipeline import config               # noqa: E402
from graph_report import Graph                # noqa: E402

QUERY = """
MATCH (c:Component)
OPTIONAL MATCH (c)-[p:PART_OF]->(m:Mixture)
WITH c, count(DISTINCT m) AS n_des,
     count(DISTINCT CASE WHEN p.role = 'HBA' THEN m END) AS n_as_hba,
     count(DISTINCT CASE WHEN p.role = 'HBD' THEN m END) AS n_as_hbd
RETURN c.name AS name, c.smiles AS smiles, c.cid AS pubchem_cid,
       c.inchikey AS inchikey, c.formula AS formula, c.cas AS cas,
       c.molecular_weight AS molecular_weight, c.matched_name AS pubchem_matched_name,
       c.merged_from AS merged_from, n_des, n_as_hba, n_as_hbd
ORDER BY n_des DESC, name
"""

# DES counts per STRUCTURE, counted over distinct mixtures so a DES that happens to
# contain two nodes of one structure is not counted twice.
BY_INCHIKEY = """
MATCH (c:Component) WHERE c.inchikey IS NOT NULL
OPTIONAL MATCH (c)-[p:PART_OF]->(m:Mixture)
RETURN c.inchikey AS inchikey, count(DISTINCT m) AS n_des,
       count(DISTINCT CASE WHEN p.role = 'HBA' THEN m END) AS n_as_hba,
       count(DISTINCT CASE WHEN p.role = 'HBD' THEN m END) AS n_as_hbd
"""


def unique_structures(df, counts):
    """One row per InChIKey. Name and identifiers from the node used in the most DES;
    every other node's name and merged spellings go into `other_names`."""
    out = []
    for key, group in df[df["inchikey"].notna()].groupby("inchikey", sort=False):
        group = group.sort_values(["n_des", "name"], ascending=[False, True])
        first = group.iloc[0].to_dict()
        names = list(group["name"])
        for merged in group["merged_from"]:
            names += [n for n in str(merged or "").split("; ") if n]
        first["other_names"] = "; ".join(dict.fromkeys(
            n for n in names if n != first["name"]))
        first["n_nodes"] = len(group)
        first.update(counts[key])
        out.append(first)
    cols = ["name", "smiles", "pubchem_cid", "inchikey", "formula", "cas",
            "molecular_weight", "other_names", "n_nodes", "n_des", "n_as_hba",
            "n_as_hbd"]
    return (pd.DataFrame(out)[cols]
            .sort_values(["n_des", "name"], ascending=[False, True]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=config.RUN_DIR / "exports")
    args = parser.parse_args()

    g = Graph()
    try:
        rows = g.q(QUERY)
        counts = {r.pop("inchikey"): r for r in g.q(BY_INCHIKEY)}
    finally:
        g.close()
    if not rows:
        sys.exit("no Component nodes -- is the graph loaded?")

    df = pd.DataFrame(rows)
    df["merged_from"] = df["merged_from"].apply(
        lambda v: "; ".join(v) if isinstance(v, list) else (v or ""))
    df["pubchem_cid"] = df["pubchem_cid"].astype("Int64")

    args.out.mkdir(parents=True, exist_ok=True)
    everything = args.out / "components_all.csv"
    pubchem = args.out / "components_pubchem.csv"
    df.to_csv(everything, index=False)
    df[df["pubchem_cid"].notna()].to_csv(pubchem, index=False)
    print(f"{len(df)} components -> {everything}")
    print(f"{int(df['pubchem_cid'].notna().sum())} in PubChem -> {pubchem}")
    unique = args.out / "components_pubchem_unique.csv"
    structures = unique_structures(df, counts)
    structures.to_csv(unique, index=False)
    print(f"{len(structures)} unique structures (by InChIKey) -> {unique}")


if __name__ == "__main__":
    main()
