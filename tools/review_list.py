#!/usr/bin/env python3
"""
Collect everything in a loaded graph (and the files it was built from) that needs a
human look, ranked by how much it matters, into one folder:

    DES_RUN_TAG=corpus_full python tools/review_list.py

Writes runs/<tag>/review_list/REVIEW.md (the overview: what each issue is, how many,
what to do) and one CSV per issue, each sorted so the rows worth checking first come
first. Reads only; changes nothing in the graph or the data.
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from des_pipeline import config               # noqa: E402
from graph_report import PROPS, Graph          # noqa: E402

# A component name that is a fragment of a sentence or a number, not a compound.
JUNK = re.compile(r"""^(the|a|an|with|for|and|of|in|when|from|molar|ratio|sample)\b   # starts like prose
                  | \d\s*(g|mg|kg)\s*[·/]?\s*(cm|ml|mL|l|L)                         # a density or concentration
                  | \b(ranged|showed|shown|started|revealed|achieved|had|was|were|order)\b
                  | ^[\d\s.,:;×–\-−]+$                                            # only numbers
                  | ^[^\w\[(]                                                     # starts with stray punctuation
                  """, re.X | re.I)
GENERIC = {"des", "nades", "hbd", "hba", "salt", "acid", "solid", "formulations", "chloride",
           "cl", "il", "ils", "solvent", "mof", "amino acids"}


def frame(rows, cols=None):
    return pd.DataFrame(rows, columns=cols) if rows or cols else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=(config.RUN_DIR or config.ROOT) / "review_list")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = config.DATA
    sections = []      # (title, n, csv name or None, what it is, what to do)

    def save(df, name):
        df.to_csv(args.out / name, index=False)
        return name

    g = Graph()
    try:
        # ---------------------------------------------------------------- values
        rng = config.PLAUSIBLE_RANGE
        rows = g.q("""
            MATCH (mx:Mixture)-[]->(m)-[:REPORTED_IN]->(p:Paper)
            WHERE labels(m)[0] IN $props AND coalesce(m.origin,'') <> 'component'
            RETURN labels(m)[0] AS property, m.value AS value, m.unit AS unit,
                   m.temperature_C AS temperature_C, m.uncertainty AS uncertainty,
                   m.plausible AS plausible, m.plausibility_note AS note, m.origin AS origin,
                   mx.name AS mixture, collect(DISTINCT p.doi) AS paper_dois,
                   m.source_text AS source_text, m.key AS key""",
                    props=PROPS)
        m = frame(rows)
        lo = m.property.map(lambda p: rng.get(p, (None, None))[0])
        hi = m.property.map(lambda p: rng.get(p, (None, None))[1])
        out = (m.value < lo) | (m.value > hi)
        below0 = m.property.isin(["Melting_point", "Boiling_point", "Glass_transition",
                                  "Decomposition_temperature"]) & (m.value < -273.15)
        bad = m[(m.plausible == False) | out | below0].copy()          # noqa: E712
        bad["why"] = [("below absolute zero; " if b else "") + ("outside " + f"{l}…{h}; " if o else "")
                      + (n or "") for b, o, l, h, n in zip(below0[bad.index], out[bad.index],
                                                          lo[bad.index], hi[bad.index], bad.note)]
        bad = bad.sort_values(["property", "value"])
        sections.append(("Implausible DES property values", len(bad), save(bad, "values_implausible.csv"),
                         "Values outside the plausible range for their property (or below absolute "
                         "zero), mostly a unit or a header factor misread, a K/°C mix-up, or a "
                         "different quantity in the column.",
                         "Check each against its `source_text` / paper; fix the paper's table profile "
                         "or drop the row. They are in the graph, flagged `plausible=false`."))

        hot = m[m.temperature_C.notna() & ((m.temperature_C < -100) | (m.temperature_C > 300))]
        sections.append(("Implausible measuring temperatures", len(hot), save(hot, "temperatures_implausible.csv"),
                         "A measuring temperature outside −100…300 °C, usually a value in K read as °C.",
                         "Check the table's temperature header."))

        # disagreeing duplicates
        dup_path = data / "duplicate_measurements.csv"
        if dup_path.exists():
            d = pd.read_csv(dup_path)
            d = d[d["agree"] == False].sort_values("n_rows", ascending=False)   # noqa: E712
            sections.append(("Same measurement, different values", len(d), save(d, "duplicates_disagreeing.csv"),
                             "The same DES, property, temperature and originating study reported with "
                             "different values (often a paper and a review quoting it).",
                             "Decide which value is right; usually a transcription or unit slip in one."))

        # ------------------------------------------------------------ components
        comps = frame(g.q("""
            MATCH (c:Component) OPTIONAL MATCH (c)-[:PART_OF]->(mx:Mixture)
            RETURN c.name AS name, c.cid AS cid, c.inchikey AS inchikey, c.origin AS origin,
                   c.qm9_contested AS qm9_contested, count(DISTINCT mx) AS n_des"""))
        junk = comps[comps.name.map(lambda s: bool(JUNK.search(s)) or s.strip().lower() in GENERIC)]
        sections.append(("Component names that are not compounds", len(junk),
                         save(junk.sort_values("n_des", ascending=False), "components_junk_names.csv"),
                         "Component nodes whose name is a fragment of text, a number or a generic word "
                         "(\"DES\", \"salt\", \"chloride\").",
                         "Find the paper rows that produced them (search the name in the paper's "
                         "mixtures.csv); usually a prose or application row read badly. Drop or correct."))

        unres = comps[comps.cid.isna() & ~comps.index.isin(junk.index)].sort_values("n_des", ascending=False)
        sections.append(("Components not identified in PubChem", len(unres),
                         save(unres, "components_unresolved.csv"),
                         "No PubChem identity, so no SMILES, properties or merging. Often an "
                         "abbreviation (TBAB, ChCl), an ionic liquid, a polymer or a typo.",
                         "Work down the list by `n_des`; for real compounds add the reviewed alias to "
                         "`component_aliases.json` and re-run the components step."))

        shared = comps[comps.inchikey.notna()].groupby("inchikey").filter(lambda x: len(x) > 1)
        sections.append(("Different component nodes with the same structure", len(shared),
                         save(shared.sort_values(["inchikey", "n_des"], ascending=[True, False]),
                              "components_same_inchikey.csv"),
                         "Names that PubChem resolves to one structure but that are still separate nodes "
                         "(the merge for them is 'no' or 'check').",
                         "Decide in `data/review/component_duplicates.csv`."))

        qc = comps[comps.qm9_contested.notna()]
        sections.append(("Contested QM9 matches", len(qc), save(qc, "components_qm9_contested.csv"),
                         "The QM9 entry matched by structure is also claimed by other names.",
                         "Check that the QM9 molecule is this component."))

        cd_path = config.SHARED_DATA / "review" / "component_duplicates.csv"
        if cd_path.exists():
            cd = pd.read_csv(cd_path)
            todo = cd[cd["merge"].astype(str).str.lower().isin(["check", "", "nan"])]
            sections.append(("Duplicate-component groups awaiting a decision", len(todo),
                             save(todo, "component_duplicates_to_decide.csv"),
                             f"Of {len(cd)} groups of names sharing a structure, these are marked "
                             f"'check' ({(cd['merge'] == 'yes').sum()} are proposed 'yes', "
                             f"{(cd['merge'] == 'no').sum()} 'no').",
                             "Set `merge` to yes/no in `data/review/component_duplicates.csv` (it is "
                             "shared by all runs); skim the proposed 'yes' rows too."))

        # -------------------------------------------------------------- mixtures
        flags = frame(g.q("""
            MATCH (mx:Mixture) WHERE coalesce(mx.ratio_flag,'') <> '' OR coalesce(mx.component_flag,'') <> ''
            OPTIONAL MATCH (mx)-[:REPORTED_IN]->(p:Paper)
            RETURN mx.name AS mixture, mx.ratio_raw AS ratio_raw, mx.ratio_flag AS ratio_flag,
                   mx.component_flag AS component_flag, collect(DISTINCT p.doi)[..3] AS papers"""))
        sections.append(("DES with a doubtful ratio or component set", len(flags),
                         save(flags, "mixtures_flagged.csv"),
                         "Ratio given only by weight, unknown, or a mixture of four or more components.",
                         "Weight ratios are converted with PubChem molecular weights; check the "
                         "unknown ones and the 4+-component sets."))

        empty = frame(g.q("""
            MATCH (mx:Mixture) WHERE NOT (mx)-->(:ApplicationRecord)
              AND NOT EXISTS { MATCH (mx)-->(m) WHERE labels(m)[0] IN $props }
            OPTIONAL MATCH (mx)-[:REPORTED_IN]->(p:Paper)
            RETURN mx.name AS mixture, mx.origin AS origin, collect(DISTINCT p.doi)[..3] AS papers""",
                          props=PROPS))
        sections.append(("DES with no data", len(empty), save(empty, "mixtures_without_data.csv"),
                         "Defined in a paper (composition table, dictionary) but no property or "
                         "application was attached.",
                         "Mostly fine (preparation tables); a cluster from one paper suggests its data "
                         "table was missed."))

        # ---------------------------------------------------------------- papers
        nodata = frame(g.q("""
            MATCH (p:Paper) WHERE p.extracted AND NOT ()-[:REPORTED_IN]->(p)
            RETURN p.doi AS doi, p.title AS title, p.role AS role, p.journal AS journal, p.year AS year"""))
        meta = pd.read_csv(data / "papers.csv") if (data / "papers.csv").exists() else pd.DataFrame()
        if len(meta) and len(nodata):
            nodata = nodata.merge(meta[["Paper_DOI", "slug", "article_type"]], how="left",
                                  left_on="doi", right_on="Paper_DOI").drop(columns="Paper_DOI")
            nodata = nodata.sort_values(["article_type", "role"], key=lambda s: s != "research-article")
        sections.append(("Extracted papers that gave no data", len(nodata),
                         save(nodata, "papers_without_data.csv"),
                         "Processed but nothing was linked to them. Expected for editorials, "
                         "corrections and some reviews; for research articles, a table or prose read "
                         "may have failed.",
                         "Open a few research articles from the top of the list and check their tables."))

        doubtful = frame(g.q("""
            MATCH (p:Paper) WHERE p.title_agreement IS NOT NULL AND p.title_agreement < 0.6
            OPTIONAL MATCH (x)-[:REPORTED_IN]->(p)
            RETURN p.doi AS doi, p.title AS matched_title, p.raw AS reference_as_printed,
                   p.title_agreement AS title_agreement, count(x) AS n_data
            ORDER BY n_data DESC"""))
        sections.append(("Cited papers with a doubtful DOI match", len(doubtful),
                         save(doubtful, "references_doubtful.csv"),
                         "A cited reference whose Crossref match has a title that disagrees with the "
                         "reference as printed, so data may be credited to the wrong paper.",
                         "Compare `reference_as_printed` with `matched_title`."))

        # ---------------------------------------------------------- applications
        other = frame(g.q("""
            MATCH (r:ApplicationRecord)-[:IN_DOMAIN]->(:Application {domain:'other'})
            RETURN r.subject AS subject, r.source_text AS source_text"""))
        n_app = g.q("MATCH (r:ApplicationRecord) RETURN count(r) AS n")[0]["n"]
        sections.append(("Applications filed as 'other'", len(other), save(other, "applications_other.csv"),
                         f"{len(other) / max(n_app, 1):.0%} of application records fit none of the "
                         "16 named domains.",
                         "Skim for recurring uses that deserve their own domain (e.g. food, "
                         "cosmetics, CO2 conversion); the domain list is in config."))
    finally:
        g.close()

    # --------------------------------------------------- from the validation step
    qpath = data / "review" / "queue.csv"
    if qpath.exists():
        q = pd.read_csv(qpath).sort_values(["priority", "rows_affected"], ascending=[True, False])
        save(q, "extraction_queue.csv")
        for kind, part in q.groupby("kind", sort=False):
            name = "queue_" + re.sub(r"\W+", "_", kind).strip("_") + ".csv"
            sections.append((f"Extraction: {kind}", len(part), save(part, name),
                             f"From the validation step's review queue (priority {part.priority.min()}"
                             f"–{part.priority.max()}, {int(part.rows_affected.sum())} rows affected).",
                             QUEUE_ADVICE.get(kind, "See `detail`; fix the paper's table profile if "
                                              "the rows matter.")))

    skipped = [pd.read_csv(f).assign(file=f.name) for f in sorted((data / "review").glob("*_skipped_cells.csv"))
               if f.stat().st_size > 0]
    skipped = [s for s in skipped if len(s)]
    if skipped:
        s = pd.concat(skipped)
        per = (s.groupby(["Paper_key", "Property"]).size().rename("cells").reset_index()
               .sort_values("cells", ascending=False))
        sections.append(("Table cells that could not be read", len(s), save(per, "skipped_cells_by_paper.csv"),
                         "Cells in property columns that did not parse as a value (ranges, footnotes, "
                         "'–', text). Per paper and property.",
                         "Large counts in one paper usually mean one column format the parser misses."))

    write_md(args.out, sections)
    print((args.out / "REVIEW.md").read_text())


QUEUE_ADVICE = {
    "table no_rows": "A table judged relevant gave no row; check whether it really holds DES data.",
    "table failed": "The model's labels for the table failed validation; the table was skipped.",
    "column relabelled by code": "A deterministic rule overrode the model's column label; spot-check.",
    "DES definition not verified": "A dictionary entry whose components could not be found in the paper.",
    "unresolved abbreviation": "An abbreviation used in data rows with no definition found.",
    "implausible value": "Flagged at extraction; see the values list as well.",
    "prose: quote_not_found": "A prose value whose supporting quote is not in the paper text (possible invention); dropped.",
    "prose: value_not_in_quote": "A prose value missing from its own quote; dropped.",
    "doubtful Crossref match": "See the doubtful references list.",
}


def write_md(out, sections):
    lines = [f"# Review list — {config.RUN_TAG or 'default run'}", "",
             "Each row links to a CSV in this folder, sorted so the rows to check first come first.", "",
             "| # | issue | count | file |", "|---:|---|---:|---|"]
    for i, (title, n, f, *_rest) in enumerate(sections, 1):
        lines.append(f"| {i} | {title} | {n:,} | {f'[{f}]({f})' if f else ''} |")
    lines.append("")
    for i, (title, n, f, what, todo) in enumerate(sections, 1):
        lines += [f"## {i}. {title} ({n:,})", "", what, "", f"**To do:** {todo}", ""]
    (out / "REVIEW.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
