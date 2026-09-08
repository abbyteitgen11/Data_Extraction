#!/usr/bin/env python3
"""
Pick a handful of downloaded PMC papers to develop and validate the pipeline against.

    python select_papers.py                  # score the corpus, choose ~10
    python select_papers.py --property 6 --application 2 --review 2
    python select_papers.py --score-only     # write the scores, choose nothing

THIS IS A ONE-OFF TOOL, NOT A PIPELINE STEP. It is deliberately absent from
run_pipeline.ALL_STEPS. The scoring below is a convenience for choosing ten papers to
work on by hand -- it encodes guesses about what makes a paper useful, and those
guesses should never silently decide what enters the graph. fetch_pmc.py downloads
without filtering for exactly that reason.

What it does: reads data/review/pmc_corpus.csv, re-reads the promising files to see
what is actually in their tables, and copies the chosen papers to xml/selected/ under
the `<Surname>_<Year>` name the pipeline uses. It does NOT move them into xml/ and does
not run them -- promoting a paper stays a deliberate act.
"""
import argparse
import re
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
from lxml import etree

from des_pipeline import config, dialects, xml_utils

# What a property looks like when it is a table column, rather than a word in a
# sentence. Both the name and the unit count, because a table header is often only the
# symbol: "ρ (g cm-3)" names no property at all in words.
#
# The unit patterns are all anchored on a word boundary. Without it, the thermal
# conductivity unit `w m` matched inside "raw material" and put a bioassay paper in the
# property category.
PROPERTY_PATTERNS = {
    "Density": r"densit|\bg\s*[·⋅/]?\s*cm|\bkg\s*[·⋅/]?\s*m|\bρ\b",
    "Viscosity": r"viscosit|\bmpa\s*[·⋅.]?\s*s|\bcp\b|\bη\b",
    "Conductivity": r"conductivit|\bms\s*[·⋅/]?\s*cm|\bκ\b|\bσ\b",
    "Refractive_index": r"refractive|\bn\s*d\b|\bnd\b|\bn20\b",
    "Surface_tension": r"surface tension|\bmn\s*[·⋅/]\s*m|\bγ\b",
    "Melting_point": r"melting point|freezing point|eutectic point|\btm\b|\btf\b|\btonset\b",
    "Thermal_conductivity": r"thermal conductivit|\bw\s*[·⋅/]\s*m\b",
}

# Does the table say what the solvent is MADE OF? A property table with no composition
# is a table of numbers we cannot attach to a mixture.
COMPOSITION_PATTERNS = {
    "hba_hbd": r"\bhba\b|\bhbd\b|hydrogen bond (?:acceptor|donor)",
    "molar_ratio": r"molar ratio|mole ratio|\b\d\s*:\s*\d\b",
    "known_component": r"choline chloride|\bchcl\b|\burea\b|glycerol|betaine|"
                       r"ethylene glycol|\bdes\b",
}

# The applications route: substrates, yields and products rather than properties.
APPLICATION_PATTERNS = {
    "yield": r"\byield\b|efficiency|recovery|conversion",
    "substrate": r"substrate|feedstock|biomass|sample|analyte|matrix",
    "product": r"\bproduct\b|\bfame\b|lignin|cellulose|extract",
}


def _table_text(root, dialect):
    """The flattened text of every table in the paper, lowercased. -> str."""
    blobs = []
    for table in dialect.tables(root):
        element = getattr(table, "element", None)
        if element is not None:
            blobs.append(etree.tostring(element, method="text", encoding="unicode"))
        else:
            # dialects.Table exposes the parsed grid; fall back to it when the reader
            # keeps no reference to the element. The header matters most of all here --
            # a property is usually named in a column heading and nowhere else.
            for row in table.header + table.rows:
                blobs.append(" ".join(cell.text for cell in row))
            blobs.append(f"{table.label} {table.caption} {table.footnotes}")
    return re.sub(r"\s+", " ", " ".join(blobs)).lower()


def _hits(text, patterns):
    return sorted(name for name, pattern in patterns.items() if re.search(pattern, text))


# A cell that STARTS with a number. `xml_utils.clean_number` is deliberately not used
# here: it is strict, and a real measurement cell is rarely a bare number. Sadeghi's
# Table 2 prints "1.1867a" with the footnote marker inline and "25.4-26.1" for a range,
# so clean_number scored the best table in the corpus 0.08 numeric and this check threw
# it out. Reading the value is extract_table's job; all that matters here is that a
# number is what the cell is trying to say.
_LEADS_WITH_NUMBER = re.compile(r"^[(\[]?\s*[−–—+-]?\s*\d")

# How much of a COLUMN has to be numbers before that column is holding measurements.
# Applied per column, not per table: Sadeghi's Table 2 is only 0.29 numeric overall,
# because three of its ten columns are component names and many property cells are "–",
# yet its density column is 0.97. A whole-table threshold that admits it would have to
# sit so low it admits anything.
NUMERIC_SHARE = 0.5


def property_tables(root, dialect):
    """Properties that some table in this paper actually tabulates. -> (props, n_tables).

    Counting property WORDS anywhere in a table's text does not work, and the first
    version of this file did exactly that. Reviews are full of tables like

        Table 2  Characteristic of DESs        | Property | Characteristic |
        Table 1  Comparison of DESs with ...   | Property | Water | Ionic Liquids | DES |

    which name all six properties and tabulate none of them -- and three such reviews
    came top of the property ranking with six "properties" each.

    So a property has to be named in a COLUMN HEADER and that column's own cells have to
    be mostly numbers. This is deliberately the same evidence `profile_table.validate`
    demands before it will believe a column map -- "a column labelled as a property must
    actually contain numbers" -- so what is being predicted here is extractability, not
    aboutness. Checking per column rather than per table also survives Sadeghi's Table 2,
    where three of ten columns are chemical names.
    """
    found, tables = set(), 0
    for table in dialect.tables(root):
        hit = False
        for index in range(table.n_columns):
            header = " ".join(table.column(index)).lower()
            properties = _hits(header, PROPERTY_PATTERNS)
            if not properties:
                continue
            # `config.DASH` cells are excluded from the denominator, not counted against
            # it. They mean "this paper did not report this property for this mixture",
            # and in a wide compilation table that is most of the column: each of
            # Sadeghi's 1539 rows fills one or two of six property columns, which put
            # every one of them under 0.25 numeric and failed the whole paper.
            column = [row[index].text for row in table.rows
                      if index < len(row) and row[index].text.strip()
                      and row[index].text.strip() not in config.DASH]
            if not column:
                continue
            numeric = sum(1 for text in column if _LEADS_WITH_NUMBER.match(text))
            if numeric / len(column) >= NUMERIC_SHARE:
                found.update(properties)
                hit = True
        tables += 1 if hit else 0
    return sorted(found), tables


def score(row):
    """Read one paper and judge what it could exercise. -> dict of signals."""
    # `score_error`, not `error`: the manifest already has an `error` column and a
    # duplicate silently drops one of the two when the frames are concatenated.
    out = {"property_terms": 0, "properties": "", "numeric_tables": 0, "composition": 0,
           "application_terms": 0, "score": 0, "category": "", "score_error": ""}
    try:
        root = xml_utils.load_root(Path(row["path"]))
        dialect = dialects.detect(root)
        text = _table_text(root, dialect)
        properties, numeric_tables = property_tables(root, dialect)
    except Exception as exc:
        out["score_error"] = f"{type(exc).__name__}: {exc}"[:120]
        return out

    composition = _hits(text, COMPOSITION_PATTERNS)
    applications = _hits(text, APPLICATION_PATTERNS)
    out.update({"property_terms": len(properties), "properties": ";".join(properties),
                "numeric_tables": numeric_tables,
                "composition": len(composition), "application_terms": len(applications)})

    n_tables = int(row.get("n_tables") or 0)
    refs = int(row.get("n_references") or 0)
    is_review = "review" in str(row.get("article_type") or "").lower()

    # A property paper has to have BOTH numbers and a composition to attach them to.
    # A review qualifies here too, and should: Sadeghi is a review, and its tabulated
    # measurements are the best data in the corpus.
    if len(properties) >= 2 and composition:
        out["category"] = "property"
        out["score"] = (10 * len(properties) + 4 * len(composition)
                        + 3 * min(numeric_tables, 6) + min(n_tables, 8))
    elif is_review and refs >= 60:
        out["category"] = "review"
        out["score"] = min(refs, 300) // 10 + 2 * len(properties) + min(n_tables, 8)
    elif len(applications) >= 2 and n_tables >= 2:
        out["category"] = "application"
        out["score"] = 6 * len(applications) + 3 * len(composition) + min(n_tables, 8)
    else:
        out["category"] = "weak"
        out["score"] = len(properties) + len(applications)
    # A paper that only mentions DES in passing is a poor thing to validate against,
    # whatever its tables look like.
    if row.get("phrase_in") != "title_abstract":
        out["score"] = int(out["score"] * 0.5)
    return out


def candidates(frame):
    """The papers worth opening. -> DataFrame.

    Gated on the manifest alone, to avoid re-parsing 13,000 files: a paper needs a DOI
    (run_pipeline refuses one without), at least one table, no read error, and must not
    already be in the corpus.
    """
    keep = frame[
        frame["doi"].notna()
        & (frame["n_tables"].fillna(0) >= 1)
        & frame["error"].isna()
        & frame["already_in_corpus"].isna()
        & frame["phrase_in"].isin(["title_abstract", "body"])
    ].copy()
    # Reviews are wanted for their bibliographies, so they get in on references even
    # when the phrase is only in the body.
    return keep


def choose(scored, quotas):
    """Fill each quota with the best remaining paper that adds a new journal. -> list."""
    picked, journals = [], Counter()
    for category, wanted in quotas.items():
        pool = scored[(scored["category"] == category)].sort_values(
            "score", ascending=False)
        taken = 0
        # Two passes: first only papers from a journal not yet represented, then relax.
        for new_journal_only in (True, False):
            for _, row in pool.iterrows():
                if taken >= wanted:
                    break
                if row["pmcid"] in {p["pmcid"] for p in picked}:
                    continue
                if new_journal_only and journals[row["journal"]]:
                    continue
                picked.append(row.to_dict())
                journals[row["journal"]] += 1
                taken += 1
            if taken >= wanted:
                break
    return picked


def stage(picked, out_dir=None):
    """Copy the chosen papers to xml/selected/ under their pipeline slug. -> list."""
    out_dir = Path(out_dir or config.SELECTED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    used = set()
    for row in picked:
        stem = str(row.get("slug") or row["pmcid"])
        name, suffix = stem, ord("a")
        while name in used or (out_dir / f"{name}.xml").exists():
            suffix += 1
            name = f"{stem}_{chr(suffix)}"
        used.add(name)
        target = out_dir / f"{name}.xml"
        shutil.copy2(row["path"], target)
        row["selected_as"] = str(target)
    return picked


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--property", type=int, default=6)
    parser.add_argument("--application", type=int, default=2)
    parser.add_argument("--review", type=int, default=2)
    parser.add_argument("--score-only", action="store_true",
                        help="write the scores without choosing or copying anything")
    args = parser.parse_args(argv)

    if not config.PMC_CORPUS_CSV.exists():
        raise SystemExit(f"no manifest at {config.PMC_CORPUS_CSV}. Run fetch_pmc.py first.")
    frame = pd.read_csv(config.PMC_CORPUS_CSV)
    print(f"manifest: {len(frame)} downloaded papers")

    pool = candidates(frame)
    print(f"worth opening: {len(pool)} "
          f"(has a DOI, at least one table, readable, not already extracted)")

    scores = [score(row) for _, row in pool.iterrows()]
    scored = pd.concat([pool.reset_index(drop=True), pd.DataFrame(scores)], axis=1)
    scored = scored.sort_values("score", ascending=False)
    print("  categories: " + ", ".join(f"{k} {v}" for k, v in
                                       Counter(scored["category"]).most_common()))

    if args.score_only:
        xml_utils.write_csv(scored.to_dict("records"), config.PAPER_SHORTLIST_CSV)
        return

    quotas = {"property": args.property, "application": args.application,
              "review": args.review}
    picked = stage(choose(scored, quotas))

    print(f"\nselected {len(picked)} papers -> {config.SELECTED_DIR}/")
    print(f"  {'category':<12} {'score':>5}  {'props':<5} {'tabs':>4} {'refs':>5}  "
          f"{'journal':<26} paper")
    for row in sorted(picked, key=lambda r: (r["category"], -r["score"])):
        print(f"  {row['category']:<12} {row['score']:>5}  "
              f"{row['property_terms']:<5} {int(row['n_tables']):>4} "
              f"{int(row['n_references']):>5}  {str(row['journal'])[:25]:<26} "
              f"{Path(row['selected_as']).name}")

    xml_utils.write_csv(picked, config.PAPER_SHORTLIST_CSV)
    print(f"\n  journals represented: {len({p['journal'] for p in picked})}")
    print("  These are staged only. Move a file into xml/ to run it.")


if __name__ == "__main__":
    main()
