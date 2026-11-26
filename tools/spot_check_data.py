"""
Draw the human spot-check sample from the graph and find where each value is printed.

    DES_RUN_TAG=corpus_full python tools/spot_check_data.py [--n 200] [--seed 2026]

Runs in the GraphDataBase env (Neo4j + the XML readers). Writes
runs/<tag>/spot_check/sample_data.json, which tools/spot_check.py turns into the
spreadsheet reviewers fill in. Normally you run that script, which calls this one.

Sampling is a simple random sample of the DES property values in the graph, drawn by
ranking every value on a hash of (seed, where it is printed in the paper) and taking
the top n. Because the rank depends only on the location, regenerating after a round
of fixes keeps every value that is still in the graph and replaces only the ones that
left -- the sample stays random and stays mostly the same, so verdicts carry over.
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from des_pipeline import config, store, validate   # noqa: E402
from graph_report import Graph                     # noqa: E402

QUERY = """
MATCH (m) WHERE m.origin IN ['table', 'prose'] AND m.property IS NOT NULL
OPTIONAL MATCH (m)-[:REPORTED_IN]->(p:Paper) WHERE p.extracted
RETURN m.key AS key, m.property AS property, m.value AS value, m.unit AS unit,
       m.temperature_C AS temperature_C, m.uncertainty AS uncertainty,
       m.subject AS subject, m.origin AS origin, m.member_keys AS member_keys,
       m.source_text AS source_text, m.plausible AS plausible,
       collect(DISTINCT p.key) AS papers
"""


def _hash(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _num(value):
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def _blank(value):
    return value is None or str(value).strip() in ("", "nan", "None")


def table_key(r):
    """Where a table value is printed. Independent of Row_id's running counter."""
    return f"{r['Paper_key']}|{r['Table_id']}|r{int(float(r['Source_row']))}" \
           f"|c{int(float(r['Source_col']))}"


def prose_key(r):
    return f"{r['Paper_key']}|{r.get('section_id')}|{r.get('property')}|" \
           f"{r.get('value_as_written')}|{_hash(str(r.get('source_text') or ''))[:10]}"


def members_of(node, table_rows, prose_rows):
    """The extracted rows behind one graph value. -> [(stable key, origin, row)]."""
    out = []
    if node["origin"] == "table":
        for k in node["member_keys"] or []:
            r = table_rows.get(k)
            if r is not None and not _blank(r.get("Source_row")):
                out.append((table_key(r), "table", r))
    else:
        # Prose Measurement_keys are numbered within a paper ("P-0004"), so match on
        # the paper, the key, the value and the quote together.
        for k in node["member_keys"] or []:
            for r in prose_rows.get(k, []):
                if r["Paper_key"] in node["papers"] or not node["papers"]:
                    if _num(r.get("value")) == _num(node["value"]) and \
                            str(r.get("source_text") or "")[:60] == \
                            str(node.get("source_text") or "")[:60]:
                        out.append((prose_key(r), "prose", r))
    return out


def locate_tables(items, papers):
    """Fill in table label, caption, column header and the printed row for table items."""
    by_paper = defaultdict(list)
    for it in items:
        if it["origin"] == "table":
            by_paper[it["row"]["Paper_key"]].append(it)
    ledger = {(t["Paper_key"], t["table_id"]): t for t in store.read_all("tables")}

    for paper_key, group in by_paper.items():
        paper_row = papers.get(paper_key)
        try:
            tables, profiles, _columns = validate._reread_context(paper_row)
        except Exception as exc:                       # noqa: BLE001
            print(f"  could not re-read {paper_key}: {type(exc).__name__}")
            tables, profiles = {}, {}
        for it in group:
            r = it["row"]
            t_id = r["Table_id"]
            entry = ledger.get((paper_key, t_id), {})
            it["table_label"] = entry.get("label") or t_id
            it["caption"] = str(entry.get("caption") or "")
            table, profile = tables.get(t_id), profiles.get(t_id)
            row_i, col_i = int(float(r["Source_row"])), int(float(r["Source_col"]))
            it["row_number"] = row_i + 1
            it["n_rows"] = len(table.rows) if table is not None else None
            if table is None or profile is None or row_i >= len(table.rows):
                continue
            it["column_header"] = " ".join(t for t in table.column(col_i) if t).strip()
            cells = table.rows[row_i]
            label = [cells[c.index].text.strip() for c in profile.columns
                     if c.role in ("mixture", "component", "ratio", "water")
                     and c.index < len(cells) and cells[c.index].text.strip()]
            it["row_label"] = " | ".join(dict.fromkeys(label))
            if col_i < len(cells):
                it["cell_text"] = cells[col_i].text.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", default="2026")
    parser.add_argument("--out", type=Path, default=config.RUN_DIR / "spot_check")
    args = parser.parse_args()

    g = Graph()
    try:
        nodes = g.q(QUERY)
    finally:
        g.close()
    if not nodes:
        sys.exit("no DES property values in the graph -- is it loaded?")

    table_rows = {r["Measurement_key"]: r for r in store.read_all("measurements")
                  if (r.get("status") or "ok") == "ok"}
    prose_rows = defaultdict(list)
    for r in store.read_all("sections_llm"):
        if r.get("status") == "ok":
            prose_rows[r["Measurement_key"]].append(r)
    import pandas as pd
    papers = {p["Paper_key"]: p for p in
              pd.read_csv(config.DATA / "papers.csv").to_dict("records")}

    ranked, unmatched = [], 0
    for node in nodes:
        members = members_of(node, table_rows, prose_rows)
        if not members:
            unmatched += 1
            continue
        # The representative report: fixed by the hash too, so it does not flip
        # between regenerations when a node gains or loses a duplicate.
        key, origin, row = min(members, key=lambda m: _hash(f"{args.seed}|{m[0]}"))
        ranked.append((_hash(f"{args.seed}|{key}"), key, origin, row, node, len(members)))
    ranked.sort(key=lambda x: x[0])

    items = [{"rank": i + 1, "review_key": key, "origin": origin, "row": row,
              "node": node, "n_reports": n}
             for i, (_h, key, origin, row, node, n) in enumerate(ranked[:args.n])]
    locate_tables(items, papers)
    for it in items:
        it["paper"] = {k: papers.get(it["row"]["Paper_key"], {}).get(k) for k in
                       ("Paper_key", "Paper_DOI", "slug", "title", "authors", "journal",
                        "year", "path", "role")}

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "sample_data.json"
    meta = {"run_tag": config.RUN_TAG, "seed": args.seed, "n": args.n,
            "population": len(ranked), "unmatched_nodes": unmatched,
            "population_by_origin": {o: sum(1 for x in ranked if x[2] == o)
                                     for o in ("table", "prose")}}
    path.write_text(json.dumps({"meta": meta, "items": items}, default=str, indent=1),
                    encoding="utf-8")
    print(f"{len(items)} of {len(ranked)} graph values sampled -> {path}"
          + (f"  ({unmatched} values could not be traced to a row)" if unmatched else ""))


if __name__ == "__main__":
    main()
