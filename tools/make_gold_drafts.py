#!/usr/bin/env python3
"""
Draft the gold-standard files a human corrects, from one run's output.

    python tools/make_gold_drafts.py --tag qwen3            # -> eval/gold/<slug>/*.csv
    python tools/make_gold_drafts.py --tag qwen3 --papers Fan_2024,Azad_2025
    python tools/make_gold_drafts.py --tag qwen3 --force    # overwrite existing drafts

Every row is a PREDICTION, pre-filled into `gold_*` columns, with an empty `verdict`.
Reviewing a row means setting `verdict` to one of:

    correct   the gold_* values are right as drafted
    fix       you corrected one or more gold_* values
    wrong     this item is not real (the paper does not report it) -- excluded from gold
    added     you added this row yourself: something the run missed

Rows left blank are UNREVIEWED and are not scored. eval/ANNOTATION.md has the rules.

Existing files are never overwritten without --force, because they hold your verdicts.
Sadeghi's ~1,650 table values are scored on a sample of source rows (--sample-rows),
recorded in meta.json; its existing spot-check verdicts are carried over as `correct`.
"""
import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GOLD_DIR = Path(os.environ.get("DES_GOLD_DIR") or ROOT / "eval" / "gold")
LARGE_PAPER_ROWS = 200         # a paper with more table rows than this is sampled


def _shown(path):
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def _write(path, rows, columns, force):
    if path.exists() and not force:
        print(f"    kept     {_shown(path)} (exists; --force to redraft)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: ("" if row.get(c) is None else row.get(c)) for c in columns})
    print(f"    wrote    {_shown(path)}  ({len(rows)} rows)")


def _read(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _components(row, n=3):
    names = [row.get(f"Component_{i}") for i in range(1, n + 1)]
    return ";".join(str(c) for c in names if c not in (None, "", "nan"))


def draft_paper(paper_row, args):
    from des_pipeline import config, validate

    slug = paper_row["slug"]
    src = config.PAPERS_DIR / slug
    out = GOLD_DIR / slug
    print(f"  {slug}")

    # ---- tables: is it DES data, and what kinds ----
    ledger = _read(src / "tables.csv")
    tables = [{
        "table_id": r["table_id"], "label": r["label"], "caption": r["caption"][:200],
        "pred_status": r["status"], "pred_content_types": r["content_types"],
        "gold_relevant": "no" if r["status"] in ("not_relevant", "not_profiled", "not_des")
        else "yes",
        "gold_content_types": r["content_types"] if r["status"] not in (
            "not_relevant", "not_profiled", "not_des") else "",
        "verdict": "", "note": "",
    } for r in ledger]
    _write(out / "tables.csv", tables,
           ["table_id", "label", "caption", "pred_status", "pred_content_types",
            "gold_relevant", "gold_content_types", "verdict", "note"], args.force)

    # ---- columns of the tables that were used ----
    columns = []
    profiles_path = src / "table_profiles.json"
    profiles = json.loads(profiles_path.read_text()) if profiles_path.exists() else {}
    labels = {r["table_id"]: r["label"] for r in ledger}
    for table_id, record in profiles.items():
        profile = record.get("profile", {})
        if not profile.get("relevant"):
            continue
        for column in profile.get("columns", []):
            columns.append({
                "table_id": table_id, "label": labels.get(table_id, ""),
                "column": column.get("index"), "header": column.get("header"),
                "pred_role": column.get("role"), "pred_property": column.get("property") or "",
                "gold_role": column.get("role"), "gold_property": column.get("property") or "",
                "verdict": "", "note": "",
            })
    _write(out / "columns.csv", columns,
           ["table_id", "label", "column", "header", "pred_role", "pred_property",
            "gold_role", "gold_property", "verdict", "note"], args.force)

    # ---- the paper's DES dictionary ----
    dictionary = [{
        "kind": r["kind"], "code": r["code"], "source": r["source"],
        "evidence": r["evidence"][:200],
        "gold_components": r["components"],
        "gold_ratio": r.get("ratio_molar") or r.get("ratio_raw") or "",
        "verdict": "" if str(r.get("verified")).lower() not in ("false", "0") else "wrong?",
        "note": r.get("note", ""),
    } for r in _read(src / "des_dictionary.csv")]
    _write(out / "des_dictionary.csv", dictionary,
           ["kind", "code", "source", "evidence", "gold_components", "gold_ratio",
            "verdict", "note"], args.force)

    # ---- measurements: table values (sampled for a large paper) + accepted prose ----
    table_rows = _read(src / "measurements.csv")
    mixture_rows = {r["Row_id"]: r for r in _read(src / "mixtures.csv")}
    source_rows = sorted({(r["Table_id"], r["Source_row"]) for r in table_rows})
    meta = {"slug": slug, "measurement_scope": "all"}
    if len(source_rows) > LARGE_PAPER_ROWS:
        rng = random.Random(0)
        chosen = set(rng.sample(source_rows, min(args.sample_rows, len(source_rows))))
        table_rows = [r for r in table_rows if (r["Table_id"], r["Source_row"]) in chosen]
        meta = {"slug": slug, "measurement_scope": "sampled_rows",
                "rows": sorted(f"{t}:{int(float(r))}" for t, r in chosen)}
        print(f"    sampled  {len(chosen)} of {len(source_rows)} source rows")

    spot = {}
    for path in [config.SHARED_DATA / "review" / f"{slug}_spotcheck.csv",
                 config.REVIEW_DIR / f"{slug}_spotcheck.csv"]:
        for r in _read(path):
            spot[r.get("Review_key")] = r.get("correct")

    raw_cells = {}
    try:
        tables_x, _profiles_x, _cols = validate._reread_context(paper_row)
        for r in table_rows:
            table = tables_x.get(r["Table_id"])
            i, j = int(float(r["Source_row"])), int(float(r["Source_col"]))
            if table is not None and i < len(table.rows) and j < len(table.rows[i]):
                raw_cells[r["Measurement_key"]] = table.rows[i][j].text
    except Exception as exc:                               # noqa: BLE001
        print(f"    (could not re-read raw cells: {type(exc).__name__}: {exc})")

    measurements = []
    for r in table_rows:
        mix = mixture_rows.get(r["Row_id"], {})
        key = f"{r['Table_id']}:{int(float(r['Source_row']))}"
        measurements.append({
            "origin": "table", "table_id": r["Table_id"],
            "source_row": int(float(r["Source_row"])), "source_col": r["Source_col"],
            "raw_cell": raw_cells.get(r["Measurement_key"], ""),
            "evidence": mix.get("Components_written", ""),
            "pred_status": r.get("status", ""),
            "gold_components": _components(mix),
            "gold_ratio": mix.get("Ratio_molar") or mix.get("Ratio_raw") or "",
            "gold_property": r["Property"], "gold_value": r["Value"],
            "gold_unit": r.get("Unit", ""), "gold_temperature_C": r.get("Temperature_C", ""),
            "gold_solute": r.get("Solute", ""),
            "verdict": "correct" if spot.get(key) == "yes" else "",
            "note": "carried over from the spot check" if spot.get(key) == "yes" else "",
        })
    for r in _read(src / "sections_llm.csv"):
        if r.get("status") != "ok":
            continue
        measurements.append({
            "origin": "prose", "table_id": r.get("section_id", ""), "source_row": "",
            "source_col": "", "raw_cell": "", "evidence": r.get("source_text", "")[:300],
            "pred_status": r.get("status", ""),
            "gold_components": r.get("components_resolved", ""),
            "gold_ratio": r.get("molar_ratio", "") or "",
            "gold_property": r.get("property", ""), "gold_value": r.get("value", ""),
            "gold_unit": r.get("unit", ""), "gold_temperature_C": r.get("temperature_C", ""),
            "gold_solute": r.get("Solute", ""), "verdict": "", "note": "",
        })
    _write(out / "measurements.csv", measurements,
           ["origin", "table_id", "source_row", "source_col", "raw_cell", "evidence",
            "pred_status", "gold_components", "gold_ratio", "gold_property", "gold_value",
            "gold_unit", "gold_temperature_C", "gold_solute", "verdict", "note"],
           args.force)

    # ---- mixtures: the distinct DES the paper reports values for ----
    seen, mixtures = set(), []
    in_scope = set(meta.get("rows") or [])
    for r in mixture_rows.values():
        if in_scope and f"{r.get('Table_id')}:{int(float(r.get('Source_row') or 0))}" \
                not in in_scope:
            continue                       # a sampled paper: only the sampled rows
        key = (frozenset(_components(r).lower().split(";")),
               r.get("Ratio_molar") or r.get("Ratio_raw") or "")
        if key in seen:
            continue
        seen.add(key)
        mixtures.append({
            "table_id": r.get("Table_id"), "source_row": r.get("Source_row"),
            "evidence": r.get("Components_written", ""), "des_code": r.get("DES_code", ""),
            "gold_components": _components(r),
            "gold_ratio": r.get("Ratio_molar") or r.get("Ratio_raw") or "",
            "gold_water_fraction": r.get("Water_fraction", ""),
            "verdict": "", "note": r.get("Composition_note", ""),
        })
    _write(out / "mixtures.csv", mixtures,
           ["table_id", "source_row", "evidence", "des_code", "gold_components",
            "gold_ratio", "gold_water_fraction", "verdict", "note"], args.force)

    # ---- applications: which DES was used for what ----
    applications = []
    for r in _read(src / "applications.csv"):
        if (r.get("status") or "ok") != "ok":
            continue
        applications.append({
            "extractor": r.get("Extractor") or "table", "table_id": r.get("Table_id", ""),
            "evidence": (r.get("source_text") or r.get("Detail") or "")[:300],
            "gold_components": _components(r),
            "gold_ratio": r.get("Ratio_molar") or r.get("Ratio_raw") or "",
            "gold_domain": r.get("Domain", ""), "domain_detail": r.get("Domain_detail", ""),
            "verdict": "", "note": "",
        })
    _write(out / "applications.csv", applications,
           ["extractor", "table_id", "evidence", "gold_components", "gold_ratio",
            "gold_domain", "domain_detail", "verdict", "note"], args.force)

    meta_path = out / "meta.json"
    if args.force or not meta_path.exists():
        meta["drafted_from"] = args.tag
        meta_path.write_text(json.dumps(meta, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True, help="the run tag to draft from")
    parser.add_argument("--papers", default="", help="comma-separated slugs (default: all)")
    parser.add_argument("--sample-rows", type=int, default=40,
                        help="source rows sampled from a large paper (Sadeghi)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing drafts -- this DISCARDS your verdicts")
    args = parser.parse_args(argv)

    os.environ["DES_RUN_TAG"] = args.tag
    from des_pipeline import config, store

    if not config.PAPERS_CSV.exists():
        sys.exit(f"no run at {config.DATA}; run the pipeline with DES_RUN_TAG={args.tag}")
    wanted = {s.strip() for s in args.papers.split(",") if s.strip()}
    print(f"drafting gold from run {args.tag!r} ({config.DATA})")
    for paper_row in store.papers():
        if wanted and paper_row["slug"] not in wanted:
            continue
        draft_paper(paper_row, args)
    print(f"\nnext: review eval/gold/<slug>/*.csv (see eval/ANNOTATION.md), then\n"
          f"      python tools/evaluate.py --tag {args.tag}")


if __name__ == "__main__":
    main()
