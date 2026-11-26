"""
Human spot check of the graph: a sheet colleagues can fill in without knowing the pipeline.

    # draw the sample and write the sheet (base env; calls spot_check_data.py itself)
    /opt/anaconda3/bin/python tools/spot_check.py --tag corpus_full

    # after fixing things: regenerate, carrying over verdicts that still apply
    /opt/anaconda3/bin/python tools/spot_check.py --tag corpus_full \
        --previous runs/corpus_full/spot_check/spot_check_2026-11-24_filled.xlsx

    # score a filled-in sheet
    /opt/anaconda3/bin/python tools/spot_check.py --tag corpus_full \
        --score runs/corpus_full/spot_check/spot_check_2026-11-24_filled.xlsx

Writes runs/<tag>/spot_check/spot_check_<date>.xlsx (dropdowns, instructions sheet,
frozen header) and the same rows as .csv. Never overwrites an existing sheet.

Options: --n (default 200), --seed (default 2026; change it only to draw a NEW sample),
--double N (mark N items for a second, independent reviewer), --no-refresh (reuse
sample_data.json instead of querying the graph again).
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GRAPH_PYTHON = os.environ.get("DES_GRAPH_PYTHON",
                              "/opt/anaconda3/envs/GraphDataBase/bin/python")

PROPERTY_LABEL = {
    "Melting_point": "Melting point", "Boiling_point": "Boiling point",
    "Density": "Density", "Viscosity": "Viscosity", "Conductivity": "Electrical conductivity",
    "Thermal_conductivity": "Thermal conductivity", "Surface_tension": "Surface tension",
    "Refractive_index": "Refractive index", "pH": "pH",
    "Decomposition_temperature": "Decomposition temperature",
    "Glass_transition": "Glass transition temperature", "Zeta_potential": "Zeta potential",
    "Heat_capacity": "Heat capacity", "Polarity_ET30": "Polarity E_T(30)",
    "Solubility": "Solubility",
}
UNIT_LABEL = {"C": "°C", "g*cm^-3": "g/cm³", "mPa*s": "mPa·s", "mS*cm^-1": "mS/cm",
              "mN*m^-1": "mN/m", "W*m^-1*K^-1": "W/(m·K)", "J*g^-1*K^-1": "J/(g·K)",
              "kcal*mol^-1": "kcal/mol", "mg*mL^-1": "mg/mL", "mV": "mV"}
TEMP_SOURCE = {
    "cell_inline": "written in the cell", "marker": "from a footnote marker",
    "condition_column": "from a temperature column", "column_header": "from the column heading",
    "caption_default": "from the caption/legend",
}
PHASE = {"Melting_point", "Boiling_point", "Decomposition_temperature", "Glass_transition"}

# The reviewer's columns. Order here is order in the sheet.
CHECKS = [
    ("Found in paper?", "Could you find this value at the place given? Yes / No"),
    ("DES correct?", "Is it the right DES: same components and ratio (and water, if stated)?"),
    ("Property correct?", "Is the property right (e.g. viscosity, not kinematic viscosity)?"),
    ("Value correct?", "Does 'Value as printed' match the paper exactly?"),
    ("Unit correct?", "Is 'Unit as printed' right, and is 'Stored as' the same quantity "
                      "in the converted unit?"),
    ("Temperature correct?", "Is the measuring temperature right? 'Not stated' is correct "
                             "only if the paper really gives none. N/A for melting points "
                             "etc."),
]
CHOICES = ["Yes", "No", "Unsure", "N/A"]
REVIEWER_COLS = [c for c, _ in CHECKS] + ["Reviewer", "Comments"]
HIDDEN = ["_review_key", "_signature", "_origin", "_property", "_paper_key", "_rank"]


# --------------------------------------------------------------------------- building
def _fmt(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x).strip()


def _first_author(authors):
    first = str(authors or "").split(";")[0].strip()
    if not first:
        return ""
    surname = first.split(",")[0].strip() if "," in first else first.split()[-1]
    return f"{surname} et al."


def _pmc_link(paper):
    text = f"{paper.get('slug') or ''} {paper.get('path') or ''}"
    m = re.search(r"PMC\d+", text)
    return f"https://pmc.ncbi.nlm.nih.gov/articles/{m.group(0)}/" if m else ""


def _signature(it):
    """What the reviewer judged. A carried-over verdict is only valid if this is equal."""
    n = it["node"]
    return "|".join(_fmt(v) for v in (n.get("subject"), n.get("property"), n.get("value"),
                                      n.get("unit"), n.get("temperature_C")))


def sheet_row(i, it, double):
    n, r, p = it["node"], it["row"], it["paper"]
    table = it["origin"] == "table"
    prop = n.get("property")

    if table:
        caption = re.sub(r"\s+", " ", it.get("caption") or "")
        where = f"{it.get('table_label') or 'Table'}" + (
            f" — “{caption[:110]}{'…' if len(caption) > 110 else ''}”" if caption else "")
        row_label = it.get("row_label") or \
            f"(no DES name in this row; it is named in a heading row above: {n.get('subject')})"
        n_rows = it.get("n_rows")
        row_no = f"{it.get('row_number')} of {n_rows}" if n_rows else _fmt(it.get("row_number"))
        column = it.get("column_header") or ""
        printed = it.get("cell_text") or _fmt(r.get("Value_as_written"))
        unit_printed = r.get("Unit_as_written") or ""
    else:
        where = f"Main text, section “{r.get('section_title') or '?'}”"
        row_label = f"Sentence: “{(r.get('source_text') or '').strip()}”"
        row_no, column = "", ""
        printed = _fmt(r.get("value_as_written") or r.get("value"))
        unit_printed = r.get("unit_raw") or ""

    t = n.get("temperature_C")
    if prop in PHASE:
        temp = "N/A (the value is itself a temperature)"
    elif t is None:
        temp = "Not stated"
    else:
        how = TEMP_SOURCE.get(r.get("Temperature_source") or "", "")
        temp = f"{_fmt(t)}" + (f" ({how})" if how else "")

    notes = []
    if str(r.get("Literature_value")).lower() in ("true", "1"):
        notes.append("Literature value the paper reprints from another study — check it "
                     "is printed here as recorded")
    if _fmt(r.get("Header_scale")) not in ("", "1"):
        notes.append(f"The column heading carries a factor ({_fmt(r.get('Header_scale'))}); "
                     "the stored value applies it")
    if r.get("Qualifier"):
        notes.append(f"Printed with “{r['Qualifier']}”")
    if n.get("uncertainty") is not None:
        notes.append(f"Uncertainty recorded: ± {_fmt(n['uncertainty'])}")
    if r.get("Solute") or r.get("Solute") == 0:
        notes.append(f"Solute: {r['Solute']}")
    if it.get("n_reports", 1) > 1:
        notes.append(f"Same value also reported in {it['n_reports'] - 1} other place(s)")
    if double:
        notes.append("DOUBLE CHECK: a second person reviews this one independently")

    row = {
        "ID": f"S{i:03d}",
        "Paper": f"{_first_author(p.get('authors'))} ({_fmt(p.get('year'))[:4]}) "
                 f"{p.get('title') or ''}".strip(),
        "Journal": p.get("journal") or "",
        "DOI link": f"https://doi.org/{p['Paper_DOI']}" if p.get("Paper_DOI") else "",
        "Free full text": _pmc_link(p),
        "Where in the paper": where,
        "Column heading": column,
        "Row (as printed)": row_label,
        "Row number": row_no,
        "DES (as recorded)": n.get("subject") or "",
        "Property": PROPERTY_LABEL.get(prop, prop),
        "Value as printed": printed,
        "Unit as printed": unit_printed,
        "Stored as": f"{_fmt(n.get('value'))} {UNIT_LABEL.get(n.get('unit'), n.get('unit') or '')}".strip(),
        "Measuring temperature (°C)": temp,
        "Notes": "; ".join(notes),
    }
    row.update({c: "" for c in REVIEWER_COLS})
    row.update({"_review_key": it["review_key"], "_signature": _signature(it),
                "_origin": it["origin"], "_property": prop,
                "_paper_key": p.get("Paper_key") or "", "_rank": it["rank"]})
    return row


def read_sheet(path):
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        return pd.read_excel(path, sheet_name="Values", dtype=str).fillna("")
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")


def _is_second(row):
    return str(row["ID"]).endswith("b")


def _slot(row):
    return (row["_review_key"], _is_second(row))


def carry_over(df, previous):
    """Copy verdicts from a filled sheet where the same value is still recorded the same
    way. -> (df, n carried, n changed since checked)."""
    old = read_sheet(previous)
    old = old[old[REVIEWER_COLS[:len(CHECKS)]].ne("").any(axis=1)]
    # A second-review copy ("S001b") only ever inherits from a second-review copy, so
    # the first reviewer's answer never leaks into the independent check.
    by_key = {_slot(r): r for r in old.to_dict("records")}
    carried = changed = 0
    df["Checked before"] = ""
    for i, row in df.iterrows():
        prev = by_key.get(_slot(row))
        if prev is None:
            continue
        if prev.get("_signature") == row["_signature"]:
            for c in REVIEWER_COLS:
                df.at[i, c] = prev.get(c, "")
            df.at[i, "Checked before"] = "yes — unchanged, verdict carried over"
            carried += 1
        else:
            df.at[i, "Checked before"] = "yes — but the extracted value has CHANGED; " \
                                         "please check again"
            changed += 1
    return df, carried, changed


def write_xlsx(df, path, meta):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active
    ws.title = "Values"
    cols = list(df.columns)
    ws.append(cols)
    for row in df.itertuples(index=False):
        ws.append([str(v) if v is not None else "" for v in row])

    head = Font(bold=True)
    fill_info, fill_check = PatternFill("solid", fgColor="E8EEF7"), PatternFill("solid", fgColor="FFF4D6")
    widths = {"Paper": 40, "Where in the paper": 45, "Row (as printed)": 45,
              "DES (as recorded)": 32, "Notes": 40, "Comments": 40, "Column heading": 22,
              "Checked before": 30, "DOI link": 28, "Free full text": 28}
    for j, name in enumerate(cols, 1):
        cell = ws.cell(row=1, column=j)
        cell.font = head
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        cell.fill = fill_check if name in REVIEWER_COLS else fill_info
        letter = get_column_letter(j)
        ws.column_dimensions[letter].width = widths.get(name, 16)
        if name.startswith("_"):
            ws.column_dimensions[letter].hidden = True
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            cell.number_format = "@"          # text: Excel must not turn 1:2 into a time
    for j, name in enumerate(cols, 1):
        if name in ("DOI link", "Free full text"):
            for i in range(2, ws.max_row + 1):
                c = ws.cell(row=i, column=j)
                if c.value:
                    c.hyperlink, c.style = c.value, "Hyperlink"
    dv = DataValidation(type="list", formula1='"' + ",".join(CHOICES) + '"',
                        allow_blank=True)
    ws.add_data_validation(dv)
    for name, _ in CHECKS:
        letter = get_column_letter(cols.index(name) + 1)
        dv.add(f"{letter}2:{letter}{ws.max_row}")
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = ws.dimensions

    info = wb.create_sheet("Instructions", 0)
    lines = [
        ("Spot check of extracted DES property values", True),
        ("", False),
        (f"Each row of the 'Values' sheet is one value, picked at random from the "
         f"database ({meta['n']} of {meta['population']:,}). Please check each one "
         "against the paper.", False),
        ("", False),
        ("How to check a row", True),
        ("1. Open the paper with 'Free full text' (or 'DOI link').", False),
        ("2. Go to 'Where in the paper'. For a table, find the column under 'Column "
         "heading' and the row shown under 'Row (as printed)'; 'Row number' counts data "
         "rows below the heading. For main text, search for the quoted sentence.", False),
        ("3. Compare with the blue columns, then fill in the yellow ones from the "
         "dropdowns: Yes / No / Unsure / N/A.", False),
        ("4. If anything is No or Unsure, say what the paper actually shows in "
         "'Comments' (e.g. 'value is 1.183', 'ratio is 1:3', 'column is kinematic "
         "viscosity').", False),
        ("5. Put your initials in 'Reviewer'.", False),
        ("", False),
        ("What each yellow column asks", True),
    ] + [(f"• {name} — {desc}", False) for name, desc in CHECKS] + [
        ("", False),
        ("Good to know", True),
        ("• 'Value as printed' and 'Unit as printed' are what the paper shows. "
         "'Stored as' is the same value converted to a standard unit — only check that it "
         "is the same quantity (e.g. 1.2 g/mL = 1.2 g/cm³; 1200 kg/m³ = 1.2 g/cm³).", False),
        ("• In review articles the value is usually cited from another study. Check only "
         "that it is printed in THIS paper as recorded; you do not need to open the "
         "original study.", False),
        ("• Judge each row on its own. Do not skip a row because it looks fine — "
         "please confirm it.", False),
        ("• If a row is marked DOUBLE CHECK, fill it in without looking at anyone "
         "else's answer.", False),
        ("• If you cannot find the value at all, set 'Found in paper?' to No, leave the "
         "rest blank, and describe what you found in 'Comments'.", False),
        ("• Please do not sort the sheet, delete columns, or edit the blue columns. "
         "Hidden columns at the end are used to read the answers back.", False),
        ("", False),
        (f"Sample: seed {meta['seed']}, run {meta['run_tag']}, generated {date.today()}.",
         False),
    ]
    for text, bold in lines:
        info.append([text])
        info.cell(row=info.max_row, column=1).font = Font(bold=bold, size=13 if bold else 11)
        info.cell(row=info.max_row, column=1).alignment = Alignment(wrap_text=True)
    info.column_dimensions["A"].width = 120
    wb.save(path)


# --------------------------------------------------------------------------- scoring
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def score(path, out_dir):
    df = read_sheet(path)
    answered = df[df["Found in paper?"].str.strip() != ""].copy()
    if answered.empty:
        sys.exit("no rows have 'Found in paper?' filled in yet")

    def norm(v):
        v = str(v).strip().lower()
        return {"y": "yes", "n": "no", "na": "n/a"}.get(v, v)

    for name, _ in CHECKS:
        answered[name] = answered[name].map(norm)
    fields = [c for c, _ in CHECKS[1:]]

    def verdict(r):
        if r["Found in paper?"] == "no":
            return "wrong"
        vals = [r[f] for f in fields]
        if "no" in vals:
            return "wrong"
        if "unsure" in vals or r["Found in paper?"] == "unsure" or "" in vals:
            return "unsure"
        return "correct"

    answered["verdict"] = answered.apply(verdict, axis=1)
    # Second-review copies measure agreement only; counting them in the accuracy would
    # weight those values twice.
    second = answered[answered.apply(_is_second, axis=1)]
    answered = answered[~answered.apply(_is_second, axis=1)]
    df = df[~df.apply(_is_second, axis=1)]
    decided = answered[answered["verdict"] != "unsure"]
    k, n = int((decided["verdict"] == "correct").sum()), len(decided)
    lo, hi = wilson(k, n)

    lines = [f"# Spot-check result — {Path(path).name}", "",
             f"Rows answered: {len(answered)} of {len(df)}; unsure or incomplete: "
             f"{int((answered['verdict'] == 'unsure').sum())}.", "",
             f"**Fully correct: {k} of {n} = {k / n:.1%} "
             f"(95 % CI {lo:.1%}–{hi:.1%}, Wilson).**", "",
             "## Per field", "", "| field | yes | no | unsure | n/a |", "|---|---:|---:|---:|---:|"]
    for name, _ in CHECKS:
        c = answered[name].value_counts()
        lines.append(f"| {name} | {c.get('yes', 0)} | {c.get('no', 0)} | "
                     f"{c.get('unsure', 0)} | {c.get('n/a', 0)} |")
    for group, label in (("_origin", "Where the value came from"), ("Property", "Property")):
        lines += ["", f"## By {label.lower()}", "", f"| {label} | correct | checked | % |",
                  "|---|---:|---:|---:|"]
        for g, sub in decided.groupby(group):
            kk = int((sub["verdict"] == "correct").sum())
            lines.append(f"| {g} | {kk} | {len(sub)} | {kk / len(sub):.0%} |")

    # Agreement on double-checked items: the first and second review of one value.
    first = dict(zip(answered["_review_key"], answered["verdict"]))
    pairs = [(first[k], v) for k, v in zip(second["_review_key"], second["verdict"])
             if k in first]
    if pairs:
        agree = sum(1 for a, b in pairs if a == b)
        # Cohen's kappa on correct / wrong / unsure.
        cats = sorted({c for p in pairs for c in p})
        pe = sum((sum(a == c for a, _ in pairs) / len(pairs)) *
                 (sum(b == c for _, b in pairs) / len(pairs)) for c in cats)
        po = agree / len(pairs)
        kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
        lines += ["", "## Agreement between reviewers", "",
                  f"{agree} of {len(pairs)} double-checked values got the same verdict "
                  f"({po:.0%}; Cohen's kappa {kappa:.2f})."]
        disagree = [k for (k, v) in zip(second["_review_key"], second["verdict"])
                    if k in first and first[k] != v]
        if disagree:
            ids = answered.set_index("_review_key").loc[disagree, "ID"].tolist()
            lines.append(f"Disagreements to resolve: {', '.join(ids)}.")

    wrong = answered[answered["verdict"] == "wrong"]
    if not wrong.empty:
        lines += ["", "## Values marked wrong", "",
                  "| ID | paper | where | DES | property | stored | comment |",
                  "|---|---|---|---|---|---|---|"]
        for r in wrong.to_dict("records"):
            cells = [r["ID"], r["Paper"][:40], r["Where in the paper"][:40],
                     r["DES (as recorded)"], r["Property"], r["Stored as"],
                     r["Comments"]]
            lines.append("| " + " | ".join(str(c).replace("|", "/") for c in cells) + " |")

    out = out_dir / f"{Path(path).stem}_score.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:6]))
    print(f"\nwrote {out}")


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", default="2026")
    parser.add_argument("--double", type=int, default=20,
                        help="how many items to mark for a second reviewer (0 = none)")
    parser.add_argument("--previous", help="a filled sheet whose verdicts to carry over")
    parser.add_argument("--score", help="score a filled sheet instead of drawing one")
    parser.add_argument("--no-refresh", action="store_true")
    parser.add_argument("--force", action="store_true", help="overwrite today's sheet")
    args = parser.parse_args()

    out_dir = ROOT / "runs" / args.tag / "spot_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.score:
        score(args.score, out_dir)
        return

    data = out_dir / "sample_data.json"
    if not args.no_refresh or not data.exists():
        env = {**os.environ, "DES_RUN_TAG": args.tag}
        subprocess.run([GRAPH_PYTHON, str(ROOT / "tools" / "spot_check_data.py"),
                        "--n", str(args.n), "--seed", str(args.seed)],
                       cwd=ROOT, env=env, check=True)
    payload = json.loads(data.read_text(encoding="utf-8"))
    meta, items = payload["meta"], payload["items"]

    # The double-check items are the first --double by rank: random, and stable.
    rows = [sheet_row(i, it, it["rank"] <= args.double) for i, it in enumerate(items, 1)]
    df = pd.DataFrame(rows)
    if args.double:
        twins = df[df["_rank"] <= args.double].copy()
        twins["ID"] = twins["ID"] + "b"
        df = pd.concat([df, twins], ignore_index=True)

    carried = changed = 0
    if args.previous:
        df, carried, changed = carry_over(df, args.previous)
    df = df[[c for c in df.columns if c not in HIDDEN] + HIDDEN]

    stem = out_dir / f"spot_check_{date.today()}"
    xlsx, csv = stem.with_suffix(".xlsx"), stem.with_suffix(".csv")
    if (xlsx.exists() or csv.exists()) and not args.force:
        sys.exit(f"{xlsx.name} already exists -- it may hold answers. Use --force, or "
                 f"rename the filled sheet first.")
    df.to_csv(csv, index=False, encoding="utf-8-sig")
    write_xlsx(df, xlsx, meta)

    by = df.drop_duplicates("_review_key")["_origin"].value_counts().to_dict()
    print(f"{len(items)} values ({by.get('table', 0)} from tables, {by.get('prose', 0)} "
          f"from text) + {args.double} second-review copies -> {xlsx}")
    print(f"same rows as CSV -> {csv}")
    if args.previous:
        print(f"carried over {carried} verdict(s); {changed} value(s) changed since they "
              f"were checked and need checking again")


if __name__ == "__main__":
    main()
