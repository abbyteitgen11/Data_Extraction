"""
Is the extracted data right? Five separate questions, deliberately kept apart.

  fidelity      Did we transcribe what the paper prints?   -> ours to get right
  skipped cells Did we miss anything the paper prints?     -> the other half of it
  plausibility  Is what the paper prints physically sane?  -> the source's problem
  invariants    Is the dataset internally consistent?      -> structural
  spot check    ...and a human confirms a sample of it.    -> the measured error rate

Fidelity and plausibility must not be conflated. Checking the first paper's three
implausible values against its table showed all three were faithful transcriptions --
the paper really does print a melting point of 2298 C, a viscosity of 325,000, and a
conductivity of 1548. Had those been reported as "extraction errors" we would have
gone looking for a bug that does not exist; had fidelity been reported as "the source
is odd" we would have missed real bugs. So they are measured and reported separately.

Fidelity is the valuable one, because it is exhaustive and needs no judgement: every
measurement records the table, row and column it came from, so the source cell can be
re-read and the value re-derived. That is what makes it safe to change the extractor.

But fidelity only ever looks at values we did extract, so it cannot see a value we
dropped. `skipped_cells` is the complement: it walks every property cell that had
content and produced nothing, and says why. That is how the one genuinely lost value
in this corpus -- a footnote marker trailing its number instead of leading it -- was
found, after fidelity had reported 1648/1648.

    python run_pipeline.py --steps validate            # report card
    python run_pipeline.py --steps validate --review   # + interactive spot check
"""
import random
from collections import Counter

import pandas as pd

from . import config, dialects, paper as paper_mod, profile_table, store, xml_utils
from .extract_table import read_value


# ---------- re-reading the source ----------
def _reread_context(paper_row):
    """Everything needed to read a paper's tables again. -> (tables, profiles, columns).

    Fidelity, the skipped-cell audit and the spot check all re-derive values from the
    XML, so they all need the same four things. Built once here rather than three
    times, and it is what guarantees they agree about what the source says.
    """
    from .schema import TableProfile

    root = xml_utils.load_root(paper_row["path"])
    dialect = dialects.detect(root)
    pap = paper_mod.from_metadata(dialect.paper_metadata(root), paper_row["path"],
                                  dialect.name)
    tables = {t.id: t for t in dialect.tables(root)}

    profiles, columns = {}, {}
    for table_id, record in profile_table.load_overrides(pap).items():
        profile = TableProfile(**record["profile"])
        profiles[table_id] = profile
        columns[table_id] = {c.index: c for c in profile.columns}
    return tables, profiles, columns


# ---------- (a) fidelity: re-read every value from its source cell ----------
def check_fidelity(paper_row, sample=None):
    """Re-derive every measurement from the XML. -> (checked, list of mismatches).

    Exhaustive and automatic. A mismatch means the extractor and the source disagree,
    which is always our bug -- never the paper's.
    """
    rows = [r for r in store.read_all("measurements")
            if r.get("Paper_key") == paper_row["Paper_key"]]
    if not rows:
        return 0, []

    tables, profiles, columns = _reread_context(paper_row)

    if sample is not None and sample < len(rows):
        rows = random.Random(0).sample(rows, sample)

    mismatches, checked = [], 0
    for row in rows:
        table = tables.get(row.get("Table_id"))
        profile = profiles.get(row.get("Table_id"))
        if table is None or profile is None:
            mismatches.append({**_ident(row), "problem": "no table or profile to re-read"})
            continue
        try:
            index, col = int(row["Source_row"]), int(row["Source_col"])
        except (TypeError, ValueError):
            mismatches.append({**_ident(row), "problem": "no source row/column recorded"})
            continue
        if index >= len(table.rows) or col >= len(table.rows[index]):
            mismatches.append({**_ident(row), "problem": "source cell is out of range"})
            continue

        cell = table.rows[index][col]
        spec = columns[row["Table_id"]].get(col)
        value, temperature, _marker, _note = read_value(cell, spec, profile)
        checked += 1
        if value is None or abs(float(value) - float(row["Value"])) > 1e-9:
            mismatches.append({**_ident(row), "problem": "value does not round-trip",
                               "extracted": row["Value"], "re_read": value,
                               "raw_cell": cell.text})
        elif row.get("Temperature_C") is not None and temperature is not None and \
                abs(float(temperature) - float(row["Temperature_C"])) > 1e-9:
            mismatches.append({**_ident(row), "problem": "temperature does not round-trip",
                               "extracted": row["Temperature_C"], "re_read": temperature,
                               "raw_cell": cell.text})
    return checked, mismatches


def _ident(row):
    return {"Paper_key": row.get("Paper_key"), "Measurement_key": row.get("Measurement_key"),
            "Table_id": row.get("Table_id"), "Source_row": row.get("Source_row"),
            "Source_col": row.get("Source_col"), "Property": row.get("Property")}


# ---------- (a2) the other half of fidelity: what did we NOT extract? ----------
def skipped_cells(paper_row, write=True):
    """Every property cell that had content and yielded no value. -> list[dict].

    Fidelity cannot see these, because a value that was never extracted has no row to
    re-read. Most are correct refusals -- this paper's "DT" means "reported at
    different temperatures", which is not a number -- but they are the only place a
    silently lost value can show up, so all of them are written out.
    """
    tables, profiles, columns = _reread_context(paper_row)

    out = []
    for table_id, profile in profiles.items():
        table = tables.get(table_id)
        if table is None or not profile.relevant:
            continue
        property_cols = [c for c in profile.columns if c.role == "property"]
        ragged = {index for index, _ in table.ragged}
        for index, row in enumerate(table.rows):
            if index in ragged:
                continue
            for column in property_cols:
                if column.index >= len(row):
                    continue
                cell = row[column.index]
                value, _temp, _marker, note = read_value(cell, column, profile)
                if value is not None or not note or note == "not reported":
                    continue
                out.append({"Paper_key": paper_row["Paper_key"], "Table_id": table_id,
                            "Source_row": index, "Source_col": column.index,
                            "Property": column.property, "raw": cell.text.strip(),
                            "note": note})
    if write:
        config.REVIEW_DIR.mkdir(parents=True, exist_ok=True)
        path = config.REVIEW_DIR / f"{paper_row['slug']}_skipped_cells.csv"
        pd.DataFrame(out, columns=["Paper_key", "Table_id", "Source_row", "Source_col",
                                   "Property", "raw", "note"]).to_csv(path, index=False)
    return out


# ---------- (b) plausibility: is the printed number physically sane? ----------
def check_plausibility(rows):
    """-> list of rows outside their property's range, annotated.

    These are usually the SOURCE being wrong or using a different unit, not us. They
    are flagged and still loaded, because the graph is meant to be a faithful record
    of the literature; a training set can filter on `plausible`.
    """
    out = []
    for row in rows:
        bounds = config.PLAUSIBLE_RANGE.get(row.get("Property"))
        value = row.get("Value")
        if not bounds or value is None:
            continue
        low, high = bounds
        if not (low <= float(value) <= high):
            out.append({**_ident(row), "Mixture": row.get("Mixture"),
                        "Value": value, "Unit": row.get("Unit"),
                        "note": f"outside the plausible range {low} to {high}"})
    return out


# ---------- (c) structural invariants ----------
def check_invariants():
    """-> list of problems. Cheap, exhaustive, and run every time."""
    problems = []
    measurements = store.read_all("measurements")
    mixtures = store.read_all("mixtures")

    for kind, rows in (("measurements", measurements), ("mixtures", mixtures)):
        missing = [r for r in rows if not r.get("Paper_key")]
        if missing:
            problems.append(f"{len(missing)} {kind} row(s) have no Paper_key")

    keys = Counter(r["Measurement_key"] for r in measurements)
    duplicates = [k for k, n in keys.items() if n > 1]
    if duplicates:
        problems.append(f"{len(duplicates)} Measurement_key(s) are not unique across "
                        f"papers, e.g. {duplicates[:3]}")

    known = {r["key"] for r in store.read_all("references") if r.get("key")}
    cited = {k for r in mixtures
             for k in str(r.get("Source_paper_keys") or "").split(config.SOURCE_SEP) if k}
    orphan = cited - known
    if orphan:
        problems.append(f"{len(orphan)} cited paper key(s) have no reference row, "
                        f"e.g. {sorted(orphan)[:3]}")

    row_ids = {r["Row_id"] for r in mixtures}
    dangling = {r["Row_id"] for r in measurements} - row_ids
    if dangling:
        problems.append(f"{len(dangling)} measurement(s) point at a missing mixture row")
    return problems


def duplicate_report():
    """Measurements sharing a Dedup_key. -> list of groups, written to a CSV.

    A duplicate is the same primary datum reported twice: same components, ratio,
    property, value and originating study. Across papers that is two reviews copying
    one measurement; within a paper it means the table lists it twice.

    Every merge is written out with both values and an `agree` column, because
    collapsing them silently would destroy the evidence that two sources concur -- or
    hide that they do not.
    """
    rows = store.read_all("measurements")
    groups = {}
    for row in rows:
        key = row.get("Dedup_key")
        if key:
            groups.setdefault(key, []).append(row)

    out = []
    for key, members in groups.items():
        if len(members) < 2:
            continue
        values = {round(float(m["Value"]), 6) for m in members}
        papers_involved = sorted({m.get("Paper_key") for m in members})
        out.append({
            "Dedup_key": key,
            "property": members[0].get("Property"),
            "mixture": members[0].get("Mixture"),
            "primary_doi": str(members[0].get("Source_DOIs") or "").split(
                config.SOURCE_SEP)[0],
            "n_rows": len(members),
            "papers": config.SOURCE_SEP.join(papers_involved),
            "cross_paper": len(papers_involved) > 1,
            "values": config.SOURCE_SEP.join(str(v) for v in sorted(values)),
            "agree": len(values) == 1,
            "row_ids": config.SOURCE_SEP.join(m["Row_id"] for m in members),
        })
    out.sort(key=lambda g: (not g["cross_paper"], g["agree"]))
    config.DUPLICATES_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(config.DUPLICATES_CSV, index=False)
    return out


# ---------- (d) the review queue ----------
def review_queue():
    """One prioritised worklist, ranked by how much data hangs on each decision."""
    items = []

    def add(priority, kind, detail, affects, paper=""):
        items.append({"priority": priority, "kind": kind, "detail": detail[:200],
                      "rows_affected": affects, "Paper_key": paper, "verdict": ""})

    for row in store.read_all("tables_unhandled"):
        add(1, "table not extracted", f"{row.get('label')}: {row.get('status')}",
            row.get("n_rows") or 0, row.get("Paper_key", ""))

    prose = store.read_all("sections_llm")
    unresolved = Counter()
    for row in prose:
        for name in str(row.get("unresolved_components") or "").split(";"):
            if name.strip():
                unresolved[name.strip()] += 1
    for name, n in unresolved.most_common():
        add(2, "unresolved abbreviation",
            f"{name!r} -- add it to component_aliases.json if the paper defines it", n)

    measurements = store.read_all("measurements")
    for row in check_plausibility(measurements):
        add(3, "implausible value",
            f"{row['Mixture']} {row['Property']} = {row['Value']} {row['Unit']}: "
            f"{row['note']}", 1, row.get("Paper_key", ""))

    for row in store.read_all("references"):
        agreement = row.get("title_agreement")
        if agreement is not None and float(agreement) < 0.5:
            add(4, "doubtful Crossref match",
                f"ref {row.get('ref_number')} {row.get('doi')}: title agreement "
                f"{agreement} -- {str(row.get('title'))[:60]}", 1,
                row.get("Paper_key", ""))

    for row in prose:
        if row.get("status") == "unverified":
            add(5, "prose value not in the text",
                f"{row.get('components_resolved')} {row.get('property')} = "
                f"{row.get('value')}", 1, row.get("Paper_key", ""))

    items.sort(key=lambda i: (i["priority"], -i["rows_affected"]))
    config.REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(items).to_csv(config.REVIEW_QUEUE_CSV, index=False)
    return items


# ---------- (e) the human spot check ----------
def _render_row(table, profile, columns, index, measurements):
    """One source row as the reviewer should see it. -> list of printable lines.

    A table row usually carries several values -- 2.45 on average here, up to 6 -- so
    showing one sampled measurement next to it invites the reader to mark a correct
    row wrong for the values it appeared to omit. Every property column is printed,
    including the ones that yielded nothing and why, so "nothing was missed" is
    something the reviewer can see rather than assume.
    """
    row = table.rows[index]
    by_property = {m.get("Property"): m for m in measurements}
    lines = []

    labels = []
    for column in profile.columns:
        if column.role in ("component", "ratio") and column.index < len(row):
            text = row[column.index].text.strip()
            if text:
                labels.append(text)
    lines.append("    " + " | ".join(labels))

    for column in profile.columns:
        if column.role != "property" or column.index >= len(row):
            continue
        raw = row[column.index].text.strip() or "(empty)"
        value, temperature, _marker, note = read_value(row[column.index], column, profile)
        if value is None:
            shown = note
        else:
            got = by_property.get(column.property)
            unit = (got or {}).get("Unit") or ""
            shown = f"-> {value} {unit} @ {temperature}C"
            if got is None:
                shown += "   <-- re-read finds a value the data does not have!"
        lines.append(f"      col {column.index}  {column.property or '':<18}"
                     f"{raw[:14]:<16}{shown}")

    for column in profile.columns:
        if column.role == "reference" and column.index < len(row):
            dois = str((measurements[0] if measurements else {}).get("Source_DOIs") or "")
            lines.append(f"      {row[column.index].text.strip()} -> "
                         f"{dois.replace(config.SOURCE_SEP, ', ') or '(unresolved)'}")
    return lines


def spot_check(paper_row, n=20, seed=0, interactive=False):
    """Show sampled source rows in full and record one human verdict per row.

    Verdicts persist in data/review/<slug>_spotcheck.csv and are read back, so a row
    is never asked about twice -- the same hand-edited-file-is-authoritative pattern
    as component_aliases.json.

    Keyed on "<table>:<source row>" rather than on Measurement_key, because Row_id
    embeds a sequential counter that shifts whenever an earlier row's handling
    changes; the table and XML row index do not.
    """
    slug = paper_row["slug"]
    path = config.REVIEW_DIR / f"{slug}_spotcheck.csv"
    answered = {}
    if path.exists():
        for row in pd.read_csv(path).to_dict("records"):
            # rows from the older measurement-keyed format have no Review_key; drop
            # them rather than crash, since their verdicts judged a different view
            if row.get("Review_key") and not pd.isna(row["Review_key"]):
                answered[row["Review_key"]] = row

    rows = [r for r in store.read_all("measurements")
            if r.get("Paper_key") == paper_row["Paper_key"]]
    if not rows:
        return answered

    grouped = {}
    for row in rows:
        grouped.setdefault(f"{row['Table_id']}:{row['Source_row']}", []).append(row)

    keys = sorted(grouped)
    picked = random.Random(seed).sample(keys, min(n, len(keys)))
    todo = [k for k in picked if k not in answered]
    if not interactive:
        return answered

    tables, profiles, columns = _reread_context(paper_row)

    print(f"\n  spot check: {len(todo)} of {len(picked)} sampled rows still unanswered")
    for i, key in enumerate(todo, 1):
        members = grouped[key]
        table_id, index = members[0]["Table_id"], int(members[0]["Source_row"])
        table, profile = tables.get(table_id), profiles.get(table_id)

        print(f"\n {i}/{len(todo)}  {slug}  {table_id} row {index}"
              f"   ({len(members)} value(s))")
        if table is None or profile is None or index >= len(table.rows):
            print("    (source row unavailable)")
        else:
            for line in _render_row(table, profile, columns, index, members):
                print(line)

        answer = input("\n    all correct? [y/n/?/q] ").strip().lower()
        if answer == "q":
            break
        answered[key] = {
            "Review_key": key, "Paper_key": paper_row["Paper_key"],
            "Table_id": table_id, "Source_row": index,
            "Mixture": members[0].get("Mixture"),
            "n_measurements": len(members),
            "values": config.SOURCE_SEP.join(
                f"{m['Property']}={m['Value']}" for m in members),
            "source_row": " | ".join(c.text for c in table.rows[index])[:300]
            if table is not None and index < len(table.rows) else "",
            "correct": {"y": "yes", "n": "no"}.get(answer, "unsure"),
        }
        config.REVIEW_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(list(answered.values())).to_csv(path, index=False)
    return answered


# ---------- (f) the report card ----------
def report(interactive=False, sample=20):
    """Per-paper report card. -> True when every paper is acceptable."""
    papers = store.papers()
    if not papers:
        print("  no papers processed yet")
        return False

    measurements = store.read_all("measurements")
    all_ok = True

    for paper_row in papers:
        rows = [r for r in measurements if r.get("Paper_key") == paper_row["Paper_key"]]
        source_rows = {(r.get("Table_id"), r.get("Source_row")) for r in rows}
        checked, mismatches = check_fidelity(paper_row)
        skipped = skipped_cells(paper_row)
        implausible = check_plausibility(rows)
        unhandled = [r for r in store.read_all("tables_unhandled")
                     if r.get("Paper_key") == paper_row["Paper_key"]]
        verdicts = spot_check(paper_row, n=sample, interactive=interactive)
        mine = [v for v in verdicts.values() if v["Paper_key"] == paper_row["Paper_key"]]
        wrong = sum(1 for v in mine if v["correct"] == "no")
        values_seen = sum(int(v.get("n_measurements") or 0) for v in mine)

        share = (len(implausible) / len(rows) * 100) if rows else 0.0
        reasons = Counter(s["note"] for s in skipped)
        print(f"\n  {paper_row['slug']}   {paper_row['Paper_key']}")
        print(f"    measurements     {len(rows)} from {len(source_rows)} source row(s)")
        print(f"    fidelity         {checked - len(mismatches)}/{checked} re-read identically"
              f"{'   <-- BLOCKER' if mismatches else ''}")
        print(f"    cells skipped    {len(skipped)}"
              f"   ({', '.join(f'{n} {r}' for r, n in reasons.most_common()) or 'none'})")
        print(f"    plausible        {len(rows) - len(implausible)}/{len(rows)}"
              f"   ({len(implausible)} flagged, {share:.1f}%)")
        print(f"    tables skipped   {len(unhandled)}")
        print(f"    spot check       {len(mine)} row(s) reviewed covering {values_seen} "
              f"value(s), {wrong} wrong"
              f"{'' if mine else '   (none yet -- run with --review)'}")

        accepted = (not mismatches and wrong == 0 and share < 2.0
                    and len(mine) >= min(sample, len(source_rows)))
        print(f"    accepted         {accepted}")
        all_ok &= accepted
        for m in mismatches[:5]:
            print(f"      MISMATCH {m['Measurement_key']}: {m['problem']} "
                  f"({m.get('extracted')} vs {m.get('re_read')})")

    problems = check_invariants()
    print(f"\n  invariants       {'all pass' if not problems else f'{len(problems)} FAILED'}")
    for problem in problems:
        print(f"      {problem}")

    duplicates = duplicate_report()
    cross = [d for d in duplicates if d["cross_paper"]]
    disagree = [d for d in duplicates if not d["agree"]]
    print(f"  duplicates       {len(duplicates)} group(s) "
          f"({len(cross)} across papers, {len(disagree)} disagreeing) "
          f"-> {config.DUPLICATES_CSV.name}")
    for group in disagree[:4]:
        print(f"      DISAGREE {group['mixture']} {group['property']}: "
              f"{group['values']}  ({group['papers']})")

    items = review_queue()
    print(f"  review queue     {len(items)} item(s) -> {config.REVIEW_QUEUE_CSV.name}")
    for item in items[:6]:
        print(f"      P{item['priority']} {item['kind']:<26}{item['detail'][:64]}")
    return all_ok and not problems
