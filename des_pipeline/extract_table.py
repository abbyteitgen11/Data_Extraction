"""
The table route: read a property table using the profile that describes it.

Nothing here knows which column holds density. That comes from a TableProfile, which
`profile_table.py` derives from the table's own caption, headers and legend. So the
same code reads a table it has never seen, and a paper whose table cannot be profiled
extracts nothing rather than guessing -- missing beats wrong.

Two rules carry most of the weight:

  * A superscript counts as a footnote marker only if the profile's LEGEND defines it.
    Previously a hard-coded map meant `[Br-]` and `[N1116(2OH)+]` had their charge
    signs read as temperature markers.
  * Every value records where it came from (table id, row, column), so
    `validate.check_fidelity` can re-read the source cell and prove the number
    round-trips. That is the pipeline's own regression test.

Output is wide -- one row per mixture, a value/unit/temperature triple per property --
because that is how a table reads; `to_measurements` derives the long view the graph
and any ML training want.
"""
import re

from . import config, xml_utils
from .extract_references import sources
from .schema import MeasurementRow, MixtureRow, TableRow  # re-exported for the driver


def parse_ratio(text, markers=()):
    """Read a molar-ratio cell -> (raw text, [r1, r2, r3], flag)."""
    raw = (text or "").strip()
    flag = ""
    if "i" in markers:
        flag = "weight_ratio"
    if "us" in markers or raw.startswith("us"):
        flag = "unstable"
    if raw.startswith("C") or "C" in markers:
        flag = "unknown_ratio"
    stripped = re.sub(r"^(i|us|C-?)", "", raw)      # "i6:4" -> "6:4"
    return raw, xml_utils.split_ratio(stripped), flag


def splittable(text):
    """Is a "/" in this cell separating two components, or part of one name?

    Reviews pack ternary mixtures into one cell as "Caffeic acid/Ethylene glycol",
    but "/" also appears inside a single name as a stereodescriptor -- "D/L-Proline"
    is racemic proline, not proline plus something called D. Requiring every piece to
    carry at least three letters distinguishes them: on the real table this splits all
    176 genuine ternary cells and none of the 14 racemates.
    """
    parts = [p.strip() for p in str(text or "").split("/")]
    if len(parts) < 2:
        return False
    return all(len(re.findall(r"[A-Za-z]", p)) >= 3 for p in parts)


def parse_components(names):
    """-> ([up to 3 names], flag). A cell may pack several components with "/"."""
    comps = []
    for name in names:
        text = str(name or "").strip()
        if not text:
            continue
        if splittable(text):
            comps += [c.strip() for c in text.split("/") if c.strip()]
        else:
            comps.append(text)
    flag = "quaternary+" if len(comps) > 3 else ""
    return (comps + [None, None, None])[:3], flag


def read_value(cell, column, profile):
    """One property cell -> (value, temperature_C, marker, note).

    The marker lookup is the important part: a superscript is only a footnote marker
    if this table's legend says so, otherwise it is just part of the chemistry.

    `note` says why a cell yielded nothing, so `validate.skipped_cells` can show a
    human that "DT" was recognised and declined rather than quietly missed. It is
    derived from the cell's own text and decides nothing.
    """
    raw = cell.text.strip()
    missing = set(profile.missing_value_tokens) | set(config.DASH)
    if not raw or raw in missing:
        return None, None, "", "not reported"

    defined = {m.marker: m for m in profile.footnote_markers if m.marker}
    marker = next((m for m in cell.markers if m in defined), "")
    if not marker:                                   # some markers only appear in the text
        marker = next((k for k in defined if raw.startswith(k) and len(k) <= 2), "")

    # A marker usually leads the number ("d1.267") but occasionally trails it
    # ("0.688a"), and the trailing form was silently unreadable. Only accept the
    # trailing strip when what is left is actually a number, so a unit suffix or a
    # word ending in a marker letter cannot be mistaken for one.
    text = raw
    if marker and raw.startswith(marker):
        text = raw[len(marker):]
    elif marker and raw.endswith(marker) and \
            xml_utils.clean_number(raw[:-len(marker)]) is not None:
        text = raw[:-len(marker)]

    value = xml_utils.clean_number(text)
    if value is None:
        # No digits at all means the cell holds a token standing in for a value --
        # this table's "DT" ("reported at different temperatures"). Digits that still
        # will not parse are a source typo or two numbers in one cell.
        note = "unparseable" if any(c.isdigit() for c in raw) else "no numeric value"
        return None, None, marker, note

    spec = defined.get(marker)
    if spec is not None and spec.meaning == "temperature" and spec.temperature_C is not None:
        temperature = round(spec.temperature_C, 3)
    elif profile.default_temperature_C is not None:
        temperature = round(profile.default_temperature_C, 3)
    else:
        temperature = config.DEFAULT_TEMP
    return value, temperature, marker, ""


def _looks_like_header(row, profile):
    """A repeated header row inside the body: its cells echo the header text."""
    printed = {c.header.strip().lower() for c in profile.columns if c.header.strip()}
    cells = [c.text.strip().lower() for c in row if c.text.strip()]
    return bool(cells) and sum(c in printed for c in cells) >= max(2, len(cells) // 2)


def extract_property_table(table, profile, paper, reference_map):
    """-> (list[MixtureRow], list[dict]) -- the rows, and the ones we could not read."""
    by_role = {}
    for column in profile.columns:
        by_role.setdefault(column.role, []).append(column)
    component_cols = by_role.get("component", [])
    ratio_col = next(iter(by_role.get("ratio", [])), None)
    ref_col = next(iter(by_role.get("reference", [])), None)
    property_cols = by_role.get("property", [])
    context_cols = by_role.get("context", [])

    rows, skipped = [], []
    ragged = {index for index, _ in table.ragged}

    for index, row in enumerate(table.rows):
        if index in ragged:
            skipped.append({"Table_id": table.id, "Source_row": index,
                            "reason": "cell count did not fit the grid",
                            "raw": " | ".join(c.text for c in row)[:300]})
            continue
        if _looks_like_header(row, profile):
            continue

        names = [row[c.index].text for c in component_cols if c.index < len(row)]
        if not any(n.strip() for n in names):
            continue                                  # a spacer or continuation row

        (c1, c2, c3), component_flag = parse_components(names)
        if ratio_col is not None and ratio_col.index < len(row):
            cell = row[ratio_col.index]
            ratio_raw, (r1, r2, r3), ratio_flag = parse_ratio(cell.text, cell.markers)
        else:
            ratio_raw, (r1, r2, r3), ratio_flag = "", (None, None, None), ""

        ref_text, ref_numbers = "", []
        if ref_col is not None and ref_col.index < len(row):
            ref_text = row[ref_col.index].text.strip("[]").replace("–", "-")
            ref_numbers = xml_utils.expand_ref_field(ref_text)
        cited = sources(ref_numbers, reference_map, paper.key)

        mixture_names = ":".join(n for n in (c1, c2, c3) if n)
        record = {
            "Row_id": f"{paper.slug}:{table.id}:{len(rows) + 1:04d}",
            "Paper_key": paper.key, "Paper_DOI": paper.doi,
            "Paper_authors": paper.authors, "Paper_title": paper.title,
            "Paper_journal": paper.journal, "Paper_volume": paper.volume,
            "Paper_issue": paper.issue, "Paper_year": paper.year,
            "Table_id": table.id, "Source_row": index,
            "Component_1": c1, "Component_2": c2, "Component_3": c3,
            "Ratio_component_1": r1, "Ratio_component_2": r2, "Ratio_component_3": r3,
            "Ratio_raw": ratio_raw,
            "Mixture": f"{mixture_names} ({ratio_raw})" if ratio_raw else mixture_names,
            "Ratio_flag": ratio_flag, "Component_flag": component_flag,
            "DOI": paper.doi, "Ref": ref_text,
            "Source_ref_numbers": ",".join(str(n) for n in ref_numbers),
            "Source_DOIs": cited["doi"], "Source_paper_keys": cited["key"],
            "Source_authors": cited["authors"], "Source_titles": cited["title"],
            "Source_journals": cited["journal"], "Source_volumes": cited["volume"],
            "Source_issues": cited["issue"], "Source_pages": cited["pages"],
            "Source_years": cited["year"],
            "Context": " | ".join(f"{c.context_field or c.header}={row[c.index].text}"
                                  for c in context_cols if c.index < len(row)
                                  and row[c.index].text)[:300],
        }

        # Every declared property gets its triple; the ones this table lacks stay null.
        for name in config.PROPERTY_NAMES:
            suffix = name.lower()
            record[name] = None
            record[f"Units_{suffix}"] = None
            record[f"Temperature_{suffix}"] = None
            record[f"Source_col_{suffix}"] = None
        for column in property_cols:
            if column.index >= len(row):
                continue
            value, temperature, _marker, _note = read_value(row[column.index], column,
                                                            profile)
            suffix = column.property.lower()
            record[column.property] = value
            record[f"Units_{suffix}"] = (
                _unit_for(column) if value is not None else None)
            record[f"Temperature_{suffix}"] = temperature
            record[f"Source_col_{suffix}"] = column.index
        rows.append(MixtureRow(**record))
    return rows, skipped


def _unit_for(column):
    """Canonicalise the unit the table printed, falling back to the property's own."""
    from .extract_text_llm import canonical_unit

    written = (column.unit_as_written or "").strip().strip("()")
    if written:
        canonical = canonical_unit(written, column.property)
        if canonical:
            return canonical
    return config.PROPERTY_UNITS[column.property]


def dedup_key(components, ratio, prop, value, temperature, primary_doi):
    """Identify the same underlying datum reported by two different papers.

    Two reviews tabulating the same primary measurement is a real duplicate, and the
    primary DOI is what makes it one: same original study, same mixture, same number.
    Keyed on the resolved component SET so component order cannot split a pair.
    """
    import hashlib

    parts = [
        "+".join(sorted(c.lower() for c in components if c)),
        str(ratio or ""), prop,
        f"{float(value):.6g}" if value is not None else "",
        f"{float(temperature):.6g}" if temperature is not None else "",
        (primary_doi or "").split(config.SOURCE_SEP)[0].lower(),
    ]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


def to_measurements(rows, paper):
    """Flatten wide mixture rows into one MeasurementRow per reported value."""
    out = []
    for mixture in rows:
        for name in config.PROPERTY_NAMES:
            value = getattr(mixture, name, None)
            if value is None:
                continue
            suffix = name.lower()
            out.append(MeasurementRow(
                Measurement_key=f"{mixture.Row_id}:{name}",
                Row_id=mixture.Row_id,
                Paper_key=paper.key, Paper_DOI=paper.doi,
                Table_id=mixture.Table_id, Source_row=mixture.Source_row,
                Source_col=getattr(mixture, f"Source_col_{suffix}", None),
                Mixture=mixture.Mixture,
                Property=name,
                Value=value,
                Unit=getattr(mixture, f"Units_{suffix}"),
                Temperature_C=getattr(mixture, f"Temperature_{suffix}"),
                Source=f"{mixture.Table_id} row {mixture.Source_row}",
                Locus=f"row {mixture.Source_row}",
                Source_ref_numbers=mixture.Source_ref_numbers,
                Source_DOIs=mixture.Source_DOIs,
                Source_paper_keys=mixture.Source_paper_keys,
                Dedup_key=dedup_key(
                    [mixture.Component_1, mixture.Component_2, mixture.Component_3],
                    mixture.Ratio_raw, name, value,
                    getattr(mixture, f"Temperature_{suffix}"), mixture.Source_DOIs),
                **_plausibility(name, value),
            ))
    return out


def _plausibility(prop, value):
    """Flag a value outside its property's physical range. It still loads."""
    bounds = config.PLAUSIBLE_RANGE.get(prop)
    if not bounds or value is None:
        return {}
    low, high = bounds
    if low <= float(value) <= high:
        return {}
    return {"plausible": False,
            "plausibility_note": f"outside the plausible range {low} to {high}"}


def extract_tables(tables, profiles, paper, reference_map):
    """Every profiled property table in one paper. -> (mixtures, measurements, skipped)."""
    mixtures, skipped = [], []
    for table in tables:
        profile = profiles.get(table.id)
        if profile is None or not profile.relevant:
            continue
        if profile.record_type != "des_properties":
            print(f"    {table.label or table.id}: {profile.record_type} -- no extractor "
                  f"for that record type yet, skipping")
            continue
        rows, bad = extract_property_table(table, profile, paper, reference_map)
        mixtures += rows
        skipped += [{**b, "Paper_key": paper.key} for b in bad]
        print(f"    {table.label or table.id}: {len(rows)} rows"
              f"{f', {len(bad)} unreadable' if bad else ''}")
    return mixtures, to_measurements(mixtures, paper), skipped


def unhandled(tables, profiles, problems):
    """Tables that produced no data, with the reason. Nothing disappears silently."""
    out = []
    for table in tables:
        profile = profiles.get(table.id)
        if profile is not None and profile.relevant and profile.record_type == "des_properties":
            continue
        if profile is None:
            reason = "; ".join(problems.get(table.id, ["no usable profile"]))[:300]
        elif not profile.relevant:
            reason = f"profiled as not relevant: {profile.reason}"[:300]
        else:
            reason = f"record_type {profile.record_type}, no extractor yet"
        out.append(TableRow(table_id=table.id, label=table.label,
                            caption=table.caption[:300], n_rows=len(table.rows),
                            status=reason))
    return out
