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
from . import profile_table
from .extract_references import numbers_for_ids as ref_numbers_for_ids, sources
from .profile_table import _is_band_row, panel_period
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


# Separators a paper uses to pack more than one component into a single cell.
# ":" needs surrounding spaces: "ChCl : PTSA" is two components, but a bare "1:2" is a
# ratio and "N,N:2" would be part of a name.
_SEPARATORS = ("/", " : ", " + ")


def splittable(text, separator="/"):
    """Is this separator dividing two components, or part of one name?

    Reviews pack mixtures into one cell as "Caffeic acid/Ethylene glycol" or
    "ChCl : PTSA", but "/" also appears inside a single name as a stereodescriptor --
    "D/L-Proline" is racemic proline, not proline plus something called D. Requiring
    every piece to carry at least three letters distinguishes them: on the first
    paper's table this splits all 176 genuine ternary cells and none of the 14
    racemates.
    """
    parts = [p.strip() for p in str(text or "").split(separator)]
    if len(parts) < 2:
        return False
    return all(len(re.findall(r"[A-Za-z]", p)) >= 3 for p in parts)


def split_component_cell(text):
    """One component cell -> the components it names."""
    text = str(text or "").strip()
    for separator in _SEPARATORS:
        if splittable(text, separator):
            return [p.strip() for p in text.split(separator) if p.strip()]
    return [text] if text else []


def parse_components(names):
    """-> ([up to 3 names], flag). A cell may pack several components."""
    comps = []
    for name in names:
        comps += split_component_cell(name)
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


def read_measurement(row, column, profile):
    """One property cell in its row -> (value, temperature_C, marker, note).

    The single entry point for re-reading a value, so `validate` cannot disagree with
    the extractor about what a cell says. It exists because temperature reaches a
    measurement two different ways: from a footnote marker in a wide table, and from a
    sibling column in a paneled one. Reading only the marker would have re-derived
    every one of Fan's 144 measurements at the default 25 C and reported them all as
    fidelity failures.
    """
    value, temperature, marker, note = read_value(row[column.index], column, profile)
    if value is None:
        return value, temperature, marker, note

    condition = _condition_for(column, profile)
    if condition is not None and condition.index < len(row):
        from_column = read_condition(row[condition.index], condition)
        if from_column is not None:
            temperature = from_column
    return value, temperature, marker, note


def _condition_for(column, profile):
    """The condition column governing this property column, within its panel.

    Only paneled tables take temperature from a column; a wide table encodes it in
    footnote markers, and reaching for a stray `condition` column there is actively
    dangerous. A profiler once labelled Sadeghi's melting-point column `condition`,
    and without this guard every one of that table's other properties was re-stamped
    with the melting point as its measurement temperature.
    """
    period = panel_period([c.role for c in profile.columns])
    if period < 2 or "component" in [c.role for c in profile.columns]:
        return None
    start = (column.index // period) * period
    return next((c for c in profile.columns[start:start + period]
                 if c.role == "condition"), None)


def read_references(cell, reference_map):
    """Which references a citation cell points at. -> list[int].

    The cell's own `<xref rid>` links win where the format has them, because they are
    what the publisher asserted rather than what the cell happens to print. Elsevier's
    tables carry no rids, so those fall back to parsing "[40,42-44]" as before.
    """
    linked = ref_numbers_for_ids(cell.ref_ids, reference_map)
    if linked:
        return linked
    return xml_utils.expand_ref_field(cell.text.strip("[]").replace("–", "-"))


def _looks_like_header(row, profile):
    """A repeated header row inside the body: its cells echo the header text."""
    printed = {c.header.strip().lower() for c in profile.columns if c.header.strip()}
    cells = [c.text.strip().lower() for c in row if c.text.strip()]
    return bool(cells) and sum(c in printed for c in cells) >= max(2, len(cells) // 2)


def mixture_record(paper, table, index, ordinal, components, ratios, ratio_raw="",
                   ratio_flag="", component_flag="", ref_text="", ref_numbers=(),
                   cited=None, context=""):
    """The provenance-carrying half of a MixtureRow, shared by both layouts.

    Every mixture row records the same thing about where it came from, whatever the
    table looked like; only *finding* the components differs between a wide table and
    a paneled one.
    """
    cited = cited or {k: "" for k in ("doi", "key", "authors", "title", "journal",
                                      "volume", "issue", "pages", "year")}
    c1, c2, c3 = components
    r1, r2, r3 = ratios
    names = ":".join(n for n in (c1, c2, c3) if n)
    record = {
        "Row_id": f"{paper.slug}:{table.id}:{ordinal:04d}",
        "Mixture_key": mixture_key((c1, c2, c3), ratio_raw),
        "Paper_key": paper.key, "Paper_DOI": paper.doi,
        "Paper_authors": paper.authors, "Paper_title": paper.title,
        "Paper_journal": paper.journal, "Paper_volume": paper.volume,
        "Paper_issue": paper.issue, "Paper_year": paper.year,
        "Table_id": table.id, "Source_row": index,
        "Component_1": c1, "Component_2": c2, "Component_3": c3,
        "Ratio_component_1": r1, "Ratio_component_2": r2, "Ratio_component_3": r3,
        "Ratio_raw": ratio_raw,
        "Mixture": f"{names} ({ratio_raw})" if ratio_raw else names,
        "Ratio_flag": ratio_flag, "Component_flag": component_flag,
        "DOI": paper.doi, "Ref": ref_text,
        "Source_ref_numbers": ",".join(str(n) for n in ref_numbers),
        "Source_DOIs": cited["doi"], "Source_paper_keys": cited["key"],
        "Source_authors": cited["authors"], "Source_titles": cited["title"],
        "Source_journals": cited["journal"], "Source_volumes": cited["volume"],
        "Source_issues": cited["issue"], "Source_pages": cited["pages"],
        "Source_years": cited["year"],
        "Context": context,
    }
    for name in config.PROPERTY_NAMES:
        suffix = name.lower()
        record[name] = None
        record[f"Units_{suffix}"] = None
        record[f"Temperature_{suffix}"] = None
        record[f"Source_col_{suffix}"] = None
    return record


def with_implied(names, profile):
    """Add components the caption states but no column lists. -> list[str].

    Canela-Xandri's Tables 1 and 7 print only the HBA because the caption already says
    every mixture is "PTSA based"; Tables 4 and 5 spell "ChCl : PTSA" out in full. Not
    adding it would leave the same solvent as two unrelated one-component mixtures.

    The cells are split BEFORE the check, or "ChCl : PTSA" reads as one unfamiliar
    string and PTSA gets appended to a mixture that already contains it.
    """
    split = [c for name in names for c in split_component_cell(name)]
    implied = [n for n in (profile.implied_components or []) if n and n.strip()]
    if not implied:
        return split
    present = {_component_key(n) for n in split if n}
    return split + [n for n in implied if _component_key(n) not in present]


def _component_key(name):
    """Normalise a component name for identity comparison.

    Drops a leading stoichiometric coefficient: this paper writes "ChCl : 2PTSA" for
    two equivalents of PTSA, which is the same compound as "PTSA" and must not have a
    second PTSA implied alongside it.
    """
    return re.sub(r"^\d+", "", re.sub(r"\W+", "", str(name or "")).lower())


def read_condition(cell, column):
    """A temperature printed in its own column -> Celsius.

    Fan tabulates T/K alongside every value instead of encoding it in a footnote
    marker, so the conversion the marker path did by lookup has to happen by unit here.
    """
    value = xml_utils.clean_number(cell.text)
    if value is None:
        return None
    unit = str(column.unit_as_written or "").strip().strip("()/ ").lower()
    if unit.startswith("k"):
        return round(value - 273.15, 3)
    if unit.startswith("f"):
        return round((value - 32) * 5 / 9, 3)
    return round(value, 3)


def extract_paneled_table(table, profile, paper, reference_map, definitions=None):
    """A table whose column pattern repeats, one panel per DES. -> (rows, skipped).

    Fan's thermal-conductivity table is 3 panels of (T, lambda) with the DES named in a
    row spanning each panel. `_expand` has already broadcast those spanning cells
    across their columns, so the mixture for a panel is just the band row's text at
    that panel's first column -- no span arithmetic is needed here.
    """
    period = panel_period([c.role for c in profile.columns])
    if period < 2:
        return [], [{"Table_id": table.id, "Source_row": -1,
                     "reason": "paneled layout with no repeating column group",
                     "raw": ""}]

    panels = [profile.columns[start:start + period]
              for start in range(0, len(profile.columns), period)]
    rows, skipped = [], []
    ragged = {index for index, _ in table.ragged}
    current = {}                       # panel number -> the label naming its mixture

    for index, row in enumerate(table.rows):
        if index in ragged or _looks_like_header(row, profile):
            continue
        if _is_band_row(row):
            for n, panel in enumerate(panels):
                first = panel[0].index
                if first < len(row) and row[first].text.strip():
                    current[n] = row[first].text.strip()
            continue

        for n, panel in enumerate(panels):
            label = current.get(n)
            if not label:
                continue
            property_col = next((c for c in panel if c.role == "property"), None)
            condition_col = next((c for c in panel if c.role == "condition"), None)
            if property_col is None or property_col.index >= len(row):
                continue

            value, marker_temp, _marker, note = read_value(
                row[property_col.index], property_col, profile)
            if value is None:
                if note and note not in ("not reported",):
                    skipped.append({"Table_id": table.id, "Source_row": index,
                                    "reason": note,
                                    "raw": row[property_col.index].text[:120]})
                continue

            temperature = marker_temp
            if condition_col is not None and condition_col.index < len(row):
                from_column = read_condition(row[condition_col.index], condition_col)
                if from_column is not None:
                    temperature = from_column

            defined = (definitions or {}).get(_definition_key(label))
            if defined:
                names, ratio_raw = defined["components"], defined["ratio_raw"]
            else:
                names, ratio_raw = [label], ""
            (c1, c2, c3), component_flag = parse_components(with_implied(names, profile))

            record = mixture_record(
                paper, table, index, len(rows) + 1,
                components=(c1, c2, c3),
                ratios=xml_utils.split_ratio(ratio_raw),
                ratio_raw=ratio_raw, component_flag=component_flag,
                cited=sources([], reference_map, paper.key),
                context=f"panel={label}")
            suffix = property_col.property.lower()
            record[property_col.property] = value
            record[f"Units_{suffix}"] = _unit_for(property_col)
            record[f"Temperature_{suffix}"] = temperature
            record[f"Source_col_{suffix}"] = property_col.index
            rows.append(MixtureRow(**record))
    return rows, skipped


def _definition_key(text):
    return re.sub(r"\W+", "", str(text or "")).lower()


def extract_definitions(table, profile, paper):
    """A table that only NAMES mixtures. -> {key: {components, ratio_raw, label}}.

    Fan's Table 2 is the only thing that says what "[ChCl][Gl]3" means, and its data
    tables label their panels with nothing else. This is the same
    abbreviation-is-authoritative idea as component_aliases.json, except the paper
    defines it in its own table so nobody has to write it out by hand.
    """
    by_role = {}
    for column in profile.columns:
        by_role.setdefault(column.role, []).append(column)
    component_cols = by_role.get("component", [])
    ratio_cols = by_role.get("ratio", [])
    # Find the naming column by what its header SAYS, across every role. Fan's
    # "Abbreviation" column was labelled `reference` -- understandable, since
    # "[ChCl][Gl]3" looks like a citation key -- and restricting the search by role
    # would have lost the one column the whole table exists to provide.
    label_cols = [c for c in profile.columns
                  if c.role != "component"
                  and re.search(r"abbrev|acronym|code|symbol|short|label|des\b|name",
                                c.header or "", re.I)]

    out = {}
    for row in table.rows:
        if _looks_like_header(row, profile) or _is_band_row(row):
            continue
        label = next((row[c.index].text.strip() for c in label_cols
                      if c.index < len(row) and row[c.index].text.strip()), "")
        names = [row[c.index].text for c in component_cols if c.index < len(row)]
        names = [n for n in names if n and n.strip()]
        if not label or not names:
            continue
        # A split ratio ("1 | 3") is two ratio columns, one number each.
        parts = [row[c.index].text.strip() for c in ratio_cols if c.index < len(row)]
        parts = [p for p in parts if p]
        ratio_raw = ":".join(parts) if len(parts) > 1 else (parts[0] if parts else "")
        out[_definition_key(label)] = {"label": label, "components": names,
                                       "ratio_raw": ratio_raw}
    return out


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
            cell = row[ref_col.index]
            ref_text = cell.text.strip("[]").replace("–", "-")
            ref_numbers = read_references(cell, reference_map)
        cited = sources(ref_numbers, reference_map, paper.key)

        record = mixture_record(
            paper, table, index, len(rows) + 1,
            components=(c1, c2, c3), ratios=(r1, r2, r3), ratio_raw=ratio_raw,
            ratio_flag=ratio_flag, component_flag=component_flag,
            ref_text=ref_text, ref_numbers=ref_numbers, cited=cited,
            context=" | ".join(f"{c.context_field or c.header}={row[c.index].text}"
                               for c in context_cols if c.index < len(row)
                               and row[c.index].text)[:300])

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
    """Canonicalise the unit the table printed, falling back to the property's own.

    A property config declares dimensionless takes no unit from the header, whatever
    the header says. Tables print "nD" over the refractive index column -- the symbol
    for the quantity (at the sodium D line), not a unit -- and `canonical_unit` keeps
    unrecognised strings verbatim by design, so it was landing in the data as one.

    Unrecognised units on a property that DOES have one are still kept verbatim: a
    table printing Pa·s should read as Pa·s, not be quietly relabelled mPa*s.
    """
    from .extract_text_llm import canonical_unit

    default = config.PROPERTY_UNITS[column.property]
    written = (column.unit_as_written or "").strip().strip("()")
    if written and default:
        canonical = canonical_unit(written, column.property)
        if canonical:
            return canonical
    return default


def _hash(parts):
    import hashlib

    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _component_set(components):
    """The components as an order-independent string. The basis of both keys below."""
    return "+".join(sorted(c.lower() for c in components if c))


def _ratio_key(ratio):
    """The ratio with spacing removed, so "1 : 1" and "1:1" are the same mixture.

    Application tables print "1 : 1" where property tables print "1:2"; without this
    the same solvent from the two routes would key differently and stay two nodes.
    """
    return re.sub(r"\s+", "", str(ratio or ""))


def mixture_key(components, ratio):
    """Identify a DES by what it IS, not by how a table happened to spell it.

    Merging mixtures on their printed name made the same solvent two nodes whenever two
    tables listed its components in a different order -- ten times in this corpus, e.g.
    "Choline chloride:Proline:Malic acid (1:1:1)" and
    "Malic acid:Proline:Choline chloride (1:1:1)". Sorting the component set removes the
    ordering; the name stays on the node for display.
    """
    return _hash([_component_set(components), _ratio_key(ratio)])


def dedup_key(components, ratio, prop, value, temperature, primary_doi):
    """Identify the same underlying datum reported by two different papers.

    Two reviews tabulating the same primary measurement is a real duplicate, and the
    primary DOI is what makes it one: same original study, same mixture, same number.
    Keyed on the resolved component SET so component order cannot split a pair.
    """
    return _hash([
        _component_set(components),
        str(ratio or ""), prop,
        f"{float(value):.6g}" if value is not None else "",
        f"{float(temperature):.6g}" if temperature is not None else "",
        (primary_doi or "").split(config.SOURCE_SEP)[0].lower(),
    ])


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
                Mixture_key=mixture.Mixture_key,
                Paper_key=paper.key, Paper_DOI=paper.doi,
                Table_id=mixture.Table_id, Source_row=mixture.Source_row,
                Source_col=getattr(mixture, f"Source_col_{suffix}", None),
                Mixture=mixture.Mixture,
                Property=name,
                Value=value,
                Unit=getattr(mixture, f"Units_{suffix}"),
                Temperature_C=getattr(mixture, f"Temperature_{suffix}"),
                Source=f"{mixture.Table_id} row {mixture.Source_row}",
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
    """Every profiled table in one paper. -> (mixtures, measurements, skipped).

    Definition tables are read first, because a paneled table's panels are labelled
    with abbreviations that only a definition table explains.
    """
    definitions = {}
    for table in tables:
        profile = profiles.get(table.id)
        if profile is not None and profile.relevant \
                and profile.record_type == "des_definitions":
            found = extract_definitions(table, profile, paper)
            definitions.update(found)
            print(f"    {table.label or table.id}: {len(found)} DES definition(s)")

    mixtures, skipped = [], []
    for table in tables:
        profile = profiles.get(table.id)
        if profile is None or not profile.relevant:
            continue
        if profile.record_type != "des_properties":
            continue                              # applications and definitions elsewhere
        if profile_table.detected_layout(profile, table) == "paneled_by_mixture":
            rows, bad = extract_paneled_table(table, profile, paper, reference_map,
                                              definitions)
        else:
            rows, bad = extract_property_table(table, profile, paper, reference_map)
        mixtures += rows
        skipped += [{**b, "Paper_key": paper.key} for b in bad]
        print(f"    {table.label or table.id}: {len(rows)} rows"
              f"{f' [{profile.layout}]' if profile.layout != 'wide_per_mixture' else ''}"
              f"{f', {len(bad)} unreadable' if bad else ''}")
    return mixtures, to_measurements(mixtures, paper), skipped, definitions


EXTRACTED_TYPES = ("des_properties", "des_application", "des_definitions")


def unhandled(tables, profiles, problems):
    """Tables that produced no data, with the reason. Nothing disappears silently."""
    out = []
    for table in tables:
        profile = profiles.get(table.id)
        if profile is not None and profile.relevant \
                and profile.record_type in EXTRACTED_TYPES:
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
