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
from dataclasses import dataclass

from . import config, units, xml_utils
from . import profile_table
from .composition import compose, des_problem, has_ionic_liquid
from .extract_references import numbers_for_ids as ref_numbers_for_ids, sources
from .profile_table import _is_band_row, panel_period
from .schema import (MeasurementRow, MixtureRow, TableLedgerRow,  # re-exported for the driver
                     TableRow)


# What a ratio marker MEANS, by the meaning the profile's legend gave it. This used to
# be the literals "i", "us" and "C" -- the letters one paper's legend happens to use --
# tested directly against the cell text, which is precisely the hard-coding the profiler
# exists to remove. Sadeghi's legend defines those three with these meanings, so its
# output is unchanged; a paper using different letters now works too.
_RATIO_FLAGS = {"ratio_basis": "weight_ratio", "stability": "unstable"}


def parse_ratio(text, markers=(), profile=None):
    """Read a molar-ratio cell -> (raw text, [r1, r2, r3], flag)."""
    raw = (text or "").strip()
    flag = ""
    defined = {m.marker: m for m in (profile.footnote_markers if profile else [])
               if m.marker}
    # A marker attached to the cell, or written at the front of its text ("i6:4").
    present = [m for m in markers if m in defined]
    present += [m for m in defined if raw.startswith(m) and len(m) <= 2
                and m not in present]
    for marker in present:
        mapped = _RATIO_FLAGS.get(defined[marker].meaning)
        if mapped:
            flag = mapped
        elif defined[marker].meaning == "not_reported":
            flag = "unknown_ratio"
    stripped = raw
    for marker in sorted(present, key=len, reverse=True):
        if stripped.startswith(marker):
            stripped = stripped[len(marker):].lstrip("-").strip()
            break
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


# A stoichiometric coefficient written onto the front of a name: "ChCl : 2PTSA" means
# two parts PTSA, not a chemical called 2PTSA. The digit must be followed directly by a
# LETTER, which is what separates it from the 81 component names that legitimately begin
# with a digit -- 1-Butanol, 2-EH, 1,2-Propanediol all have punctuation or a hyphen next.
# Verified against all 565 names in the corpus: it fires on 2PTSA and nothing else.
_COEFFICIENT = re.compile(r"^(\d{1,2})([A-Za-z].*)$")


def split_coefficient(name):
    """'2PTSA' -> ('PTSA', 2). -> (name, None) when there is no coefficient."""
    match = _COEFFICIENT.match(str(name or "").strip())
    if not match:
        return name, None
    return match.group(2).strip(), int(match.group(1))


def parse_components(names):
    """-> ([up to 3 names], flag, [coefficients]). A cell may pack several components."""
    comps, coefficients = [], []
    for name in names:
        for piece in split_component_cell(name):
            clean, coefficient = split_coefficient(piece)
            comps.append(clean)
            coefficients.append(coefficient)
    flag = "quaternary+" if len(comps) > 3 else ""
    return ((comps + [None, None, None])[:3], flag,
            (coefficients + [None, None, None])[:3])


def ratio_from_coefficients(coefficients, n_components):
    """'ChCl : 2PTSA' with no ratio column -> "1:2". -> "" when none were written.

    The coefficient IS the stoichiometry when a table states no ratio of its own, which
    is the case for the lignocellulose table -- dropping it would lose the only thing
    distinguishing ChCl:2PTSA from ChCl:PTSA.

    `n_components` trims the padding: the component list is always three long, so
    without it a two-component mixture reports a three-part ratio.
    """
    used = list(coefficients)[:max(0, n_components)]
    if not any(used):
        return ""
    return ":".join(str(c or 1) for c in used)


@dataclass
class ParsedCell:
    """What one printed measurement cell says, before any conversion.

    `clean_number` returns None for every form below, so all of them were being dropped
    as "unparseable". Counted over the ten new papers, in columns a property extractor
    actually reads: 130 cells carry an uncertainty, 25 an inline temperature, 15 an
    inline citation. Ranges and "<0.001" appear only in FTIR and p-value columns, so
    those are recorded and refused rather than guessed at.
    """

    value: float = None
    uncertainty: float = None
    low: float = None
    high: float = None
    qualifier: str = ""
    temperature_C: float = None          # from an inline "(40 °C)"
    ref_numbers: tuple = ()              # from an inline "(ref. 22)"
    note: str = ""                       # why nothing parsed, for the skipped-cell audit
    status: str = ""                     # keeps the row out of the graph


_UNCERTAINTY = re.compile(r"^\s*([-−+]?[\d.,]+)\s*(?:±|\+/-|\+-)\s*([\d.,]+)")
_RANGE = re.compile(r"^\s*([-−]?[\d.,]+)\s*[–—]\s*([-−]?[\d.,]+)\s*$")
_QUALIFIED = re.compile(r"^\s*([<>≤≥~≈])\s*([-−]?[\d.,]+)\s*$")
# "(40 °C)", "(at 25 C)", "(298 K)" trailing a value.
_INLINE_TEMPERATURE = re.compile(
    r"\(\s*(?:at\s*)?(\d{1,3}(?:\.\d+)?)\s*(?:°|˚|º)?\s*([CK])\s*\)", re.I)
# "(ref. 22)", "(ref 22)", "(Ref. 22, 23)"
_INLINE_REF = re.compile(r"\(\s*refs?\.?\s*([\d,\s–—-]+)\)", re.I)


def parse_measurement_cell(text, markers=(), profile=None):
    """One printed cell -> ParsedCell. No conversion, no unit knowledge.

    Everything here is about what the cell SAYS. Turning kelvin into Celsius or
    kg m-3 into g cm-3 happens later, in `units`, once the column's unit is known.
    """
    out = ParsedCell()
    raw = str(text or "").strip()
    if not raw:
        out.note = "not reported"
        return out

    # Peel the annotations off first, so what is left is just the number. An inline
    # temperature is the strongest statement of a measurement condition there is -- the
    # paper put it on that one value -- so it outranks the footnote marker and the
    # table default.
    match = _INLINE_TEMPERATURE.search(raw)
    if match:
        value = float(match.group(1))
        if match.group(2).upper() == "K":
            out.temperature_C = round(value - 273.15, 3) if 200 <= value <= 500 else None
        elif -100 <= value <= 500:
            out.temperature_C = value
        raw = _INLINE_TEMPERATURE.sub(" ", raw).strip()

    match = _INLINE_REF.search(raw)
    if match:
        out.ref_numbers = tuple(xml_utils.expand_ref_field(match.group(1)))
        raw = _INLINE_REF.sub(" ", raw).strip()

    match = _UNCERTAINTY.match(raw)
    if match:
        out.value = xml_utils.clean_number(match.group(1))
        out.uncertainty = xml_utils.clean_number(match.group(2))
        if out.value is not None:
            return out

    match = _RANGE.match(raw)
    if match:
        out.low = xml_utils.clean_number(match.group(1))
        out.high = xml_utils.clean_number(match.group(2))
        if out.low is not None and out.high is not None:
            # Deliberately no midpoint. The component route already refuses to average a
            # range ("emit low and high as two records"), and a mean the paper never
            # printed is a number nobody can check against the source.
            out.status = "range_only"
            out.note = "a range, not a value"
            return out

    match = _QUALIFIED.match(raw)
    if match:
        out.qualifier = {"≤": "<", "≥": ">", "≈": "~"}.get(match.group(1), match.group(1))
        out.value = xml_utils.clean_number(match.group(2))
        if out.value is not None:
            return out

    out.value = xml_utils.clean_number(raw)
    if out.value is None:
        # A number followed by a BRACKETED annotation we could not classify -- "0.50 (0)".
        # The value is not in doubt, only the annotation, so it is kept with a note.
        #
        # The bracket requirement is not cosmetic. Without it this also matched
        # "b140 84.5" (one cell holding two values, of which it silently kept the first)
        # and "1. 2201" (a typo for 1.2201, which it read as 1.0 -- a density wrong by
        # 20%). Both had previously been reported as unparseable, which was right:
        # a half-read cell is worse than an unread one, because nothing downstream can
        # tell it happened.
        annotated = re.match(r"^\s*([-−]?[\d.,]+)\s*\([^)]*\)\s*$", raw)
        if annotated and xml_utils.clean_number(annotated.group(1)) is not None:
            out.value = xml_utils.clean_number(annotated.group(1))
            out.note = f"unexplained annotation after the value: {raw[:40]!r}"
        elif re.match(r"^\s*[-−]?[\d.,]+\s+[-−]?[\d.,]+\s*$", raw):
            out.note = "two values in one cell"
        else:
            out.note = ("unparseable" if any(c.isdigit() for c in raw)
                        else "no numeric value")
    return out


def read_value(cell, column, profile):
    """One property cell -> (value, temperature_C, marker, note).

    The marker lookup is the important part: a superscript is only a footnote marker
    if this table's legend says so, otherwise it is just part of the chemistry.

    `note` says why a cell yielded nothing, so `validate.skipped_cells` can show a
    human that "DT" was recognised and declined rather than quietly missed. It is
    derived from the cell's own text and decides nothing.

    Kept as the two-value entry point every caller already uses; `read_cell` below
    returns everything the cell said, for the callers that build a measurement row.
    """
    parsed, _marker, temperature, _source = read_cell(cell, column, profile)
    return parsed.value, temperature, _marker, parsed.note


def read_cell(cell, column, profile):
    """One property cell -> (ParsedCell, marker, temperature_C, temperature_source).

    Temperature precedence, strongest evidence first:
      cell_inline       the cell itself says "(40 °C)"
      marker            a footnote marker this table's legend defines
      column_header     the header says "298 K" where a unit would go
      caption_default   the caption or legend states one for the whole table
      unstated          nothing said so -- and it stays null.

    There is deliberately no last resort. `config.DEFAULT_TEMP = 25` used to be one, and
    it stamped a condition the paper never reported onto every unmarked value.
    """
    raw = cell.text.strip()
    missing = set(profile.missing_value_tokens) | set(config.DASH)
    if not raw or raw in missing:
        return ParsedCell(note="not reported"), "", None, "unstated"

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

    parsed = parse_measurement_cell(text, cell.markers, profile)
    if parsed.value is None and parsed.low is None:
        return parsed, marker, None, "unstated"

    spec = defined.get(marker)
    header = " ".join([column.header or "", column.unit_as_written or ""])
    if parsed.temperature_C is not None:
        return parsed, marker, parsed.temperature_C, "cell_inline"
    if spec is not None and spec.meaning == "temperature" and spec.temperature_C is not None:
        return parsed, marker, round(spec.temperature_C, 3), "marker"
    from_header = units.temperature_in_header(header)
    if from_header is not None:
        return parsed, marker, from_header, "column_header"
    if profile.default_temperature_C is not None:
        return parsed, marker, round(profile.default_temperature_C, 3), "caption_default"
    return parsed, marker, None, "unstated"


def read_measurement_cell(row, column, profile):
    """One property cell in its row -> (ParsedCell, marker, temperature, source).

    The single entry point for reading a value, so `validate` cannot disagree with the
    extractor about what a cell says. It exists because temperature reaches a
    measurement several ways, and a sibling condition column is one of them: reading
    only the marker would have re-derived every one of Fan's 144 measurements at the old
    default 25 C and reported them all as fidelity failures.

    A condition column outranks a footnote marker -- it states the temperature for that
    ROW, where a marker states it for a whole class of cells -- but it does not outrank
    the cell writing "(40 °C)" on itself.
    """
    parsed, marker, temperature, source = read_cell(row[column.index], column, profile)
    if parsed.value is None and parsed.low is None:
        return parsed, marker, temperature, source
    if source == "cell_inline":
        return parsed, marker, temperature, source

    condition = _condition_for(column, profile)
    if condition is not None and condition.index < len(row):
        from_column = read_condition(row[condition.index], condition)
        if from_column is not None:
            return parsed, marker, from_column, "condition_column"
    return parsed, marker, temperature, source


def read_measurement(row, column, profile):
    """The 4-tuple view, for callers that only need the value. -> (value, T, marker, note)."""
    parsed, marker, temperature, _source = read_measurement_cell(row, column, profile)
    return parsed.value, temperature, marker, parsed.note


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


def mixture_record(paper, table, index, ordinal, comp, ratio_flag="", ref_text="",
                   ref_numbers=(), cited=None, context=""):
    """The provenance-carrying half of a MixtureRow, shared by every layout.

    `comp` is a composition.Composition: what the DES is made of, however the table
    wrote it. Only *finding* the composition differs between layouts.
    """
    cited = cited or {k: "" for k in ("doi", "key", "authors", "title", "journal",
                                      "volume", "issue", "pages", "year")}
    (c1, c2, c3), (r1, r2, r3), component_flag = comp.slots()
    record = {
        "Row_id": f"{paper.slug}:{table.id}:{ordinal:04d}",
        "Mixture_key": mixture_key((c1, c2, c3), comp.key_ratio),
        "Paper_key": paper.key, "Paper_DOI": paper.doi,
        "Paper_authors": paper.authors, "Paper_title": paper.title,
        "Paper_journal": paper.journal, "Paper_volume": paper.volume,
        "Paper_issue": paper.issue, "Paper_year": paper.year,
        "Table_id": table.id, "Source_row": index,
        "Component_1": c1, "Component_2": c2, "Component_3": c3,
        "Ratio_component_1": r1, "Ratio_component_2": r2, "Ratio_component_3": r3,
        "Ratio_raw": comp.ratio_raw,
        "Mixture": comp.label(),
        "Ratio_flag": ratio_flag, "Component_flag": component_flag,
        "Components_written": ";".join(comp.components_written)[:300],
        "Ratio_molar": comp.ratio_molar,
        "Ratio_source": comp.ratio_source,
        "Composition_basis": comp.basis,
        "Water_fraction": comp.water_fraction,
        "Water_basis": comp.water_basis,
        "DES_code": comp.code,
        "Composition_note": "; ".join(comp.notes)[:300],
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


def extract_paneled_table(table, profile, paper, reference_map, dictionary=None,
                          resolve=None, strict=""):
    """A table whose column pattern repeats, one panel per DES. -> (rows, skipped, extras).

    Fan's thermal-conductivity table is 3 panels of (T, lambda) with the DES named in a
    row spanning each panel. `_expand` has already broadcast those spanning cells
    across their columns, so the mixture for a panel is just the band row's text at
    that panel's first column -- no span arithmetic is needed here. The label is read
    through the paper's own dictionary, which is what says "[ChCl][Gl]3" means
    choline chloride and glycerol at 1:3.
    """
    period = panel_period([c.role for c in profile.columns])
    if period < 2:
        return [], [{"Table_id": table.id, "Source_row": -1,
                     "reason": "paneled layout with no repeating column group",
                     "raw": ""}], {}

    panels = [profile.columns[start:start + period]
              for start in range(0, len(profile.columns), period)]
    rows, skipped, extras = [], [], {}
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

            parsed, _marker, temperature, source = read_cell(
                row[property_col.index], property_col, profile)
            if parsed.value is None and parsed.low is None:
                if parsed.note and parsed.note not in ("not reported",):
                    skipped.append({"Table_id": table.id, "Source_row": index,
                                    "reason": parsed.note,
                                    "raw": row[property_col.index].text[:120]})
                continue

            # The panel's own condition column, not `_condition_for`: the panel is the
            # scope here, and the columns have already been sliced into panels above.
            if source != "cell_inline" and condition_col is not None \
                    and condition_col.index < len(row):
                from_column = read_condition(row[condition_col.index], condition_col)
                if from_column is not None:
                    temperature, source = from_column, "condition_column"

            comp = compose([label], dictionary=dictionary, resolve=resolve,
                           implied=profile.implied_components or ())
            problem = _refusal(comp, dictionary, strict)
            if problem:
                skipped.append({"Table_id": table.id, "Source_row": index,
                                "reason": f"not extracted: {problem}", "raw": label[:120]})
                continue
            record = mixture_record(
                paper, table, index, len(rows) + 1, comp,
                cited=sources([], reference_map, paper.key),
                context=f"panel={label}")
            fields = measurement_fields(parsed, property_col, table,
                                        property_col.property, temperature, source,
                                        solute_in_row(row, profile))
            suffix = property_col.property.lower()
            record[property_col.property] = fields["Value"]
            record[f"Units_{suffix}"] = fields["Unit"]
            record[f"Temperature_{suffix}"] = fields["Temperature_C"]
            record[f"Source_col_{suffix}"] = property_col.index
            extras[(record["Row_id"], property_col.property)] = fields
            rows.append(MixtureRow(**record))
    return rows, skipped, extras


def _definition_key(text):
    return re.sub(r"\W+", "", str(text or "")).lower()


def extract_definitions(table, profile, paper, resolve=None, dictionary=None):
    """A table that says what each DES is. -> {code: Composition}.

    Fan's Table 2 is the only thing that says what "[ChCl][Gl]3" means, and its data
    tables label their panels with nothing else; Solcan's design table names runs
    N1..N39 with an HBA, an HBD, a ratio and a water percentage. The code is read from
    whichever column NAMES the row (found by its header, across every role -- Fan's
    "Abbreviation" column was labelled `reference`); when there is none, the DES's own
    printed formula is its code.
    """
    label_cols = [c for c in profile.columns
                  if c.role not in ("component", "property", "ratio", "water")
                  and _LABEL_FIELD.search(f"{_printed(table, c)} {c.context_field or ''}")]
    mixture_cols = [c for c in profile.columns if c.role == "mixture"]

    out = {}
    for row in table.rows:
        if _looks_like_header(row, profile) or _is_band_row(row) \
                or _is_group_row(row, table):
            continue
        comp, _flag = row_composition(row, table, profile, dictionary, resolve)
        if des_problem(comp):
            continue
        label = comp.code or next((row[c.index].text.strip() for c in label_cols
                                   if c.index < len(row) and row[c.index].text.strip()
                                   and row[c.index].text.strip() not in config.DASH), "")
        if not label:
            label = next((row[c.index].text.strip() for c in mixture_cols
                          if c.index < len(row) and row[c.index].text.strip()), "")
        # A run or entry NUMBER ("1", "2.1") names the row, not the DES: a later "2"
        # anywhere in the paper would otherwise resolve to it.
        if not label or not re.search(r"[A-Za-z]", label):
            label = comp.label()
        out.setdefault(label, comp)
    return out


def extract_property_table(table, profile, paper, reference_map, dictionary=None,
                           resolve=None, strict=""):
    """-> (rows, skipped, extras).

    `extras` carries everything a measurement knows that a wide mixture row has no
    column for -- the as-written value and unit, the uncertainty, the temperature's
    provenance, the status. Keyed by (Row_id, property) and consumed by
    `to_measurements`.

    A row is written to mixtures.csv only if it yielded at least one value. A row that
    only DEFINES a DES (a composition, a code, a design-of-experiments run) belongs in
    the paper's dictionary, not among the measured mixtures.
    """
    extras = {}
    by_role = {}
    for column in profile.columns:
        by_role.setdefault(column.role, []).append(column)
    name_cols = by_role.get("mixture", []) + by_role.get("component", [])
    ref_col = next(iter(by_role.get("reference", [])), None)
    property_cols = by_role.get("property", [])
    water_ids = {c.index for c in water_columns(profile, table)}
    context_cols = [c for c in by_role.get("context", []) + by_role.get("other_quantity", [])
                    if c.index not in water_ids]

    rows, skipped = [], []
    ragged = {index for index, _ in table.ragged}

    for index, row in enumerate(table.rows):
        if index in ragged:
            skipped.append({"Table_id": table.id, "Source_row": index,
                            "reason": "cell count did not fit the grid",
                            "raw": " | ".join(c.text for c in row)[:300]})
            continue
        if _looks_like_header(row, profile) or _is_group_row(row, table):
            continue

        names = [row[c.index].text for c in name_cols if c.index < len(row)]
        if not any(n.strip() and n.strip() not in config.DASH for n in names):
            continue                                  # a spacer or continuation row

        comp, ratio_flag = row_composition(row, table, profile, dictionary, resolve)
        problem = _refusal(comp, dictionary, strict)
        if problem:
            skipped.append({"Table_id": table.id, "Source_row": index,
                            "reason": f"not extracted: {problem}",
                            "raw": " | ".join(n for n in names if n)[:300]})
            continue

        ref_text, ref_numbers = "", []
        if ref_col is not None and ref_col.index < len(row):
            cell = row[ref_col.index]
            ref_text = cell.text.strip("[]").replace("–", "-")
            ref_numbers = read_references(cell, reference_map)
        cited = sources(ref_numbers, reference_map, paper.key)

        record = mixture_record(
            paper, table, index, len(rows) + 1, comp, ratio_flag=ratio_flag,
            ref_text=ref_text, ref_numbers=ref_numbers, cited=cited,
            context=" | ".join(f"{c.context_field or c.header}={row[c.index].text}"
                               for c in context_cols if c.index < len(row)
                               and row[c.index].text)[:300])

        solute = solute_in_row(row, profile)
        found = 0
        pending = {}
        for column in property_cols:
            if column.index >= len(row):
                continue
            parsed, _marker, temperature, source = read_measurement_cell(
                row, column, profile)
            if parsed.value is None and parsed.low is None:
                if parsed.note and parsed.note != "not reported":
                    skipped.append({"Table_id": table.id, "Source_row": index,
                                    "reason": parsed.note,
                                    "raw": row[column.index].text[:120]})
                continue
            fields = measurement_fields(parsed, column, table, column.property,
                                        temperature, source, solute)
            suffix = column.property.lower()
            record[column.property] = fields["Value"]
            record[f"Units_{suffix}"] = fields["Unit"]
            record[f"Temperature_{suffix}"] = fields["Temperature_C"]
            record[f"Source_col_{suffix}"] = column.index
            pending[column.property] = fields
            found += 1
        if not found:
            continue
        for prop, fields in pending.items():
            extras[(record["Row_id"], prop)] = fields
        rows.append(MixtureRow(**record))
    return rows, skipped, extras


# A context column naming what dissolved. Solubility is a property of a solute IN the
# DES, so without this the number says nothing and two solutes would merge on Dedup_key.
_SOLUTE_FIELD = re.compile(r"solute|analyte|drug|compound|substrate|api\b|solubil",
                           re.I)


def solute_in_row(row, profile):
    """What dissolved, from whichever context column names it. -> str."""
    for column in profile.columns:
        if column.role != "context" or column.index >= len(row):
            continue
        name = f"{column.context_field or ''} {column.header or ''}"
        if _SOLUTE_FIELD.search(name):
            text = row[column.index].text.strip()
            if text and text not in config.DASH:
                return text[:120]
    return ""


def measurement_fields(parsed, column, table, prop, temperature, source, solute=""):
    """Everything a MeasurementRow needs about one value. -> dict.

    This is where a printed number becomes a stored one. Three separate things can be
    wrong with it and each is recorded rather than absorbed:

      * a multiplier written into the header ("10-3 rho"), applied before the unit;
      * a unit needing conversion (K, kg m-3, S/m) -- converted, with the as-written
        number kept so `check_fidelity` still tests the transcription;
      * a unit naming a different quantity (cSt), or a basis that cannot be read
        (J per gram or per mole?) -- kept, flagged, and not loaded.

    A melting, boiling, decomposition or glass-transition temperature IS a temperature:
    it has no separate measuring temperature, so a caption's "at 25 C" never lands on it.
    """
    printed_header = " ".join(table.column(column.index)) or column.header or ""
    written_unit = (column.unit_as_written or "").strip().strip("()")
    scale = units.header_scale(printed_header)
    written = parsed.value
    value, unit, status = units.to_canonical(
        written * scale if written is not None else None, written_unit, prop)
    basis, basis_status = units.basis_of(prop, f"{printed_header} {written_unit}")

    if not status and basis_status:
        status = basis_status
    if not status and prop in config.SOLUTE_PROPERTIES and not solute:
        status = "no_solute"
    if not status and parsed.status:
        status = parsed.status
    if prop in config.PHASE_TRANSITION_PROPERTIES and source != "cell_inline":
        temperature, source = None, "not_applicable"

    return {
        "Value": value,
        "Unit": unit,
        "Value_as_written": written,
        "Unit_as_written": written_unit,
        "Header_scale": scale if scale != 1.0 else None,
        "Uncertainty": parsed.uncertainty,
        "Value_low": parsed.low,
        "Value_high": parsed.high,
        "Qualifier": parsed.qualifier,
        "Temperature_C": temperature,
        "Temperature_source": source,
        "Solute": solute,
        "Basis": basis,
        "status": status or "ok",
        "_inline_refs": parsed.ref_numbers,
    }


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


def dedup_key(components, ratio, prop, value, temperature, primary_doi, solute=""):
    """Identify the same underlying datum reported by two different papers.

    Two reviews tabulating the same primary measurement is a real duplicate, and the
    primary DOI is what makes it one: same original study, same mixture, same number.
    Keyed on the resolved component SET so component order cannot split a pair.

    `solute` is part of the identity because a solubility is a property of a solute IN
    the DES: the same solvent dissolving two different drugs gives two measurements that
    would otherwise collide here whenever the numbers happened to match.
    """
    return _hash([
        _component_set(components),
        str(ratio or ""), prop,
        f"{float(value):.6g}" if value is not None else "",
        f"{float(temperature):.6g}" if temperature is not None else "",
        (primary_doi or "").split(config.SOURCE_SEP)[0].lower(),
        str(solute or "").strip().lower(),
    ])


def to_measurements(rows, paper, extras=None):
    """Flatten wide mixture rows into one MeasurementRow per reported value.

    `extras` supplies what the wide row has no column for -- the as-written value, the
    uncertainty, how the temperature was established, and the status that decides
    whether the row loads. A row with no entry (an older CSV, or a route that does not
    produce them) still works; it simply carries the defaults.
    """
    extras = extras or {}
    out = []
    for mixture in rows:
        for name in config.PROPERTY_NAMES:
            extra = dict(extras.get((mixture.Row_id, name), {}))
            value = getattr(mixture, name, None)
            if value is None and extra.get("Value_low") is None:
                continue
            inline_refs = extra.pop("_inline_refs", ()) or ()
            suffix = name.lower()
            temperature = getattr(mixture, f"Temperature_{suffix}")
            fields = {
                "Measurement_key": f"{mixture.Row_id}:{name}",
                "Row_id": mixture.Row_id,
                "Mixture_key": mixture.Mixture_key,
                "Paper_key": paper.key, "Paper_DOI": paper.doi,
                "Table_id": mixture.Table_id, "Source_row": mixture.Source_row,
                "Source_col": getattr(mixture, f"Source_col_{suffix}", None),
                "Mixture": mixture.Mixture,
                "Property": name,
                "Value": value,
                "Unit": getattr(mixture, f"Units_{suffix}"),
                "Temperature_C": temperature,
                "Source": f"{mixture.Table_id} row {mixture.Source_row}",
                "Source_ref_numbers": mixture.Source_ref_numbers,
                "Source_DOIs": mixture.Source_DOIs,
                "Source_paper_keys": mixture.Source_paper_keys,
                **_plausibility(name, value),
                # Last, so the cell's own reading of value, unit and temperature wins
                # over the wide row's copy of it rather than colliding with it.
                **extra,
            }
            # A citation written inside the value cell -- "285.15 (ref. 22)" -- attributes
            # that one number, which is more specific than the row's reference column.
            if inline_refs:
                fields["Source_ref_numbers"] = ",".join(str(n) for n in inline_refs)
            fields["Dedup_key"] = dedup_key(
                [mixture.Component_1, mixture.Component_2, mixture.Component_3],
                mixture.Ratio_molar or mixture.Ratio_raw, name, fields["Value"],
                fields["Temperature_C"], mixture.Source_DOIs, fields.get("Solute", ""))
            out.append(MeasurementRow(**fields))
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


def extract_tables(tables, profiles, paper, reference_map, dictionary=None, resolve=None,
                   paragraphs=None):
    """Every profiled property table in one paper. -> (mixtures, measurements, skipped, stats).

    `dictionary` is the paper's DES dictionary (paper_dictionary.build), which already
    holds what its definitions tables say. `stats` is per table, for the ledger.
    """
    if dictionary is not None and dictionary.des_is_subject is False:
        print("    the paper is not about DES: only rows matching its own DES are kept")

    mixtures, skipped, extras, stats = [], [], {}, {}
    for table in tables:
        profile = profiles.get(table.id)
        if profile is None or not profile.relevant:
            continue
        if "des_properties" not in profile.content_types:
            continue
        strict, skip = table_gate(table, profile, paper, dictionary, paragraphs)
        if skip:
            stats[table.id] = {"n_mixtures": 0, "n_values": 0, "n_refused": 1, "why": skip}
            print(f"    {table.label or table.id}: skipped -- {skip}")
            continue
        if profile_table.detected_layout(profile, table) == "paneled_by_mixture":
            rows, bad, more = extract_paneled_table(table, profile, paper, reference_map,
                                                    dictionary, resolve, strict)
        else:
            rows, bad, more = extract_property_table(table, profile, paper, reference_map,
                                                     dictionary, resolve, strict)
        mixtures += rows
        extras.update(more)
        skipped += [{**b, "Paper_key": paper.key} for b in bad]
        refused = [b for b in bad if str(b.get("reason", "")).startswith("not extracted")]
        stats[table.id] = {
            "n_mixtures": len(rows), "n_values": sum(1 for k in more if k[0] in
                                                      {r.Row_id for r in rows}),
            "n_refused": len(refused),
            "why": (refused[0]["reason"] if refused and not rows else ""),
        }
        print(f"    {table.label or table.id}: {len(rows)} rows"
              f"{f' [{profile.layout}]' if profile.layout != 'wide_per_mixture' else ''}"
              f"{f', {len(refused)} refused (not a DES)' if refused else ''}"
              f"{f', {len(bad) - len(refused)} unreadable' if len(bad) > len(refused) else ''}")
    measurements = to_measurements(mixtures, paper, extras)
    held = [m for m in measurements if m.status != "ok"]
    if held:
        import collections

        why = collections.Counter(m.status for m in held)
        print(f"    {len(held)} measurement(s) written but not loadable: "
              + ", ".join(f"{k} {v}" for k, v in why.most_common()))
    return mixtures, measurements, skipped, stats


EXTRACTED_TYPES = {"des_properties", "des_application", "des_definitions"}


def unhandled(tables, profiles, problems):
    """Tables that produced no data, with the reason. Nothing disappears silently."""
    out = []
    for table in tables:
        profile = profiles.get(table.id)
        if profile is not None and profile.relevant \
                and EXTRACTED_TYPES & set(profile.content_types):
            continue
        if table.graphic_only:
            # A picture of a table. Reported the way a figure is, because it holds real
            # data that a human could digitise -- and because silently dropping 21 of
            # every 2547 tables is not a behaviour anyone would choose on purpose.
            reason = "the table is an image, not markup; needs digitisation"
        elif profile is None:
            reason = "; ".join(problems.get(table.id, ["no usable profile"]))[:300]
        elif not profile.relevant:
            reason = f"profiled as not relevant: {profile.reason}"[:300]
        else:
            reason = (f"content_types {'+'.join(profile.content_types)}, "
                      f"no extractor yet")
        out.append(TableRow(table_id=table.id, label=table.label,
                            caption=table.caption[:300], n_rows=len(table.rows),
                            status=reason))
    return out


# A context column that is really the water content the DES was made with.
_WATER_FIELD = re.compile(r"\bwater\b|\bh2o\b|moisture", re.I)
# A column naming the row's subject -- the code or name of its DES.
_LABEL_FIELD = re.compile(r"abbrev|acronym|code|symbol|short|label|\bdes\b|name|sample|"
                          r"\brun\b|entry|system|^no\.?$|^#$|^s/n$", re.I)


def _printed(table, column):
    return " ".join(table.column(column.index)) or column.header or ""


def _strip_ratio_markers(cell, profile):
    """A ratio cell -> (text without footnote markers, flag). See parse_ratio."""
    raw, _parts, flag = parse_ratio(cell.text, cell.markers, profile)
    defined = {m.marker for m in (profile.footnote_markers if profile else []) if m.marker}
    text = raw
    for marker in sorted(defined, key=len, reverse=True):
        if marker and len(marker) <= 2 and text.startswith(marker):
            text = text[len(marker):].lstrip("-").strip()
            break
    return text, flag


def water_columns(profile, table):
    """Columns holding the water the DES was made with. -> list[ColumnSpec]."""
    out = [c for c in profile.columns if c.role == "water"]
    for column in profile.columns:
        if column.role in ("context", "other_quantity") and column not in out:
            name = f"{column.context_field or ''} {_printed(table, column)}"
            if _WATER_FIELD.search(name):
                out.append(column)
    return out


def row_composition(row, table, profile, dictionary=None, resolve=None):
    """One table row -> (Composition, ratio flag). The one way every route reads a DES.

    Gathers whichever of these the table has: a column naming the whole DES, one column
    per component, one or more ratio columns, a water column, and components the caption
    implies. composition.compose turns them into one composition.
    """
    by_role = {}
    for column in profile.columns:
        by_role.setdefault(column.role, []).append(column)
    # With component columns present, a "mixture" column is the row's CODE ("CEM5",
    # "[ChCl][Gl]3"), not another constituent: Fan's Table 2 lists both, and reading
    # the code as a component made "[ChCl][Gl]3" part of its own DES.
    code_cols = by_role.get("mixture", []) if by_role.get("component") else []
    name_cols = by_role.get("component", []) or by_role.get("mixture", [])
    cells = [row[c.index].text for c in name_cols if c.index < len(row)]
    code = next((row[c.index].text.strip() for c in code_cols
                 if c.index < len(row) and row[c.index].text.strip()
                 and row[c.index].text.strip() not in config.DASH), "")

    ratio_parts, flag, ratio_header = [], "", ""
    for column in by_role.get("ratio", []):
        if column.index >= len(row):
            continue
        text, this_flag = _strip_ratio_markers(row[column.index], profile)
        flag = flag or this_flag
        if text and text not in config.DASH:
            ratio_parts.append(text)
            ratio_header = ratio_header or _printed(table, column)
    # A ratio printed across two columns ("Molar Ratio" over HBA | HBD) is one ratio.
    ratio_text = ":".join(ratio_parts) if len(ratio_parts) > 1 else \
        (ratio_parts[0] if ratio_parts else "")

    water_text, water_header = "", ""
    for column in water_columns(profile, table):
        if column.index < len(row) and row[column.index].text.strip() not in config.DASH:
            water_text, water_header = row[column.index].text, _printed(table, column)
            break

    comp = compose(cells, ratio_text, ratio_header, water_text, water_header,
                   dictionary=dictionary, resolve=resolve,
                   implied=profile.implied_components or ())
    if code and not comp.code:
        comp.code = code
    return comp, flag


def _refusal(comp, dictionary, strict):
    """Why this row is not extracted as a DES. -> "" when it is.

    `strict` is "" (any DES row), or the reason rows must match a DES the paper itself
    defines (see table_gate); "il" adds that ionic-liquid rows are refused.
    """
    problem = des_problem(comp, dictionary)
    if problem:
        return problem
    known = dictionary is not None and dictionary.matches(comp)
    if strict and "il" in strict.split("+") and has_ionic_liquid(comp) and not known:
        return "an ionic liquid, in a table of ionic liquids"
    reason = "+".join(x for x in strict.split("+") if x and x != "il")
    if reason and not known:
        return f"matches none of the DES the paper defines ({reason})"
    return ""


def table_gate(table, profile, paper, dictionary, paragraphs=None):
    """How strictly to read this table's rows. -> (mode, skip_reason).

    Rows must match a DES the paper itself defines when:
      not_about_des     the paper is not about DES (its overview says so);
      no_des_mention    the table never says it is about DES -- not in its caption,
                        legend, header, or what the paper says about it;
      research_paper    a research paper that defines its DES: its tables report values
                        for the DES it made, so anything else in them is not one.
    A table whose caption names both ionic liquids and DES, without saying which row is
    which, is skipped. A table of ionic liquids refuses rows written as one ("il").
    """
    from .paper_dictionary import caption_mixes_il_and_des, mentions_des

    if profile.des_vs_il_ambiguous or caption_mixes_il_and_des(table):
        return "", "mixes ionic liquids and DES without saying which row is which"
    reasons = []
    if dictionary is not None and dictionary.des_is_subject is False:
        reasons.append("not_about_des")
    if not mentions_des(table, paragraphs):
        reasons.append("no_des_mention")
    if getattr(paper, "role", "") == "primary" and dictionary is not None \
            and dictionary.has_definitions:
        reasons.append("research_paper")
    if profile.involves_ionic_liquids:
        reasons.append("il")
    return "+".join(reasons), ""


def _is_group_row(row, table):
    """A sub-header row inside the body, not data.

    Marco-Velasco's Table 4 interleaves rows like "Use of DESs as additive and/or
    modifier" between its data: one cell spanning the whole table, broadcast by `_expand`
    into the same text in every column. Such a row labels the rows below it.
    """
    texts = [c.text.strip() for c in row]
    filled = [t for t in texts if t and t not in config.DASH]
    if not filled or len(texts) < 2:
        return False
    if len(set(filled)) == 1 and len(filled) >= max(2, len(texts) - 1):
        return xml_utils.clean_number(filled[0]) is None
    return False


def table_ledger(tables, profiles, problems, repairs, paper, table_stats=None,
                 application_stats=None, definition_counts=None):
    """One row per table in the paper -- every one of them. -> list[TableLedgerRow]."""
    table_stats = table_stats or {}
    application_stats = application_stats or {}
    definition_counts = definition_counts or {}
    out = []
    for table in tables:
        profile = profiles.get(table.id)
        stats = table_stats.get(table.id, {})
        apps = application_stats.get(table.id, {})
        n_defs = definition_counts.get(table.id, 0)
        issues = problems.get(table.id, [])
        n_values = stats.get("n_values", 0)
        n_mixtures = stats.get("n_mixtures", 0)
        n_apps = apps.get("n_applications", 0)
        refused = stats.get("n_refused", 0) + apps.get("n_refused", 0)

        if table.graphic_only:
            status, why = "not_profiled", "the table is an image, not markup"
        elif profile is None and issues and issues[0].startswith("not profiled"):
            status, why = "not_profiled", issues[0][14:]
        elif profile is None:
            status, why = "failed", "the model's labels failed validation"
        elif not profile.relevant:
            status, why = "not_relevant", "the profile says it holds no DES data"
        elif n_values or n_mixtures or n_apps or n_defs:
            status, why = "extracted", ""
        elif refused:
            status = "not_des"
            why = stats.get("why") or apps.get("why") or "every row failed the DES test"
        elif profile.des_vs_il_ambiguous:
            status, why = "not_des", "mixes ionic liquids and DES without saying which"
        else:
            kinds = set(profile.content_types)
            if not kinds & {"des_properties", "des_application", "des_definitions"}:
                status, why = "not_relevant", (f"content is "
                                               f"{'+'.join(profile.content_types)}")
            else:
                status, why = "no_rows", stats.get("why") or "no row yielded a value"

        columns = ""
        if profile is not None:
            columns = " | ".join(
                f"{c.index}:{c.role}" + (f"={c.property}" if c.property else "")
                + (f"({c.context_field})" if c.role in ("context", "other_quantity")
                   and c.context_field else "")
                for c in profile.columns)
        out.append(TableLedgerRow(
            Paper_key=paper.key, table_id=table.id, label=table.label,
            caption=table.caption[:300], n_rows=len(table.rows),
            n_columns=table.n_columns, status=status, why=why[:300],
            content_types="+".join(profile.content_types) if profile else "",
            involves_ionic_liquids=profile.involves_ionic_liquids if profile else None,
            n_values=n_values, n_mixtures=n_mixtures, n_definitions=n_defs,
            n_applications=n_apps, n_rows_refused=refused, columns=columns[:600],
            demoted="; ".join(repairs.get(table.id, []))[:600],
            problems="; ".join(issues)[:600] if status in ("failed",) else "",
            model_reason=(profile.reason if profile else "")[:300]))
    return out
