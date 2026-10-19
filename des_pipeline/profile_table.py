"""
Work out what a table contains, so the layout does not have to be hard-coded.

Until now the pipeline knew that Table 2 of one specific paper had melting point in
column 3 and that a superscript 'b' meant 20 C, because both were written into
config.py by hand. That does not survive a second paper, let alone a thousand.

So the model reads a *card* -- caption, header rows, legend, and five sample rows --
and returns a column map. It is shown at most a few dozen cells and it never reports
a value: deterministic code then extracts all 1500 rows using the map. The same
division of labour as everywhere else in this pipeline, for the same reason.

What makes it safe is validate(): the model must echo each column's header back
verbatim, and we compare that against the header we hold. A map shifted by one
column is caught immediately, because the echoed text would not match. Anything that
fails validation extracts nothing and is reported instead -- missing data is visible,
a silently mislabelled column is not.

Profiles are cached like every other model call, and can be overridden by hand in
data/papers/<slug>/table_profiles.json.
"""
import hashlib
import json
import re

from pydantic import ValidationError

from . import config, units
from .extract_text_llm import cached_call_llm
from .schema import TableProfile

PROMPT = """You are labelling the COLUMNS of one table from a chemistry paper that \
involves deep eutectic solvents (DES). You never read data out of the table: you say what \
each column means, and code reads the values.

{card}

Return, for EVERY column index 0..{n_columns} exactly once, in order:

  index            the column number.
  header           the header text at that index, copied character-for-character from
                   above. Do not tidy, translate or expand it. This is checked.
  role             mixture | component | ratio | water | reference | property |
                   other_quantity | condition | context | ignore
                     mixture         each cell names a WHOLE DES: a formula with its
                                     ratio, or a code the paper uses for one.
                     component       each cell names ONE constituent of the DES.
                     ratio           the mixing ratio of the DES's constituents.
                     water           the water content the DES was MADE with.
                     reference       bracketed numbers pointing at the bibliography.
                     property        a measured property from this list: {properties}.
                     other_quantity  a numeric measured quantity that is NOT in that
                                     list. Name it in context_field. Never force a
                                     quantity into the nearest property name.
                     condition       the temperature or pressure AT WHICH the other
                                     columns were measured.
                     context         descriptive information about the row.
                     ignore          anything else (row numbers, blank columns).
  property         when role is property, which one from the list. null otherwise.
  unit_as_written  the unit exactly as the header or sub-header prints it. null if none.
  component_role   HBA, HBD or either, when role is component.
  context_field    for context and other_quantity: a short snake_case name.
  multi_valued     true when one cell holds several values.

Also return:
  relevant               true only when the ROWS are deep eutectic solvents (or the
                         table defines, measures or applies DES). false for tables of
                         instrument settings, analytical figures of merit, statistics,
                         reagents, biological results, or anything without DES rows.
  involves_ionic_liquids true when the rows are ionic liquids rather than DES, or
                         include them.
  des_vs_il_ambiguous    true when the table mixes ionic liquids and DES and does not say
                         which row is which.
  content_types          a LIST of every kind of data this table holds:
                           des_properties    measured physical properties of DES
                           des_application   a DES used FOR something (substrate,
                                             technique, yield, product)
                           des_definitions   says what each DES is made of
                           component_properties  properties of single pure compounds
                           other             anything else
                         A table with a composition, a measured density AND the yield it
                         achieved is all three of the first three.
                         These are always "other", even though every cell is a number:
                           fitted equation or model parameters (A, B, r2, Ea, eta0);
                           design-of-experiment matrices, ANOVA, regression coefficients;
                           group-contribution or atomic-fragment tables;
                           simulation or computed results (MD, DFT, COSMO, H-bond counts);
                           calibration, LOD, recovery and other analytical-method data.
  layout                 wide_per_mixture | paneled_by_mixture.
                         DEFAULT TO wide_per_mixture -- nearly every table is one row
                         per DES with its properties in fixed columns.
                         Choose paneled_by_mixture ONLY if BOTH are true:
                           (a) the column headers repeat a group, e.g. reading
                               "T, value, T, value, T, value" rather than naming
                               six different things, AND
                           (b) a SPANNING ROW (marked as such below) names the DES for
                               each group, because no column holds the DES name.
  implied_components     components in every mixture that no column lists, when the
                         CAPTION says so ("X based DES" beside a table listing only the
                         other constituent). Leave empty unless the caption states it.
  implied_from           the caption words that justify implied_components.
  reason                 one sentence.
  header_row_count       how many leading rows are header, not data.
                         When the card says NO HEADER MARKUP, say how many of the leading
                         rows shown are really the header, and label the columns from
                         them. 0 if the table truly starts with data.
  footnote_markers       one entry per marker the legend defines, with its meaning and,
                         for a temperature marker, the temperature in Celsius. Only
                         markers the LEGEND above actually defines. Empty if no legend.
  missing_value_tokens   the strings this table prints IN PLACE OF a value: "-", "n/a",
                         "nd", or a token its legend defines as "not measured". Not
                         footnote markers, which annotate a value that is present.
  default_temperature_C  the temperature unmarked values were measured at, if the
                         caption or legend says. null otherwise.

Rules:
  - NEVER copy, transcribe, average or convert a data value. Indices and labels only.
  - Emit exactly one record per column index. Do not skip empty-looking columns.
  - Symbols: Tm melting point, rho density, eta viscosity, kappa or sigma (in S/cm)
    conductivity, gamma surface tension, nD refractive index, Tb boiling point, lambda
    thermal conductivity, Tg glass transition, Td or Tdec decomposition temperature.
  - The UNIT decides between similar properties: mS/cm is Conductivity, W/(m K) is
    Thermal_conductivity. A unit that is not the property's own (a pressure, an energy,
    a percentage) means the column is other_quantity.
  - A component is a constituent OF THE SOLVENT ITSELF. A drug, solute, substrate,
    reagent, feedstock, product, analyte or biomass that the DES acts on is "context",
    never "component". If WHAT THE PAPER SAYS ABOUT THIS TABLE is shown above, use it.
  - A column counting WATER EQUIVALENTS ("H2O" holding 1, 2, 3) is role "context" with
    context_field "water_equivalents". A column giving water as a percentage or mass
    fraction of the DES is role "water".
  - A melting point or boiling point is ALSO a temperature, but it IS the measured
    quantity: role "property". A temperature column with no other property column
    depending on it is a property, not a condition.
"""


def table_context(table, paragraphs, max_chars=1200):
    """What the running text says about this table. -> str.

    The paper usually explains its own table, and that explanation answers exactly the
    questions the card cannot: which columns are the SOLVENT and which are what the
    solvent acts on. Canela-Xandri's Table 2 is captioned only "PSTA based DES used in
    esterification reactions", but the prose says "ammonium-based hydrogen bond
    acceptors (HBAs) are typically used as primary components" -- which is the
    difference between reading Alcohol and Acid as DES components and reading them as
    the reagents they are.

    Paragraphs only. A section can contain the table itself, so section text would feed
    the table's own contents back into the prompt meant to interpret it.
    """
    label = (table.label or "").strip()
    if not label or not paragraphs:
        return ""
    pattern = re.compile(rf"\b{re.escape(label)}\b", re.I)
    hits = [p for p in paragraphs if p and pattern.search(p)]
    out, used = [], 0
    for h in hits[:2]:
        text = re.sub(r"\s+", " ", h).strip()
        # Trim to the neighbourhood of the mention rather than pasting a whole page.
        m = pattern.search(text)
        start = max(0, m.start() - 500)
        text = ("..." if start else "") + text[start:m.end() + 500]
        if used + len(text) > max_chars:
            text = text[:max_chars - used]
        out.append(text)
        used += len(text)
        if used >= max_chars:
            break
    return "\n".join(out)


def table_card(table, paper=None, n_samples=5, paragraphs=None):
    """The small, readable view of a table that the model is asked to label."""
    lines = []
    if paper is not None:
        lines.append(f"PAPER  {getattr(paper, 'key', '')} — {getattr(paper, 'title', '')[:90]}")
    lines.append(f"TABLE  {table.label or table.id}  "
                 f"{table.n_columns} columns, {len(table.rows)} data rows")
    if table.caption:
        lines.append(f"CAPTION  {table.caption}")
    if table.footnotes:
        lines.append(f"LEGEND  {table.footnotes}")
    context = table_context(table, paragraphs or [])
    if context:
        lines.append(f"WHAT THE PAPER SAYS ABOUT THIS TABLE\n  {context}")

    if table.header_missing:
        # No <thead>, so the header is sitting in the data rows. Say so, and show the
        # candidates: without this the model is asked to echo a header it was never
        # shown, every echo fails validation, and the table extracts nothing.
        lines.append("\nNO HEADER MARKUP: this table declares no header row. The first "
                     "rows below may be the header.\n  Say how many in header_row_count "
                     "and label the columns from them.")
        for n, row in enumerate(table.rows[:3]):
            lines.append(f"\nLEADING ROW {n} (header or data?)")
            lines += [f"  col {i}: {cell.text}" for i, cell in enumerate(row)]
    for n, row in enumerate(table.header, 1):
        lines.append(f"\nHEADER ROW {n}")
        lines += [f"  col {i}: {cell.text}" for i, cell in enumerate(row)]

    # Spread the samples out: the first rows of a table are often atypical.
    total = len(table.rows)
    picks = sorted({0, 1, 2, total // 2, total - 1} & set(range(total)))[:n_samples]
    for i in picks:
        row = table.rows[i]
        # A spanning row has been broadcast across the columns it covers, so it reads
        # as the same text repeated. Saying so matters: without it the model sees a
        # chemical name in column 0 of the first sample row and labels that column
        # "component", when column 0 actually holds temperatures and the name belongs
        # to the whole panel.
        note = ""
        if _is_spanning(row):
            note = "   <- SPANNING ROW: one cell covers several columns; it labels the " \
                   "group below, it is not data"
        lines.append(f"\nSAMPLE ROW {i}{note}")
        lines += [f"  col {j}: {cell.text}" for j, cell in enumerate(row)]
    return "\n".join(lines)


def _is_spanning(row):
    """Was every cell in this row broadcast across a panel?

    `_expand` turns a spanning cell into the same text repeated over the columns it
    covered, so a band row is a sequence of equal-width contiguous runs: three panels
    of two columns read [A, A, B, B, C, C].

    Both halves of that matter. Merely "some text repeats" also describes an ordinary
    data row that prints "DT" in two property columns, and treating those as band rows
    silently dropped 34 of Sadeghi's 36 DT cells from the skipped-cell audit. Requiring
    contiguity and a uniform run width separates the two without needing the profile.
    """
    from . import xml_utils

    filled = [c.text.strip() for c in row
              if c.text.strip() and c.text.strip() not in config.DASH]
    if len(filled) < 4 or any(xml_utils.clean_number(t) is not None for t in filled):
        return False

    runs, previous = [], object()
    for text in filled:
        if text == previous:
            runs[-1] += 1
        else:
            runs.append(1)
            previous = text
    return len(runs) >= 2 and len(set(runs)) == 1 and runs[0] >= 2


def card_hash(card):
    return hashlib.sha256(card.encode("utf-8")).hexdigest()[:16]


def _numeric_fraction(table, index, profile, sample=250):
    """How much of a column actually parses as a number. -> (fraction, n_checked).

    The header echo cannot catch a column labelled with the wrong *meaning*: asked to
    profile a table of eutectic types, the model called the "Formula" column --
    holding "Cat+X- zMClx" -- a melting point, and echoed the header correctly while
    doing it. Numbers are something we can check ourselves.

    It reads cells with the EXTRACTOR's own parser. It used to use `clean_number`, which
    rejects "1.0101 ± 0.02", so every column written with uncertainties was demoted to
    context after the model had labelled it correctly -- Alotaibi's and Anđelović's whole
    property tables were lost that way. A check that disagrees with the reader it guards
    rejects exactly the cells the reader would have handled.
    """
    from .extract_table import parse_measurement_cell

    missing = set(profile.missing_value_tokens) | set(config.DASH)
    markers = {m.marker for m in profile.footnote_markers}
    checked = numeric = 0
    for row in table.rows[:sample]:
        if index >= len(row):
            continue
        text = row[index].text.strip()
        if not text or text in missing:
            continue
        checked += 1
        # A footnote marker is part of the notation, not the number -- leading or trailing.
        stripped = text
        for marker in markers:
            if marker and stripped.startswith(marker):
                stripped = stripped[len(marker):]
                break
            if marker and stripped.endswith(marker) and len(marker) <= 2:
                stripped = stripped[:-len(marker)]
                break
        parsed = parse_measurement_cell(stripped, row[index].markers, profile)
        if parsed.value is not None or parsed.low is not None:
            numeric += 1
    return (numeric / checked if checked else 0.0), checked


def validate(profile, table):
    """-> list of problems. Empty means the profile is safe to extract with.

    The checks are all against data we already hold rather than anything the model
    asserts: the echoed header must match the printed header (catching a map that has
    drifted by a column), a property column must hold numbers and carry a unit that
    belongs to that property (`repair` has already demoted the ones that do not), and
    markers and implied components must be traceable to the legend and caption.
    """
    problems = []
    indices = [c.index for c in profile.columns]
    if sorted(indices) != list(range(table.n_columns)):
        problems.append(f"expected one entry per column 0..{table.n_columns - 1}, "
                        f"got {sorted(indices)}")
        return problems                      # nothing else is meaningful after this

    def norm(s):
        return re.sub(r"[\s()]+", "", str(s or "")).lower()

    for column in profile.columns:
        printed = " ".join(table.column(column.index))
        if norm(column.header) and norm(column.header) not in norm(printed) \
                and norm(printed):
            problems.append(f"col {column.index}: model echoed {column.header!r} but "
                            f"the header reads {printed!r}")
        if column.role != "property":
            continue
        fraction, checked = _numeric_fraction(table, column.index, profile)
        if checked >= 5 and fraction < 0.5:
            problems.append(
                f"col {column.index} ({column.header!r}) is labelled "
                f"{column.property} but only {fraction:.0%} of {checked} filled cells "
                f"are numbers -- it does not hold measurements")
        problems += _unit_conflict(column, table)

    for marker in profile.footnote_markers:
        if marker.marker and marker.marker not in table.footnotes:
            problems.append(f"marker {marker.marker!r} is not in the table's legend")

    problems += _marker_temperature_problems(profile, table)
    problems += _layout_problems(profile, table)
    problems += _implied_problems(profile, table)
    return problems


def _unit_conflict(column, table):
    """Does the printed unit contradict the property the model chose? -> list.

    Conductivity (mS/cm) and Thermal_conductivity (W/(m K)) are one word apart and both
    numeric, so neither the header echo nor the numeric check separates them. The unit
    does, and the table prints it.
    """
    pattern = config.UNIT_PATTERN.get(column.property)
    if not pattern:
        return []
    printed = " ".join([column.unit_as_written or ""] + table.column(column.index)).lower()
    if not printed.strip():
        return []
    if re.search(pattern, printed):
        return []
    # Only complain when some OTHER property's unit clearly matches instead.
    for other, other_pattern in config.UNIT_PATTERN.items():
        if other != column.property and re.search(other_pattern, printed):
            return [f"col {column.index}: labelled {column.property} but the unit reads "
                    f"{printed.strip()!r}, which is {other}"]
    return []


def panel_period(roles):
    """The width of the repeating column group, or 0 when the roles do not repeat.

    A paneled table is periodic by definition: (condition, property) three times over
    is period 2. Requiring a *clean* period is what catches a map that is nearly right
    -- labelling Fan's first column "component" because the band row shows a DES name
    there gives (component, property, condition, property, condition, property), which
    has no period at all, so the profile is rejected instead of silently reading
    temperatures as though they were a chemical name.
    """
    n = len(roles)
    for period in range(1, n // 2 + 1):
        if n % period:
            continue
        if all(roles[i] == roles[i % period] for i in range(n)):
            return period
    return 0


def detected_layout(profile, table):
    """What the table's own shape says its layout is. -> 'wide_per_mixture' | 'paneled_by_mixture'.

    The model's `layout` answer is treated as a hint, not an authority: asked twice
    about Fan's viscosity table it said "paneled" once and "wide" once, and the wide
    reading extracts nothing at all. The evidence is objective and already in hand --
    a repeating group of column roles that contains a property, at least one row that
    names mixtures across a span, and no column holding the mixture name -- so code
    decides and the model only has to get the column meanings right.
    """
    roles = [c.role for c in profile.columns]
    period = panel_period(roles)
    if period < 2 or period == len(roles):
        return "wide_per_mixture"
    if "property" not in roles[:period]:
        return "wide_per_mixture"
    if "component" in roles:
        return "wide_per_mixture"
    if not any(_is_band_row(row) for row in table.rows):
        return "wide_per_mixture"
    return "paneled_by_mixture"


def _marker_temperature_problems(profile, table):
    """A marker's temperature must actually appear in the legend that defines it.

    The legend states them outright -- "At a40 C, b20 C, c60 C, ..." -- so this is
    checkable rather than a matter of trust, and it needs to be: a re-profiling run
    returned every one of those eight markers as 4.0e-16 instead of 40.0. The columns
    were right, the layout was right, and 407 measurements were silently stamped 0 C
    instead of 20-60 C.

    Fidelity cannot catch this. It re-reads each value through the SAME profile, so
    extractor and re-reader agree perfectly on a temperature that is wrong. This is the
    only place the profile itself is checked against the paper.
    """
    legend = table.footnotes or ""
    if not legend:
        return []
    problems = []
    for marker in profile.footnote_markers:
        if marker.meaning != "temperature" or marker.temperature_C is None:
            continue
        value = float(marker.temperature_C)
        # Accept the number however the legend might reasonably write it.
        spellings = {f"{value:g}", f"{value:.0f}", f"{value:.1f}"}
        if any(re.search(rf"(?<!\d){re.escape(t)}(?!\d)", legend) for t in spellings):
            continue
        problems.append(
            f"marker {marker.marker!r} claims {value:g} C but the legend does not state "
            f"that number: {legend[:90]!r}")
    return problems


def _layout_problems(profile, table):
    """Reconcile the model's layout claim with what the table looks like.

    `detected_layout` is what actually gets used, so a disagreement is not fatal -- it
    is reported, and the profile is corrected to match the evidence.
    """
    detected = detected_layout(profile, table)
    if profile.layout != detected:
        print(f"      layout: model said {profile.layout}, the table's shape says "
              f"{detected} -- using {detected}")
        profile.layout = detected
    return []


def _is_band_row(row):
    """A row that names the mixtures for the panels below it, rather than holding data.

    After `_expand` a spanning cell has been broadcast across the columns it covers, so
    a band row shows the same text repeated and no numbers anywhere.
    """
    return _is_spanning(row)


def _caption_mentions(name, haystack):
    """Does the caption name this component? -> (bool, how).

    Exact containment is the normal case. The fuzzy arm exists because papers misspell
    their own subject: this one captions two of its seven tables "PSTA based DES" and
    the other five "PTSA", so demanding the exact string would reject a correct
    inference on the strength of the authors' typo. A transposition is accepted, and
    reported as fuzzy so the review queue can show a human what was matched.
    """
    import difflib

    key = re.sub(r"[^a-z0-9]", "", name.lower())
    if not key:
        return False, ""
    # Lower-case the haystack BEFORE stripping: [^a-z0-9] eats capitals, so an
    # un-lowered caption loses the very word being looked for.
    if key in re.sub(r"[^a-z0-9]", "", haystack.lower()):
        return True, "exact"
    for token in re.findall(r"[A-Za-z0-9]+", haystack):
        token = token.lower()
        if len(token) != len(key):
            continue
        # sorted letters match => a transposition, which is what PSTA/PTSA is
        if sorted(token) == sorted(key) or \
                difflib.SequenceMatcher(None, token, key).ratio() >= 0.8:
            return True, f"fuzzy: caption writes {token!r}"
    return False, ""


def _implied_problems(profile, table):
    """An implied component must actually be traceable to the caption."""
    if not profile.implied_components:
        return []
    problems = []
    haystack = f"{table.caption} {table.footnotes}"

    # The model tends to quote the caption as `Caption: '...'`; the wrapper is not
    # part of the claim, so strip it before checking the quote is real.
    quoted = re.sub(r"^\s*caption\s*[:\-]\s*", "", profile.implied_from.strip(),
                    flags=re.I).strip(" '\"")
    if not quoted:
        problems.append("implied_components given with no implied_from to justify them")
    elif re.sub(r"\W+", "", quoted.lower()) not in re.sub(r"\W+", "", haystack.lower()):
        problems.append(f"implied_from {quoted!r} is not in the caption")

    for name in profile.implied_components:
        found, how = _caption_mentions(name, haystack)
        if not found:
            problems.append(f"implied component {name!r} is not named in the caption")
        elif how.startswith("fuzzy"):
            print(f"      implied component {name!r} matched loosely ({how}) "
                  f"-- confirm it in table_profiles.json")
    return problems


def _fix_ratio_columns(profile, table):
    """A ratio column holds numbers. One full of chemical names is a component column.

    "HBA : HBD" over cells reading "ChCl : PTSA" looks like a ratio if you go by the
    colon, and the model duly called it one -- which left that table with no component
    column at all, so it silently produced nothing instead of seventeen rows. A ratio
    is "1:2"; if the cells are words, the column names the components.
    """
    from . import xml_utils

    notes = []
    for column in profile.columns:
        if column.role != "ratio":
            continue
        cells = [c.text.strip() for c in
                 (row[column.index] for row in table.rows if column.index < len(row))
                 if c.text.strip()]
        if len(cells) < 3:
            continue
        # A ratio cell is digits and separators; anything with a run of letters is not.
        wordy = sum(1 for c in cells if len(re.findall(r"[A-Za-z]{3,}", c)) >= 1)
        if wordy / len(cells) < 0.5:
            continue
        notes.append(f"col {column.index} ({column.header!r}) labelled ratio but "
                     f"{wordy}/{len(cells)} cells name chemicals -- read as component")
        column.role = "component"
        column.component_role = column.component_role or "either"
    return notes


def repair(profile, table):
    """Demote labels that cannot be right, rather than failing the whole table. In place.

    Every demotion is evidence-based and reported (the notes go into the table ledger):

      * a property column whose cells are not numbers -> context;
      * a property column with no property name, or whose printed unit measures a
        different quantity (a pressure labelled Density, a percentage labelled pH) ->
        other_quantity. These used to FAIL the table, which lost every good column in
        it along with the one bad one;
      * a Polarity_ET30 column that is really a Kamlet-Taft alpha/beta/pi* -> other_quantity;
      * a footnote marker the legend does not define -> dropped (the model sometimes
        returns markers from its instructions rather than from this table);
      * a context column naming each row's DES, in a table with no component column ->
        mixture, so the rows have an identity.
    """
    notes = _fix_numeric_name_columns(profile, table)
    notes += _fix_condition_columns(profile, table)
    notes += _fix_ratio_columns(profile, table)
    for column in profile.columns:
        if column.role != "property":
            continue
        name = column.context_field or _snake(" ".join(table.column(column.index))
                                              or column.header)
        if column.property not in config.PROPERTY_NAMES:
            notes.append(f"col {column.index} ({column.header!r}) labelled property with "
                         f"no property name -- read as other_quantity")
            column.role, column.context_field, column.property = "other_quantity", name, None
            continue
        fraction, checked = _numeric_fraction(table, column.index, profile)
        if checked >= 5 and fraction < 0.5:
            notes.append(f"col {column.index} ({column.header!r}) labelled "
                         f"{column.property} but only {fraction:.0%} of {checked} filled "
                         f"cells are numbers -- read as context")
            column.role, column.context_field, column.property = "context", name, None
            continue
        printed = " ".join(table.column(column.index))
        conflict = units.unit_conflict(column.unit_as_written or _unit_in(printed),
                                       column.property)
        if conflict:
            notes.append(f"col {column.index} ({column.header!r}) labelled "
                         f"{column.property}, but {conflict} -- read as other_quantity")
            column.role, column.context_field, column.property = "other_quantity", name, None
            continue
        if column.property == "Polarity_ET30":
            basis, _status = units.basis_of(column.property,
                                            f"{printed} {column.unit_as_written or ''}")
            if basis and basis != "ET30":
                notes.append(f"col {column.index} ({column.header!r}) is Kamlet-Taft "
                             f"{basis}, not E_T(30) -- read as other_quantity")
                column.role, column.property = "other_quantity", None
                column.context_field = f"kamlet_taft_{basis}"

    # A temperature the legend never states is not trusted -- but it costs the marker,
    # not the table. Failing the whole table over it lost Azad's and Dinh's tables
    # because the model read "a: standard uncertainty of T is 0.1 K" as a temperature.
    # The value keeps no marker temperature (it reads as unstated), which is the honest
    # answer. A hand-pinned profile never reaches this function.
    for message in _marker_temperature_problems(profile, table):
        marker = re.match(r"marker '([^']*)'", message)
        for m in profile.footnote_markers:
            if marker and m.marker == marker.group(1) and m.meaning == "temperature":
                m.meaning, m.temperature_C = "other", None
                notes.append(f"{message[:120]} -- marker kept, temperature dropped")
    defined = [m for m in profile.footnote_markers
               if not m.marker or m.marker in (table.footnotes or "")]
    dropped = [m.marker for m in profile.footnote_markers if m not in defined]
    if dropped:
        notes.append(f"dropped footnote marker(s) {', '.join(map(repr, dropped))}: "
                     f"the table's legend does not define them")
        profile.footnote_markers = defined
    notes += _fix_mixture_columns(profile, table)
    # An implied component the caption does not state is dropped, not fatal: Azad's
    # model-parameter tables were failed outright because the model "implied" water and
    # the dissolved drug. The rest of the profile is still checkable on its own.
    if profile.implied_components and _implied_problems(profile, table):
        notes.append(f"implied component(s) {', '.join(profile.implied_components)} not "
                     f"stated by the caption -- dropped")
        profile.implied_components, profile.implied_from = [], ""
    return profile, notes


# A temperature column headed with one of these is the measured quantity, not the
# condition it was measured at. Deterministic because prompt wording could not fix it:
# told that a temperature column is a "condition", the model relabelled Sadeghi's
# "Tm (˚C)" column and 332 melting points stopped being extracted.
_PROPERTY_SYMBOLS = (
    (r"\bt\s*m\b|\btm\b|melting|freezing|eutectic point", "Melting_point"),
    (r"\bt\s*b\b|\btb\b|boiling", "Boiling_point"),
)


def _fix_condition_columns(profile, table):
    """Promote a `condition` column that is really the property being measured."""
    notes = []
    for column in profile.columns:
        if column.role != "condition":
            continue
        printed = " ".join([column.header or ""] + table.column(column.index)).lower()
        for pattern, name in _PROPERTY_SYMBOLS:
            if re.search(pattern, printed):
                notes.append(f"col {column.index} ({column.header!r}) labelled condition "
                             f"but the header says {name} -- read as a property")
                column.role = "property"
                column.property = name
                break
    return notes


def _snake(text):
    """'Lignin fractionation yield/removal rate' -> 'lignin_fractionation_yield'."""
    words = re.findall(r"[A-Za-z0-9]+", str(text or ""))[:4]
    return "_".join(w.lower() for w in words) or "detail"


# ---------- what the vocabulary is missing ----------
#
# A header that names a quantity rather than a label: it carries a unit in brackets, or
# reads like a measured thing. Deliberately loose -- this list is read by a human, and a
# false positive costs one row in a review CSV while a false negative costs a property
# nobody ever learns the corpus contains.
_QUANTITY_HEADER = re.compile(
    r"\(.*(?:%|/|·|\bg\b|\bm\b|\bs\b|\bk\b|\bj\b|\bv\b|\bpa\b|\bmol\b).*\)"
    r"|temperature|content|capacit|potential|strength|energy|point|index|ratio"
    r"|conductiv|tension|densit|viscosit|solubilit|polarit|\bph\b|weight|mass",
    re.I)


def property_candidates(table, profile, paper):
    """Numeric columns of DES data that the vocabulary does not cover. -> rows.

    The answer to "what properties do the other papers have?": the corpus says, with
    the header it printed, the unit it printed and three of its own values. Promoting
    one is then an edit to config.PROPERTIES and nothing else.

    Only `other_quantity` columns (and numeric context columns the model did not name)
    in tables whose rows are DES -- a candidate from a corrosion-inhibitor table or an
    analytical-method table is not a DES property, and listing it trains nobody. A name
    that is already a property is not a candidate.
    """
    out = []
    vocabulary = {re.sub(r"[^a-z]", "", n.lower()) for n in config.PROPERTY_NAMES}
    for column in profile.columns:
        if column.role not in ("other_quantity", "context"):
            continue
        printed = " ".join(table.column(column.index)) or column.header or ""
        if column.role == "context" and not _QUANTITY_HEADER.search(printed):
            continue
        name = column.context_field or _snake(printed)
        if re.sub(r"[^a-z]", "", name.lower()) in vocabulary:
            continue
        values = [row[column.index].text.strip() for row in table.rows
                  if column.index < len(row) and row[column.index].text.strip()
                  and row[column.index].text.strip() not in config.DASH]
        if len(values) < 2:
            continue
        numeric = sum(1 for v in values if v[:1].isdigit() or v[:1] in "-−.<>~")
        if numeric / len(values) < 0.6:
            continue
        out.append({
            "Paper_key": getattr(paper, "key", ""),
            "slug": getattr(paper, "slug", ""),
            "Table_id": table.id,
            "table_label": table.label,
            "column": column.index,
            "name": name,
            "header": printed[:120],
            "unit_as_written": (column.unit_as_written or "")[:40],
            "role_given": column.role,
            "n_values": len(values),
            "examples": " | ".join(values[:3])[:120],
            "caption": table.caption[:160],
        })
    return out


def discover_properties(tables, profiles, paper, des_tables=None):
    """Property candidates across one paper's tables. -> rows.

    `des_tables`: the ids of tables that actually produced DES rows. Candidates come
    only from those, when given.
    """
    out = []
    for table in tables:
        profile = profiles.get(table.id)
        if profile is None or not profile.relevant:
            continue
        if des_tables is not None and table.id not in des_tables:
            continue
        if "des_properties" not in profile.content_types:
            continue
        out += property_candidates(table, profile, paper)
    return out


# ---------- hand overrides ----------
def _overrides_path(paper):
    return config.PAPERS_DIR / getattr(paper, "slug", "unknown") / "table_profiles.json"


# Set by run_pipeline --ignore-human-profiles, so a model comparison profiles every
# table with the model under test, including the ones a human has pinned.
IGNORE_HUMAN_PROFILES = False


def load_overrides(paper):
    """table id -> saved profile record. Human pins in the shared tree always count.

    A tagged run (runs/<tag>/) writes its own table_profiles.json, but a profile a human
    pinned with "source": "human" lives in data/papers/<slug>/ and is a person's
    verdict, not a model's, so every run honours it unless told not to.
    """
    out = {}
    path = _overrides_path(paper)
    if path.exists():
        out = json.loads(path.read_text())
    shared = config.HUMAN_PROFILES_DIR / getattr(paper, "slug", "unknown") / \
        "table_profiles.json"
    if not IGNORE_HUMAN_PROFILES and shared.exists() and shared != path:
        for table_id, record in json.loads(shared.read_text()).items():
            if record.get("source") == "human":
                out[table_id] = record
    if IGNORE_HUMAN_PROFILES:
        out = {k: v for k, v in out.items() if v.get("source") != "human"}
    return out


def save_profiles(profiles, paper, cards, repairs=None, problems=None):
    """Record every profile -- failed ones too -- so it can be read, corrected, pinned.

    A failed profile used to be left out of this file, so a table that failed
    validation appeared in tables_unhandled.csv and nowhere else, and the two files
    each looked incomplete. Every table the model was asked about is now here, with
    `status` saying whether it was used.
    """
    path = _overrides_path(paper)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_overrides(paper)
    out = {}
    for table_id, profile in profiles.items():
        was = existing.get(table_id, {})
        out[table_id] = {
            # "human" means: use this verbatim and never call the model again.
            "source": was.get("source", "llm"),
            "status": "used" if not (problems or {}).get(table_id) else "failed",
            "card_sha256": cards.get(table_id, ""),
            "repairs": (repairs or {}).get(table_id, []),
            "problems": (problems or {}).get(table_id, []),
            "profile": profile.model_dump(),
        }
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return path


# A chemical name looks like this: a run of letters long enough not to be a symbol, or
# one of the shapes a DES table uses to name its components.
_NAMEISH = re.compile(r"[A-Za-z]{4,}|\[[A-Za-z]{2,}\]|\b[A-Z]{2,5}\b")


def worth_profiling(table):
    """Should this table cost a model call? -> (bool, reason).

    Every table is one LLM call, so the full 13,108-paper corpus is ~33,000 of them.
    A table with no numbers AND no chemical names holds neither a measurement nor a
    composition, whatever its caption says, and the call can only return "other".

    Deliberately generous: it takes evidence to REJECT, and either signal alone is
    enough to keep. A table of instrument settings survives this, and should -- it is
    the model's job to call that "other", not this function's.
    """
    if table.graphic_only:
        return False, "the table is an image, not markup; it needs digitisation"
    if not table.rows:
        return False, "no data rows"
    cells = [c.text.strip() for row in table.rows[:60] for c in row
             if c.text.strip() and c.text.strip() not in config.DASH]
    if not cells:
        return False, "every cell is empty or a dash"
    from . import xml_utils

    if any(xml_utils.clean_number(c) is not None or c[:1].isdigit() for c in cells):
        return True, ""
    header = " ".join(" ".join(table.column(i)) for i in range(table.n_columns))
    if any(_NAMEISH.search(c) for c in cells) or _NAMEISH.search(header):
        return True, ""
    return False, "no numeric cell and no chemical name anywhere in the table"


def profile_table(table, paper=None, backend=None, refresh=False, paragraphs=None):
    """-> (TableProfile | None, problems, was_cached, digest, repairs).

    Mutates `table` when its header had to be recovered from the data rows -- the same
    Table object is what `extract_table` reads, so the promotion has to happen here,
    between the model naming the header rows and `validate` checking the echo.
    """
    card = table_card(table, paper, paragraphs=paragraphs)
    digest = card_hash(card)

    override = load_overrides(paper).get(table.id) if paper is not None else None
    if override and override.get("source") == "human":
        if override.get("card_sha256") not in ("", digest):
            print(f"    {table.label or table.id}: hand-written profile is stale "
                  f"(the table has changed since it was written) -- ignoring it")
        else:
            profile = TableProfile(**override["profile"])
            table.promote_header(profile.header_row_count)
            return profile, [], True, digest, []

    usable, why = worth_profiling(table)
    if not usable:
        return None, [f"not profiled: {why}"], True, digest, []

    prompt = PROMPT.format(card=card, n_columns=table.n_columns - 1,
                           properties=", ".join(config.PROPERTY_NAMES))
    section_id = re.sub(r"[^A-Za-z0-9_.-]", "-",
                        f"tbl-{getattr(paper, 'slug', 'x')}-{table.id or 't'}")[:48]
    raw, was_cached = cached_call_llm(prompt, TableProfile.model_json_schema(),
                                      section_id=section_id, backend=backend,
                                      refresh=refresh)
    try:
        profile = TableProfile.model_validate_json(raw)
    except ValidationError as exc:
        return (None, [f"output did not validate: {exc.errors()[0]['msg']}"], was_cached,
                digest, [])

    moved = table.promote_header(profile.header_row_count)
    if moved:
        print(f"      no header markup: promoted {moved} leading row(s) into the header")

    profile, repairs = repair(profile, table)
    for note in repairs:
        print(f"      {note}")
    return profile, validate(profile, table), was_cached, digest, repairs


def profile_tables(tables, paper=None, backend=None, refresh=False, paragraphs=None):
    """-> ({id: TableProfile}, {id: problems}, {id: repair notes}, {id: failed profile})."""
    profiles, problems, cards, repairs, failed = {}, {}, {}, {}, {}
    hits = skipped = 0
    for table in tables:
        # The digest comes back from profile_table rather than being recomputed here.
        # Rebuilding the card cost a second full render of every table, and after header
        # promotion it no longer even described the same table.
        profile, issues, was_cached, digest, notes = profile_table(
            table, paper, backend, refresh, paragraphs)
        hits += was_cached
        cards[table.id] = digest
        if notes:
            repairs[table.id] = notes
        name = table.label or table.id
        if profile is None:
            problems[table.id] = issues
            if issues and issues[0].startswith("not profiled:"):
                skipped += 1
                print(f"    {name:<12} skipped  {issues[0][14:]}")
            else:
                print(f"    {name:<12} FAILED  {issues[0] if issues else ''}")
            continue
        if issues:
            problems[table.id] = issues
            failed[table.id] = profile
            print(f"    {name:<12} {len(issues)} problem(s) -- not extracting")
            for issue in issues[:3]:
                print(f"                 {issue}")
            continue
        profiles[table.id] = profile
        kinds = ", ".join(sorted({c.property for c in profile.columns if c.property}))
        print(f"    {name:<12} {'+'.join(profile.content_types):<28} "
              f"{table.n_columns} cols, {len(profile.footnote_markers)} markers"
              f"{'  cached' if was_cached else ''}")
        if kinds:
            print(f"                 properties: {kinds}")
    if paper is not None and (profiles or failed):
        save_profiles({**failed, **profiles}, paper, cards, repairs, problems)
    print(f"  profiles: {hits} cached, {len(tables) - hits} fresh"
          f"{f', {skipped} skipped without a call' if skipped else ''}")
    return profiles, problems, repairs, failed


def _unit_in(printed):
    """The bracketed unit in a printed header: 'Pc (Bar)' -> 'Bar'. -> str."""
    match = re.search(r"\(([^()]{1,20})\)\s*$|/\s*([^/]{1,20})$", str(printed or ""))
    if not match:
        return ""
    return (match.group(1) or match.group(2) or "").strip()


_MIXTURE_FIELD = re.compile(r"\bdes\b|mixture|solvent|electrolyte|sample|system|"
                            r"composition|name|nades", re.I)


_TEMPERATURE_HEADER = re.compile(r"^\s*\(?\s*(k|°c|℃|c|°f)\s*\)?\s*$|\bt\s*/|^t\b|"
                                 r"temperature", re.I)


def _fix_numeric_name_columns(profile, table):
    """A column of NUMBERS cannot name a DES or a component.

    Shown Fan's paneled table (a spanning row naming each DES, then T | lambda pairs),
    the model labelled the T/K columns "mixture" -- and 60 thermal conductivities lost
    their measuring temperature. If a name column is mostly numbers, it is a condition
    when its header is a temperature, and context otherwise.
    """
    from .extract_table import parse_measurement_cell

    notes = []
    for column in profile.columns:
        if column.role not in ("mixture", "component"):
            continue
        cells = [r[column.index].text.strip() for r in table.rows
                 if column.index < len(r) and r[column.index].text.strip()
                 and r[column.index].text.strip() not in config.DASH]
        if len(cells) < 3:
            continue
        numeric = sum(1 for c in cells if parse_measurement_cell(c).value is not None)
        if numeric / len(cells) < 0.6:
            continue
        printed = f"{' '.join(table.column(column.index))} {column.unit_as_written or ''}"
        if _TEMPERATURE_HEADER.search(printed.strip()) or \
                units.dimension_of(column.unit_as_written or "") == "temperature":
            column.role = "condition"
        else:
            column.role = "context"
            column.context_field = column.context_field or _snake(printed)
        notes.append(f"col {column.index} ({column.header!r}) labelled a name but holds "
                     f"numbers -- read as {column.role}")
    return notes


def _fix_mixture_columns(profile, table):
    """Give the rows an identity when the model labelled the DES-name column 'context'.

    Alotaibi's Table 2 is "DES | Density | Viscosity | ...", with the DES written as
    a code (CE, CEM1, ...); Dinh's Table 3 is "electrolytes | d | sigma | eta" with
    "DES 1:4". The model called those first columns context, so every row had a value
    and no mixture, and the table extracted nothing. A context column whose header names
    the DES and whose cells are words is the mixture column.
    """
    roles = {c.role for c in profile.columns}
    if roles & {"mixture", "component"} or not profile.relevant \
            or not set(profile.content_types) & {"des_properties", "des_application"}:
        return []
    for column in profile.columns:
        if column.role != "context":
            continue
        printed = f"{' '.join(table.column(column.index))} {column.context_field or ''}"
        if not _MIXTURE_FIELD.search(printed):
            continue
        cells = [r[column.index].text.strip() for r in table.rows
                 if column.index < len(r) and r[column.index].text.strip()]
        if not cells:
            continue
        wordy = sum(1 for c in cells if re.search(r"[A-Za-z]{2}", c))
        if wordy / len(cells) < 0.6:
            continue
        column.role = "mixture"
        return [f"col {column.index} ({column.header!r}) names each row's DES -- read "
                f"as mixture"]
    return []
