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

from . import config
from .extract_text_llm import cached_call_llm
from .schema import TableProfile

PROMPT = """You are labelling the COLUMNS of one table from a chemistry paper about
deep eutectic solvents (DES). You never read data out of the table: you say what each
column means, and code reads the values.

{card}

Return, for EVERY column index 0..{n_columns} exactly once, in order:

  index            the column number.
  header           the header text at that index, copied character-for-character from
                   above. Do not tidy, translate or expand it. This is checked.
  role             component | ratio | reference | property | condition | context | ignore
  property         when role is property, which of: {properties}.
                   null otherwise. A property column that is not in that list gets
                   role "ignore" -- do not force it into the nearest one.
  unit_as_written  the unit exactly as the header or sub-header prints it. null if none.
  component_role   HBA, HBD or either, when role is component.
  context_field    when role is context, a short snake_case name for what it records.
  multi_valued     true when one cell holds several values.

Also return:
  relevant               false when the table has no DES data at all.
  content_types          a LIST of every kind of data this table holds. One table often
                         holds several, and naming only the best one throws the rest
                         away:
                           des_properties    measured physical properties of DES
                           des_application   a DES used FOR something (substrate,
                                             technique, yield, product)
                           des_definitions   names/abbreviates DES, no measurements
                           component_properties  properties of single pure compounds
                           other             anything else
                         A table with a composition, a measured density AND the yield it
                         achieved is all three of the first three. Return every one that
                         applies, not the closest single label.
                         Fitted equation parameters (columns like A, B, C, r2, E, eta0)
                         are "other" -- they are coefficients, not measurements, even
                         though every cell is a number.
  layout                 wide_per_mixture | paneled_by_mixture.
                         DEFAULT TO wide_per_mixture -- nearly every table is one row
                         per DES with its properties in fixed columns.
                         Choose paneled_by_mixture ONLY if BOTH are true:
                           (a) the column headers repeat a group, e.g. reading
                               "T, lambda, T, lambda, T, lambda" rather than naming
                               six different things, AND
                           (b) a SPANNING ROW (marked as such below) names the DES for
                               each group, because no column holds the DES name.
                         If any column holds the DES name or abbreviation, the layout is
                         wide_per_mixture.
  implied_components     components in every mixture that no column lists, when the
                         CAPTION says so: "PTSA based DES" beside an HBA-only table means
                         PTSA. Leave empty unless the caption states it -- do not infer
                         from the paper's title or subject.
  implied_from           the caption words that justify implied_components.
  reason                 one sentence.
  header_row_count       how many leading rows are header, not data.
                         When the card says NO HEADER MARKUP, this is what recovers the
                         table: say how many of the leading rows shown are really the
                         header, and label the columns from them. 0 if the table truly
                         starts with data.
  footnote_markers       one entry per marker the legend defines, with its meaning and,
                         for a temperature marker, the temperature in Celsius.
                         Empty list when the table has no legend.
  missing_value_tokens   the strings this table PRINTS IN PLACE OF a value: "-", "n/a",
                         "nd", or a legend-defined token like "DT" meaning the value was
                         reported at several temperatures. These belong here, NOT in
                         footnote_markers -- a footnote marker annotates a value that is
                         present, a missing-value token replaces one that is not.
  default_temperature_C  the temperature unmarked values were measured at, if the
                         caption or legend says. null otherwise.

Rules:
  - NEVER copy, transcribe, average or convert a data value. Indices and labels only.
  - Emit exactly one record per column index. Do not skip empty-looking columns.
  - A column of bracketed numbers pointing at the bibliography is role "reference".
  - Symbols: Tm is melting point, rho density, eta viscosity, kappa conductivity,
    gamma surface tension, nD refractive index, Tb boiling point, lambda thermal
    conductivity.
  - Conductivity and Thermal_conductivity are DIFFERENT properties. The unit decides:
    mS/cm or S/m is electrical Conductivity; W/(m K) is Thermal_conductivity.
  - A "component" is a constituent OF THE SOLVENT ITSELF. In a des_application table,
    a substrate, reagent, feedstock, product, biomass source or removed compound is what
    the DES ACTS ON, not what it is made of: role "context", not "component".
    An esterification table listing HBA | Alcohol | Acid describes a two-component DES
    (the HBA plus whatever the caption implies) reacting an alcohol with an acid --
    the alcohol and the acid are role "context", named alcohol and acid.
    If WHAT THE PAPER SAYS ABOUT THIS TABLE is shown above, use it to decide: it usually
    states which columns are the solvent's own components.
  - A column counting WATER EQUIVALENTS ("H2O" holding 1, 2, 3) is not a name and not a
    plain context field: give it role "context" with context_field "water_equivalents",
    so code can read the count as part of the composition.
  - "condition" is for a column giving the temperature or pressure AT WHICH the other
    columns were measured -- a "T/K" column beside a viscosity column.
    A melting point or boiling point is ALSO a temperature, but it IS the measured
    quantity, so it is role "property" with property Melting_point / Boiling_point.
    If a temperature column has no other property column depending on it, it is a
    property, not a condition.
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
    """
    from . import xml_utils

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
        # A leading footnote marker is part of the notation, not the number.
        stripped = text
        for marker in markers:
            if marker and stripped.startswith(marker):
                stripped = stripped[len(marker):]
                break
        if xml_utils.clean_number(stripped) is not None:
            numeric += 1
    return (numeric / checked if checked else 0.0), checked


def validate(profile, table):
    """-> list of problems. Empty means the profile is safe to extract with.

    Two checks, both against data we already hold rather than anything the model
    asserts: the echoed header must match the printed header (catching a map that has
    drifted by a column), and a column labelled as a property must actually contain
    numbers (catching a plausible-looking but wrong label).
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
        if norm(column.header) and norm(column.header) not in norm(printed):
            problems.append(f"col {column.index}: model echoed {column.header!r} but "
                            f"the header reads {printed!r}")
        if column.role != "property":
            continue
        if column.property not in config.PROPERTY_NAMES:
            problems.append(f"col {column.index}: role=property but property="
                            f"{column.property!r}")
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
    """Demote labels that cannot be right, rather than failing the whole table.

    A property column has to hold numbers, and that is the ONLY reason one is demoted
    here now. This used to demote every property column in any table not labelled
    `des_properties`, on the reasoning that nothing reads properties out of an
    application table -- which was true while `record_type` forced a table to be one
    thing. It no longer is: a table can declare both, and Solcan's and Alotaibi's do, so
    demoting on the label alone would throw away exactly the measurements this change
    exists to keep.

    The numeric check stands on its own evidence and is kept. It is what caught
    "Yield model substrate (%)" being labelled Refractive_index -- a real quantity, but
    not one in the vocabulary, so the model picked the nearest name.
    """
    notes = _fix_condition_columns(profile, table)
    notes += _fix_ratio_columns(profile, table)
    for column in profile.columns:
        if column.role != "property":
            continue
        fraction, checked = _numeric_fraction(table, column.index, profile)
        if checked < 5 or fraction >= 0.5:
            continue
        notes.append(f"col {column.index} ({column.header!r}) labelled "
                     f"{column.property} but only {fraction:.0%} of {checked} filled "
                     f"cells are numbers -- read as context")
        column.role = "context"
        column.context_field = column.context_field or _snake(column.header)
        column.property = None
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
    """Numeric columns that look like properties but are not in the vocabulary. -> rows.

    The answer to "I don't know what properties the other papers have": the corpus says,
    with the header it printed, the unit it printed and three of its own values, and
    promoting one is then an edit to config.PROPERTIES and nothing else.
    """
    from . import xml_utils

    out = []
    for column in profile.columns:
        if column.role in ("property", "component", "ratio", "reference"):
            continue
        printed = " ".join(table.column(column.index)) or column.header or ""
        if not _QUANTITY_HEADER.search(printed):
            continue
        values = [row[column.index].text.strip() for row in table.rows
                  if column.index < len(row) and row[column.index].text.strip()
                  and row[column.index].text.strip() not in config.DASH]
        if len(values) < 4:
            continue
        numeric = sum(1 for v in values if xml_utils.clean_number(v) is not None
                      or v[:1].isdigit())
        if numeric / len(values) < 0.6:
            continue
        out.append({
            "Paper_key": getattr(paper, "key", ""),
            "slug": getattr(paper, "slug", ""),
            "Table_id": table.id,
            "table_label": table.label,
            "column": column.index,
            "header": printed[:120],
            "unit_as_written": (column.unit_as_written or "")[:40],
            "role_given": column.role,
            "context_field": column.context_field or "",
            "n_values": len(values),
            "examples": " | ".join(values[:3])[:120],
            "caption": table.caption[:160],
        })
    return out


def discover_properties(tables, profiles, paper):
    """Collect property candidates across one paper's tables. -> rows."""
    out = []
    for table in tables:
        profile = profiles.get(table.id)
        if profile is not None and profile.relevant:
            out += property_candidates(table, profile, paper)
    return out


# ---------- hand overrides ----------
def _overrides_path(paper):
    return config.PAPERS_DIR / getattr(paper, "slug", "unknown") / "table_profiles.json"


def load_overrides(paper):
    path = _overrides_path(paper)
    return json.loads(path.read_text()) if path.exists() else {}


def save_profiles(profiles, paper, cards):
    """Record every profile so it can be read, corrected, and pinned by hand."""
    path = _overrides_path(paper)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_overrides(paper)
    out = {}
    for table_id, profile in profiles.items():
        was = existing.get(table_id, {})
        out[table_id] = {
            # "human" means: use this verbatim and never call the model again.
            "source": was.get("source", "llm"),
            "card_sha256": cards.get(table_id, ""),
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
    """-> (TableProfile | None, problems, was_cached).

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
            return profile, [], True, digest

    usable, why = worth_profiling(table)
    if not usable:
        return None, [f"not profiled: {why}"], True, digest

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
        return None, [f"output did not validate: {exc.errors()[0]['msg']}"], was_cached, digest

    moved = table.promote_header(profile.header_row_count)
    if moved:
        print(f"      no header markup: promoted {moved} leading row(s) into the header")

    profile, repairs = repair(profile, table)
    for note in repairs:
        print(f"      {note}")
    return profile, validate(profile, table), was_cached, digest


def profile_tables(tables, paper=None, backend=None, refresh=False, paragraphs=None):
    """-> ({table id: TableProfile}, {table id: problems})."""
    profiles, problems, cards, hits, skipped = {}, {}, {}, 0, 0
    for table in tables:
        # The digest comes back from profile_table rather than being recomputed here.
        # Rebuilding the card cost a second full render of every table, and after header
        # promotion it no longer even described the same table.
        profile, issues, was_cached, digest = profile_table(
            table, paper, backend, refresh, paragraphs)
        hits += was_cached
        cards[table.id] = digest
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
    if paper is not None and profiles:
        save_profiles(profiles, paper, cards)
    print(f"  profiles: {hits} cached, {len(tables) - hits} fresh"
          f"{f', {skipped} skipped without a call' if skipped else ''}")
    return profiles, problems
