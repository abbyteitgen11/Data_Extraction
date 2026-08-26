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
  record_type            des_properties | des_application | des_definitions |
                         component_properties | other.
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
  - "condition" is for a column giving the temperature or pressure AT WHICH the other
    columns were measured -- a "T/K" column beside a viscosity column.
    A melting point or boiling point is ALSO a temperature, but it IS the measured
    quantity, so it is role "property" with property Melting_point / Boiling_point.
    If a temperature column has no other property column depending on it, it is a
    property, not a condition.
"""


def table_card(table, paper=None, n_samples=5):
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


def repair(profile, table):
    """Demote labels that cannot be right, rather than failing the whole table.

    A property column has to hold numbers. In a table of DES *applications* a
    mislabelled one is also irrelevant -- "Removed compounds" called a refractive index
    is wrong, but nothing reads properties out of an application table anyway. Failing
    the table over it would throw away nine good application rows to avoid a column no
    extractor touches, so it becomes `context` instead and says so.
    """
    notes = _fix_condition_columns(profile, table)
    if profile.record_type == "des_properties":
        return profile, notes
    for column in profile.columns:
        if column.role != "property":
            continue
        # In an application table a "property" column is almost always a yield, a
        # water equivalent or a removal rate -- a real quantity, but not one in the
        # config vocabulary, so the model picks the nearest name and is wrong. This
        # paper's "Yield model substrate (%)" came back as Refractive_index. Nothing
        # reads properties out of an application table, and keeping the number as
        # context loses none of it.
        fraction, checked = _numeric_fraction(table, column.index, profile)
        why = ("holds no numbers" if checked >= 5 and fraction < 0.5
               else "is not a physical property of the DES")
        if profile.record_type != "des_application" and fraction >= 0.5:
            continue
        notes.append(f"col {column.index} ({column.header!r}) labelled "
                     f"{column.property} {why} -- read as context")
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


def profile_table(table, paper=None, backend=None, refresh=False):
    """-> (TableProfile | None, problems, was_cached)."""
    card = table_card(table, paper)
    digest = card_hash(card)

    override = load_overrides(paper).get(table.id) if paper is not None else None
    if override and override.get("source") == "human":
        if override.get("card_sha256") not in ("", digest):
            print(f"    {table.label or table.id}: hand-written profile is stale "
                  f"(the table has changed since it was written) -- ignoring it")
        else:
            return TableProfile(**override["profile"]), [], True

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
        return None, [f"output did not validate: {exc.errors()[0]['msg']}"], was_cached

    profile, repairs = repair(profile, table)
    for note in repairs:
        print(f"      {note}")
    return profile, validate(profile, table), was_cached


def profile_tables(tables, paper=None, backend=None, refresh=False):
    """-> ({table id: TableProfile}, {table id: problems})."""
    profiles, problems, cards, hits = {}, {}, {}, 0
    for table in tables:
        profile, issues, was_cached = profile_table(table, paper, backend, refresh)
        hits += was_cached
        cards[table.id] = card_hash(table_card(table, paper))
        name = table.label or table.id
        if profile is None:
            problems[table.id] = issues
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
        print(f"    {name:<12} {profile.record_type:<18} {table.n_columns} cols, "
              f"{len(profile.footnote_markers)} markers"
              f"{'  cached' if was_cached else ''}")
        if kinds:
            print(f"                 properties: {kinds}")
    if paper is not None and profiles:
        save_profiles(profiles, paper, cards)
    print(f"  profiles: {hits} cached, {len(tables) - hits} fresh")
    return profiles, problems
