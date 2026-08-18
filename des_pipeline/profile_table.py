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
  record_type            des_properties | des_application | component_properties | other
  reason                 one sentence.
  header_row_count       how many leading rows are header, not data.
  footnote_markers       one entry per marker the legend defines, with its meaning and,
                         for a temperature marker, the temperature in Celsius.
                         Empty list when the table has no legend.
  missing_value_tokens   the strings this table uses for "not reported".
  default_temperature_C  the temperature unmarked values were measured at, if the
                         caption or legend says. null otherwise.

Rules:
  - NEVER copy, transcribe, average or convert a data value. Indices and labels only.
  - Emit exactly one record per column index. Do not skip empty-looking columns.
  - A column of bracketed numbers pointing at the bibliography is role "reference".
  - Symbols: Tm is melting point, rho density, eta viscosity, kappa conductivity,
    gamma surface tension, nD refractive index, Tb boiling point.
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
        lines.append(f"\nSAMPLE ROW {i}")
        lines += [f"  col {j}: {cell.text}" for j, cell in enumerate(table.rows[i])]
    return "\n".join(lines)


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

    for marker in profile.footnote_markers:
        if marker.marker and marker.marker not in table.footnotes:
            problems.append(f"marker {marker.marker!r} is not in the table's legend")
    return problems


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
