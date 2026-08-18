"""
Read the PubChem property text properly, keeping every measurement and its source.

enrich_components takes the first parseable number per property and drops the rest.
That is a reasonable single best guess, but it loses real data: 208 of the 256
components with text report the same property more than once, from different sources.
Ammonium thiocyanate has a melting point of 320 F from CAMEO Chemicals and 149.6 C
from elsewhere; only the first survives as a scalar.

This module is additive. The scalars stay exactly as they were; this produces the
long view, one row per (component, property, source, value).

The design that makes it checkable: **the model returns a line number, never text and
never a converted unit.** We already hold the lines, so

    attribution   comes from PubChem's own Reference map, not from the model
    conversion    is done here in Python by _to_celsius
    verification  is exact -- the number must occur character-for-character in the
                  line WE looked up, not in a quote the model wrote for itself

which leaves the model doing only the thing it is actually good at: reading a messy
sentence and saying which number is a measurement, what conditions it was under, and
whether it belongs to a different substance entirely.

    python run_pipeline.py --steps components            # runs this too
    python run_pipeline.py --steps components --limit 10 # smoke test first
"""
import re

from pydantic import ValidationError

from . import config
from .enrich_components import _alnum_key, _load_cache, _to_celsius
from .extract_text_llm import cached_call_llm, canonical_unit, verified
from .schema import (COMPONENT_PROPERTY_STATUS_ORDER, ComponentPropertyExtraction,
                     ComponentPropertyRow)

PROMPT = """You read PubChem property lines for ONE substance and turn each into records.

The substance is "{title}" (CID {cid}).

Each numbered line below is a measurement statement copied verbatim from PubChem.
Return one record per NUMBER a line states. Emit EVERY field, null where not stated.

  line          The number of the line the value came from. Copy it exactly.
  property      Melting_point, Boiling_point or Density -- the line's own heading.
  value         The number AS WRITTEN, in the unit written on the line. Do NOT
                convert units. Do NOT average a range -- emit the low and the high
                as two records. NEVER invent, estimate or calculate a number.
  unit          As written: "C", "F", "K", "g/cm3", "g/mL", or null.
  temperature_C The temperature the value was measured AT ("1.3057 @ 25 C" -> 25).
                CAUTION: for a melting or boiling point the number IS the
                temperature -- put it in value and leave this null.
  pressure      As written, e.g. "760 mm Hg", or null.
  qualifier     null, or: approximate, greater_than, less_than, decomposes, sublimes.
  applies_to    null if the value is for {title} itself. Otherwise the OTHER
                CHEMICAL SUBSTANCE, grade, form or isomer it belongs to, as
                written: "PEG 400", "dl-Form", "solution".
                This field is ONLY for a different chemical. A citation in
                brackets is NOT a different substance: "320 F (USCG, 1999)" is a
                value for {title} itself, so applies_to must be null.

Rules:
  - Every number you emit must appear character-for-character on its line.
  - A line with no number ("Solid decomposes") still gets a record: value null,
    qualifier set.
  - One line can carry several values. Emit a record for each.
  - Citations, agency names, years and CAS numbers are NOT data and NOT substances.

Examples -- illustration only, these lines are NOT below:
  7. Boiling Point: 197.00 to 198.00 C. @ 760.00 mm Hg
  -> two records, value 197.0 and value 198.0, both unit "C",
     pressure "760.00 mm Hg", temperature_C null, applies_to null.
  8. Melting Point: PEG 400: 4-8 C; PEG 3000: 50-56 C
  -> four records, each with applies_to "PEG 400" or "PEG 3000".

LINES:
{lines}
"""


def _verified_in(value, raw):
    """Does `value` occur in this source line?

    Ranges are written "1.1-1.15", so a bare number scan reads the separator as a
    minus sign and 1.15 looks absent. Spacing out a hyphen that sits between two
    digits fixes that without touching the prose route's own check.
    """
    if value is None:
        return False
    return verified(value, re.sub(r"(?<=\d)-(?=\d)", " ", raw))


def _numbered(entries):
    """The lines as the model sees them, 1-indexed to match `line`."""
    return "\n".join(f"{i}. {e['heading']}: {e['string']}"
                     for i, e in enumerate(entries, 1))


def _status(row):
    """What keeps this record out of the graph. First match wins."""
    if row.value is None:
        return "qualitative"            # the line hedges rather than measuring
    if not row.verified:
        return "unverified"             # the number is not on the line
    if row.applies_to:
        return "different_substance"    # it belongs to PEG 400, not to this component
    if row.property == "Density" and row.unit != config.PROPERTY_UNITS["Density"]:
        return "unhandled_unit"         # lb/gal, "relative density (water = 1)"
    return "ok"


# PubChem's Density heading often gives a bare number with only a condition attached
# -- "1.3230 @ 20 C/4 C", "1.34 at 68 F" -- and the model puts that condition in the
# unit field because the line states no unit. Those values are g/cm3 by the heading's
# convention. Matching only a leading "at"/"@" keeps this away from real units:
# "g/cu m" is a millionth of g/cm3 and must stay unhandled.
_CONDITION_ONLY = re.compile(r"^(at\b|@)", re.I)


def _to_row(draft, entry, name, cid, seen):
    """One validated draft + the line it points at -> a ComponentPropertyRow."""
    unit_written = (draft.unit or "").strip()
    if draft.property == "Density":
        value = draft.value
        implied = not unit_written or _CONDITION_ONLY.match(unit_written)
        unit = canonical_unit("g/cm3" if implied else unit_written, draft.property)
    else:
        value = _to_celsius(draft.value, unit_written.strip("° ")) if draft.value is not None else None
        value = round(value, 3) if value is not None else None
        unit = config.PROPERTY_UNITS[draft.property]

    raw = entry["string"]
    source_record = entry.get("source_record") or ""
    record_title = entry.get("record_title") or ""

    row = ComponentPropertyRow(
        key="",                                     # filled below, once it is unique
        name=name,
        cid=cid,
        property=draft.property,
        value=value,
        unit=unit,
        value_as_written=draft.value,
        unit_as_written=unit_written,
        temperature_C=draft.temperature_C,
        pressure=(draft.pressure or "").strip(),
        qualifier=(draft.qualifier or "").strip(),
        applies_to=(draft.applies_to or "").strip(),
        data_source=entry.get("data_source") or "",
        source_record=source_record,
        # Advisory only, never a filter: ~18% of PubChem strings carry a reference
        # whose name is a synonym rather than the record title.
        source_record_matches=(not source_record or not record_title
                               or _alnum_key(source_record) == _alnum_key(record_title)),
        source_db=entry.get("source_db") or "pubchem",
        extractor="llm",
        raw_string=raw,
        # Checked against the line we fetched, not against anything the model wrote.
        verified=_verified_in(draft.value, raw),
    )
    row.status = _status(row)

    stem = f"{name}:{row.property}:{draft.value if draft.value is not None else 'none'}:{row.data_source}"
    key, n = stem, 1
    while key in seen:                              # same value, same source, twice
        n += 1
        key = f"{stem}#{n}"
    seen.add(key)
    row.key = key
    return row


def extract_properties(name, entries, cid=None, backend=None, refresh=False):
    """-> (list[ComponentPropertyRow], was_cached) for one component."""
    pubchem = [e for e in entries if e.get("source_db") != "nist"]
    rows, seen = [], set()

    # NIST values are already structured -- they came from a regex, not prose -- so
    # they become records directly without costing an LLM call.
    for entry in entries:
        if entry.get("source_db") != "nist":
            continue
        rows.append(ComponentPropertyRow(
            key=f"{name}:{entry['property']}:{entry['value_as_written']}:NIST WebBook",
            name=name, cid=cid, property=entry["property"],
            value=round(_to_celsius(entry["value_as_written"], "K"), 3),
            unit=config.PROPERTY_UNITS[entry["property"]],
            value_as_written=entry["value_as_written"],
            unit_as_written=entry["unit_as_written"],
            data_source="NIST WebBook", source_record=entry.get("source_record") or "",
            source_db="nist", extractor="regex", raw_string=entry["string"],
            verified=True, status="ok",
        ))
        seen.add(rows[-1].key)

    if not pubchem:
        return rows, True

    prompt = PROMPT.format(title=name, cid=cid or "?", lines=_numbered(pubchem))
    raw, was_cached = cached_call_llm(
        prompt, ComponentPropertyExtraction.model_json_schema(),
        section_id=f"comp-{cid or _alnum_key(name)[:16]}", backend=backend, refresh=refresh)

    try:
        drafts = ComponentPropertyExtraction.model_validate_json(raw).values
    except ValidationError as exc:
        print(f"    {name}: output did not validate ({exc.errors()[0]['msg']})")
        return rows, was_cached

    for draft in drafts:
        if not 1 <= draft.line <= len(pubchem):     # a line number we never sent
            continue
        rows.append(_to_row(draft, pubchem[draft.line - 1], name, cid, seen))
    return rows, was_cached


def run(limit=None, refresh_llm=False, backend=None):
    """Read every cached component's PubChem text. -> list[ComponentPropertyRow]."""
    cache = _load_cache()
    todo = [(n, r) for n, r in cache.items() if r.get("property_strings")]
    if limit is not None:
        todo = todo[:limit]
    if not todo:
        print("  no component text to read -- run --steps components first")
        return []

    print(f"  reading text for {len(todo)} component(s) via "
          f"{backend or config.LLM_BACKEND}")
    rows, hits = [], 0
    for i, (name, record) in enumerate(todo, 1):
        found, was_cached = extract_properties(
            name, record["property_strings"], cid=record.get("cid"),
            backend=backend, refresh=refresh_llm)
        hits += was_cached
        rows += found
        if not was_cached or i % 50 == 0:
            print(f"    {i}/{len(todo)}  {name[:34]:<36} {len(found):>3} records"
                  f"{'  cached' if was_cached else ''}")

    from collections import Counter
    counts = Counter(r.status for r in rows)
    print(f"\n  cache: {hits} hit, {len(todo) - hits} fresh")
    print(f"  {len(rows)} property records")
    for status in COMPONENT_PROPERTY_STATUS_ORDER:
        if counts.get(status):
            print(f"    {status:<22}{counts[status]:>5}{'  -> graph' if status == 'ok' else ''}")
    return rows
