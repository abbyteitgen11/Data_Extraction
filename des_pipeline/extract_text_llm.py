"""
The prose route: ask an LLM for measurements stated in the running text, with a
pydantic schema constraining the output.

This is the least reliable route in the pipeline, so its output goes to a review
CSV rather than straight into the graph.

Two things differ from the earlier extract_LLM.py, both aimed at why it "didn't
work very well":

  1. It sent the whole flattened paper truncated to 12000 characters -- which is
     mostly Table 2 and the bibliography, i.e. noise. Here each prose section is
     sent on its own, with its title. The six property sections are 3.9k-7.6k
     characters, so nothing is truncated at all. That is the whole point of
     routing by structure first.

  2. Its verification check was circular: `str(value) in source_text`, where
     source_text is written *by the model*. A hallucinated number validated
     against its own hallucinated quote. Here the value is checked against the
     real section text from the XML.

The backend is pluggable: ollama locally by default, Anthropic if you set
DES_LLM_BACKEND=anthropic and ANTHROPIC_API_KEY.
"""
import hashlib
import json
import math
import re
import time
from collections import Counter

from pydantic import ValidationError

from . import config, xml_utils
from .enrich_components import (_norm_key, component_index, resolve_components,
                                resolve_phrase)
from .extract_references import sources
from .schema import STATUS_ORDER, LLMExtraction, LLMMeasurement

PROMPT = """You extract deep eutectic solvent (DES) property measurements from one \
section of a chemistry review paper.

The section is titled "{title}", so the measurements it describes are most likely \
{property_hint}.

Return one record per (DES, property) pair the text reports. Emit EVERY field of \
every record, using null where the text does not state it.

  components     The chemicals in the DES, HBA first, then the HBD(s).
                 EXPAND EVERY ABBREVIATION to the full chemical name:
                   "ChCl" -> "Choline chloride"
                   "EG"   -> "Ethylene glycol"
                   "TBAB" -> "Tetrabutylammonium bromide"
                 If this paper spells the abbreviation out, use the paper's wording.
                 If you do not know what an abbreviation stands for, copy it
                 verbatim. NEVER guess a chemical from the letters.
  molar_ratio    The mixing ratio exactly as written: "1:2", "1:1.5", "1:1:1".
                 It usually sits in brackets right after the components:
                 "ChAc:EG(1:2, 23 C)" -> "1:2".
  property       One of: Melting_point, Density, Viscosity, Conductivity,
                 Surface_tension, Refractive_index.
  value          The number stated in the text, or null.
                 USE null when the text only ranks or compares DESs and gives no
                 number ("A > B > C", "higher than", "increases with").
                 NEVER invent, estimate, average or calculate a number.
  unit           The unit exactly as written: "C", "g/cm3", "mPa.s", "mS/cm",
                 "mN/m". Refractive index is dimensionless: use null.
  temperature_C  The temperature the measurement was made at, if stated.
                 CAUTION: for a melting point the number IS the temperature -- put
                 it in `value` and leave temperature_C null, unless a separate
                 measuring temperature is also given.
  ref_numbers    The bracketed citation THIS number is attributed to, as written:
                 "113", "73,80", "26-28". Take the citation nearest the number,
                 not every citation in the sentence. null if there is none.
  source_text    The clause containing the number, copied verbatim. Keep it SHORT --
                 at most about 150 characters. Quote just enough to show where the
                 number came from; do not copy a whole paragraph.

Rules:
  - Only record what the text states. Never carry a value from one DES to another.
  - Citation numbers in square brackets, years and page numbers are NOT data.
  - If the section states no DES measurements, return an empty list.

Example -- illustration only, this sentence is NOT in the text below:
  "TBAB:Gly (1:3, viscosity 47.3 mPa.s at 30 C) [999]"
  -> {{"components": ["Tetrabutylammonium bromide", "Glycolic acid"],
       "molar_ratio": "1:3", "property": "Viscosity", "value": 47.3,
       "unit": "mPa.s", "temperature_C": 30.0, "ref_numbers": "999",
       "source_text": "TBAB:Gly (1:3, viscosity 47.3 mPa.s at 30 C) [999]"}}

TEXT:
{text}
"""


# ---------- backends ----------
def _ollama_options():
    """The options that change the answer. Shared with cache_key so they cannot drift."""
    return {
        "temperature": 0,                          # deterministic
        "num_ctx": config.OLLAMA_NUM_CTX,
        "num_predict": config.OLLAMA_NUM_PREDICT,
        "repeat_penalty": config.OLLAMA_REPEAT_PENALTY,
    }


def call_llm(prompt, schema_json, backend=None):
    """Send one prompt, return the raw JSON string. Backend chosen by config/env."""
    backend = backend or config.LLM_BACKEND
    if backend == "ollama":
        from ollama import chat

        response = chat(
            model=config.OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format=schema_json,                    # constrains the output to our schema
            think=config.OLLAMA_THINK,             # off: we discard reasoning, so skip it
            keep_alive=config.OLLAMA_KEEP_ALIVE,   # survive the slow steps either side
            options=_ollama_options(),
        )
        if response.done_reason == "length":
            print(f"    warning: hit num_predict ({config.OLLAMA_NUM_PREDICT}); "
                  f"the JSON is probably truncated")
        return response.message.content

    if backend == "anthropic":
        import anthropic

        client = anthropic.Anthropic()
        message = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=8000,
            tools=[{
                "name": "record_measurements",
                "description": "Record the DES measurements found in the text.",
                "input_schema": schema_json,
            }],
            tool_choice={"type": "tool", "name": "record_measurements"},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in message.content:
            if block.type == "tool_use":
                return json.dumps(block.input)
        return "{}"

    raise ValueError(f"unknown LLM backend {backend!r} (expected 'ollama' or 'anthropic')")


# ---------- response cache ----------
def cache_key(prompt, schema_json, backend=None):
    """sha256 over everything that could change the model's answer.

    Deliberately does NOT cover component_aliases.json: alias resolution happens
    after the call, so editing the alias file and re-running must be free. That is
    what makes "add a definition, then re-assess every earlier paper" practical.
    """
    backend = backend or config.LLM_BACKEND
    payload = {
        "backend": backend,
        "prompt": prompt,
        "schema": schema_json,
        "model": config.OLLAMA_MODEL if backend == "ollama" else config.ANTHROPIC_MODEL,
        "options": _ollama_options() if backend == "ollama" else {"max_tokens": 8000},
        "think": config.OLLAMA_THINK if backend == "ollama" else None,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def cached_call_llm(prompt, schema_json, section_id="", backend=None, refresh=False):
    """call_llm with an on-disk cache. -> (raw JSON string, was_cached).

    Note the key includes the prompt, so editing PROMPT discards every entry and the
    next run pays the full 30-45 minutes. run() prints the hit/miss tally so that is
    impossible to do by accident.
    """
    key = cache_key(prompt, schema_json, backend)
    path = config.LLM_CACHE_DIR / f"{section_id or 'section'}-{key[:12]}.json"
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8"), True

    raw = call_llm(prompt, schema_json, backend=backend)
    config.LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(raw, encoding="utf-8")
    return raw, False


# ---------- verification ----------
def _normalize_number(s):
    """'1.20' -> '1.2', '750.0' -> '750', '750' -> '750'.

    Trailing zeros are only meaningful after a decimal point -- stripping them
    unconditionally would turn 750 into 75.
    """
    s = s.strip()
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def _numbers_in(text):
    # The paper writes negatives with a unicode minus ("−11.06"), which a plain
    # "-?" pattern would read as a positive number.
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    return {_normalize_number(n) for n in re.findall(r"-?\d+(?:\.\d+)?", text)}


def verified(value, section_text):
    """Does this number actually occur in the real section text?

    Compared against the XML, not against the model's own quote, so a hallucinated
    value cannot validate itself. Trailing zeros are normalised so 1.20 matches 1.2.
    """
    return _normalize_number(f"{value}") in _numbers_in(section_text)


# ---------- provenance: which paper does this number come from? ----------
def refs_near_value(value, section_text, window=160):
    """Reference numbers cited just after `value` in the REAL section text."""
    if value is None:
        return []
    text = section_text.replace("−", "-").replace("–", "-")
    numbers = []
    for form in {f"{value:g}", f"{value}"}:
        for m in re.finditer(re.escape(form) + r"(?!\d)", text):
            tail = text[m.end():m.end() + window]
            found = xml_utils.CITATION.search(tail)
            if found:
                numbers += xml_utils.expand_ref_field(found.group(1))
    return sorted(set(numbers))


def harvest_refs(draft, section_text):
    """-> (reference numbers, how). Two independent signals, intersection preferred.

    Neither alone is good enough: the model quotes long sentences carrying several
    citations, and a bare number can occur more than once in a section. Where the
    two agree the answer has been right in every case checked.
    """
    from_text = set(refs_near_value(draft.value, section_text))

    from_llm = set(xml_utils.expand_ref_field(draft.ref_numbers or ""))
    for group in xml_utils.CITATION.findall(draft.source_text or ""):
        from_llm |= set(xml_utils.expand_ref_field(group))

    both = from_text & from_llm
    if both:
        return sorted(both), "agreed"
    if from_text:
        return sorted(from_text), "text"
    if from_llm:
        return sorted(from_llm), "llm"
    return [], "none"


# ---------- units ----------
def canonical_unit(raw, prop):
    """Map a written unit onto the table's spelling. Unknown units are kept as-is.

    Never substitutes a default: a row where the model said null keeps null, because
    asserting a scale nobody stated is worse than having none.
    """
    if raw is None or not str(raw).strip():
        return None
    key = re.sub(r"\s+", " ", str(raw)).strip().lower()
    key = key.replace("−", "-").replace("·", "·")
    if key in config.UNIT_ALIASES:
        return config.UNIT_ALIASES[key]
    stripped = key.replace(" ", "")
    return config.UNIT_ALIASES.get(stripped, str(raw).strip())


# ---------- deduplication against the table ----------
def _table_index(long_csv=None, table_csv=None):
    """(property, frozenset(normalised components)) -> [(Measurement_key, value)]."""
    import pandas as pd

    long_path = long_csv or config.LONG_CSV
    table_path = table_csv or config.TABLE_CSV
    if not long_path.exists() or not table_path.exists():
        return {}

    table = pd.read_csv(table_path)
    components_by_row = {}
    for r in table.to_dict("records"):
        names = [r.get(f"Component_{i}") for i in (1, 2, 3)]
        names = [_norm_key(n) for n in names if isinstance(n, str) and n.strip()]
        components_by_row[r["Row_id"]] = frozenset(names)

    index = {}
    for r in pd.read_csv(long_path).to_dict("records"):
        key = (r["Property"], components_by_row.get(r["Row_id"], frozenset()))
        index.setdefault(key, []).append((r["Measurement_key"], float(r["Value"])))
    return index


def mark_duplicates(rows, rel_tol=1e-3, long_csv=None, table_csv=None):
    """Flag prose rows that restate a Table 2 measurement. The table is authoritative.

    Matched on component set + property + value only. Temperature and ratio are
    deliberately excluded: prose usually omits them, and where prose and table
    disagree on temperature for the same datum that is a discrepancy worth seeing,
    not a reason to count it twice.

    A value match with a DIFFERENT component set is recorded as 'value_only' and
    never dropped -- a different DES that happens to share a number is not a copy.
    """
    index = _table_index(long_csv, table_csv)
    if not index:
        return rows

    by_property = {}
    for (prop, _components), entries in index.items():
        by_property.setdefault(prop, []).extend(entries)

    for row in rows:
        if row.value is None or not row.components_resolved:
            continue
        components = frozenset(_norm_key(n) for n in row.components_resolved.split(";") if n)
        for key, value in index.get((row.property, components), []):
            if math.isclose(value, row.value, rel_tol=rel_tol, abs_tol=1e-9):
                row.duplicate_of = key
                row.duplicate_kind = "components+value"
                break
        if not row.duplicate_of:
            for key, value in by_property.get(row.property, []):
                if math.isclose(value, row.value, rel_tol=rel_tol, abs_tol=1e-9):
                    row.duplicate_kind = "value_only"     # informational, not a drop
                    break
    return rows


# ---------- read the DES formula the paper itself wrote ----------
#
# This exists because the model rewrites abbreviations, and its rewrites are often
# wrong in a way nothing downstream can catch: it expanded ChAc ("choline acetate")
# to "Choline chloride", TBAC ("...chloride") to "Tetrabutylammonium bromide", NaCl
# to "Tetrabutylammonium bromide". Those are all real Table 2 names, so they resolve
# cleanly and the alias file is never consulted. The paper's own wording has to win.

_CITATION_REF = re.compile(r"\[\d[\d,\s–—-]*\]")
_PARENS = re.compile(r"\([^()]*\)")
_COMPARATOR = re.compile(r"\s*[<>≤≥≈~]\s*|\s+than\s+")
# Split a clause into segments. The space after the comma is load-bearing: a bare
# comma would break "1,3-pentanediol" and "N,N-dimethyl-N,N-didodecylammonium chloride".
_SEGMENT = re.compile(
    r",\s|;|\.{2,}|\s+(?:is|are|was|were|lie|lies|and|of|the|for|in|to|at|with|"
    r"results?|values?|range|based)\s+", re.I)
# A hyphen only separates components when the left side looks like an abbreviation
# (two consecutive capitals). Without this guard "d-Fructose", "10-undecylenic" and
# "amino acids-based" would all be split apart.
_ABBR_DASH = re.compile(r"^([A-Za-z0-9\[\]]*[A-Z]{2}[A-Za-z0-9\[\]]*)-(?=[A-Za-z])")


def formula_clauses(source_text):
    """-> [(raw clause, [component tokens])] for every DES formula in the quote.

    Comparison operators separate clauses, because one sentence often ranks several
    DESs: "TEAB-Chloroacetic acid (1:2, -79.2 C) [1] < TBAB-Chloroacetic acid ...".
    Getting the clause right is what stops a value being pinned to the wrong DES.
    """
    out = []
    for raw in _COMPARATOR.split(str(source_text or "")):
        if not raw.strip():
            continue
        # Strip citations and every parenthetical: that removes "(1:2, 23 C)" and
        # "(molar ratio 1:1:1, density 1.7012)", so any colon left is a formula colon.
        cleaned = _PARENS.sub(" ", _CITATION_REF.sub(" ", raw))
        for segment in _SEGMENT.split(cleaned):
            segment = segment.strip()
            if not segment:
                continue
            if ":" in segment:
                tokens = [t for t in segment.split(":") if t.strip()]
            else:
                dash = _ABBR_DASH.match(segment)
                if not dash:
                    continue
                tokens = [dash.group(1), segment[dash.end():]]
            tokens = [t.strip(" ,.;-") for t in tokens]
            if len(tokens) >= 2 and all(re.search(r"[A-Za-z]{2}", t) for t in tokens):
                out.append((raw, tokens[:3]))
                break                              # one formula per clause is enough
    return out


def components_from_source_text(source_text, value, model_resolved, index):
    """Resolve the DES formula the paper wrote. -> (resolved, unresolved, "source_text")

    Returns None when the quote states no formula at all -- an ordinal sentence like
    "[BF4]- (67 C) > acetate (18 C) > Cl- (12 C)" -- so the caller can tell "no
    formula" from "a formula I could not resolve", which are opposite situations.

    When a formula IS found it wins completely, unresolved tokens included. Falling
    back to the model for a token we cannot resolve would reintroduce exactly the bug
    this function exists to fix.
    """
    clauses = formula_clauses(source_text)
    if not clauses:
        return None

    if len(clauses) > 1:
        wanted = _normalize_number(f"{value}") if value is not None else None
        model_set = {m.lower() for m in (model_resolved or [])}

        def rank(item):
            raw, tokens = item
            resolved = [resolve_phrase(t, index) for t in tokens]
            # 1. the clause that actually contains this row's number -- the strongest
            #    signal, and independent of anything the model said
            has_value = bool(wanted) and wanted in _numbers_in(raw)
            overlap = sum(1 for r in resolved if r and r.lower() in model_set)
            return (has_value, overlap, sum(1 for r in resolved if r))

        clauses = [max(clauses, key=rank)]

    _raw, tokens = clauses[0]
    resolved, unresolved = [], []
    for token in tokens:
        found = resolve_phrase(token, index)
        (resolved if found else unresolved).append(found or token.strip())
    return resolved, unresolved, "source_text"


# ---------- chemdataextractor hook ----------
def normalize_components(names, section_text=""):
    """Placeholder for chemical named-entity recognition.

    When chemdataextractor is added, this is the one place it plugs in:

        from chemdataextractor import Document
        known = {c.text.lower() for c in Document(section_text).cems}
        return [n for n in names if n.lower() in known]

    That would both filter out model-invented names and give the enrichment step a
    vetted vocabulary. Until then this is the identity function, so the rest of the
    pipeline needs no changes when it lands.
    """
    return [n.strip() for n in names if n and n.strip()]


# ---------- one section ----------
def _status_for(row):
    """What keeps this row out of the graph. First match wins; see schema.STATUS_ORDER."""
    if row.value is None:
        return "qualitative"            # the text ranked, it did not measure
    if not row.verified:
        return "unverified"             # the number is not in the section text
    if row.duplicate_of:
        return "duplicate"              # Table 2 already has it
    if row.component_status != "resolved":
        return "unresolved_components"  # we will not guess at a chemical name
    return "ok"


def extract_section(section_id, title, text, review_doi="", reference_map=None,
                    index=None, backend=None, allow_lookup=False, refresh_llm=False):
    """-> (list[LLMMeasurement], elapsed_seconds, was_cached) for one prose section."""
    hint = config.PROPERTY_SECTIONS.get(title, "any of the six properties")
    prompt = PROMPT.format(title=title, property_hint=hint, text=text)

    started = time.time()
    raw, was_cached = cached_call_llm(prompt, LLMExtraction.model_json_schema(),
                                      section_id=section_id, backend=backend,
                                      refresh=refresh_llm)
    elapsed = time.time() - started

    config.RAW_LLM_DIR.mkdir(parents=True, exist_ok=True)
    (config.RAW_LLM_DIR / f"{section_id or 'section'}.json").write_text(raw, encoding="utf-8")

    try:
        drafts = LLMExtraction.model_validate_json(raw).measurements
    except ValidationError as exc:
        hint_text = ""
        if not raw.rstrip().endswith("}"):
            hint_text = f" -- output looks cut off; try a larger num_predict"
        print(f"    {section_id} {title!r}: output did not validate "
              f"({exc.errors()[0]['msg']}){hint_text}")
        return [], elapsed, was_cached

    normalised_text = re.sub(r"\s+", " ", text)
    rows = []
    for draft in drafts:
        written = normalize_components(draft.components, text)
        model_resolved, model_unresolved, _ = resolve_components(
            written, index, allow_lookup=allow_lookup)

        # The paper's own formula beats the model's expansion. Only when the quote
        # contains no formula at all do we fall back to what the model said.
        from_text = components_from_source_text(draft.source_text, draft.value,
                                                model_resolved, index)
        if from_text:
            resolved, unresolved, components_source = from_text
            components_written = ";".join(
                t for _raw, toks in formula_clauses(draft.source_text) for t in toks
            )[:200]
        else:
            resolved, unresolved = model_resolved, model_unresolved
            components_source, components_written = "model", ""
        component_status = ("resolved" if resolved and not unresolved
                            else "partial" if resolved else "unresolved")

        ratio = (draft.molar_ratio or "").strip() or None
        # Same naming convention as extract_table.parse_table2, which is what makes a
        # prose measurement attach to the Mixture node the table already created.
        mixture = f"{':'.join(resolved)} ({ratio})" if resolved and not unresolved and ratio else ""

        ref_numbers, ref_source = harvest_refs(draft, text)
        cited = sources(ref_numbers, reference_map)

        row = LLMMeasurement(
            components=";".join(written),
            components_written=components_written,
            components_source=components_source,
            components_resolved=";".join(resolved),
            unresolved_components=";".join(unresolved),
            component_status=component_status,
            molar_ratio=ratio,
            Mixture=mixture,
            property=draft.property,
            value=draft.value,
            unit=canonical_unit(draft.unit, draft.property),
            unit_raw=draft.unit,
            temperature_C=draft.temperature_C,
            source_text=draft.source_text or "",
            quote_found=bool(draft.source_text)
            and re.sub(r"\s+", " ", draft.source_text).strip() in normalised_text,
            section_id=section_id,
            section_title=title,
            Source_ref_numbers=",".join(str(n) for n in ref_numbers),
            Source_DOIs=cited["doi"],
            Source_paper_keys=cited["key"],
            ref_source=ref_source,
            Review_DOI=review_doi,
            verified=verified(draft.value, text) if draft.value is not None else False,
        )
        # Content-derived so re-loading is idempotent. Includes the components because
        # the same value recurs in one section for different DESs (melting point -10.0).
        stem = "+".join(sorted(resolved)) or "?"
        value = f"{draft.value:g}" if draft.value is not None else "none"
        row.Measurement_key = f"P-{section_id}:{draft.property}:{stem}:{value}"
        rows.append(row)
    return rows, elapsed, was_cached


# ---------- all sections ----------
def run(sections, review_doi="", reference_map=None, only_property_sections=True,
        backend=None, allow_lookup=True, refresh_llm=False):
    """Extract from every routed prose section. -> list[LLMMeasurement]."""
    chosen = [
        (sid, title, text) for sid, title, text in sections
        if not only_property_sections or title in config.PROPERTY_SECTIONS
    ]
    if not chosen:
        print("  no matching prose sections (use --all-sections to widen)")
        return []

    index = component_index()

    backend_name = backend or config.LLM_BACKEND
    detail = ""
    if backend_name == "ollama":
        detail = (f" ({config.OLLAMA_MODEL}, think={config.OLLAMA_THINK}, "
                  f"num_ctx={config.OLLAMA_NUM_CTX})")
    print(f"  {len(chosen)} section(s) via {backend_name}{detail}")

    rows, total_seconds, hits = [], 0.0, 0
    for sid, title, text in chosen:
        found, elapsed, was_cached = extract_section(
            sid, title, text, review_doi, reference_map=reference_map, index=index,
            backend=backend, allow_lookup=allow_lookup, refresh_llm=refresh_llm)
        total_seconds += elapsed
        hits += was_cached
        ok = sum(1 for r in found if r.verified)
        timing = "  cached" if was_cached else f"{elapsed:6.1f}s"
        print(f"    {sid} {title[:30]:<32} {timing}  "
              f"{len(found):>3} found, {ok:>3} verified")
        rows += found

    mark_duplicates(rows)
    for n, row in enumerate(rows, 1):
        row.Row_id = f"P-{n:04d}"
        row.status = _status_for(row)

    counts = Counter(r.status for r in rows)
    print(f"\n  cache: {hits} hit, {len(chosen) - hits} fresh")
    print(f"  {len(rows)} records in {total_seconds / 60:.1f} min")
    for status in STATUS_ORDER:
        if counts.get(status):
            note = " -> graph" if status == "ok" else ""
            print(f"    {status:<24}{counts[status]:>4}{note}")

    refs = Counter(r.ref_source for r in rows)
    print(f"  references: " + ", ".join(f"{k} {v}" for k, v in refs.most_common()))

    sources = Counter(r.components_source for r in rows)
    print(f"  components read from: " + ", ".join(f"{k} {v}" for k, v in sources.most_common()))

    unresolved = sorted({n for r in rows for n in r.unresolved_components.split(";") if n})
    if unresolved:
        print(f"  unresolved component names ({len(unresolved)}): {', '.join(unresolved)}")
        print(f"  -> add them to {config.COMPONENT_ALIASES.name}, or run "
              f"--steps aliases for candidates")
    return rows
