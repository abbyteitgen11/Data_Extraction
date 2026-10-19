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
import threading
import time
from collections import Counter

from pydantic import ValidationError

from . import config, units, xml_utils
from .composition import des_problem, is_water
from .enrich_components import (_norm_key, component_index, resolve_components,
                                resolve_phrase)
from .extract_references import sources
from .schema import STATUS_ORDER, ApplicationRow, LLMExtraction, LLMMeasurement
from .textmatch import normalise, number_in, quote_found

PROMPT = """You extract two things from one section of a chemistry paper about deep \
eutectic solvents (DES): property MEASUREMENTS, and APPLICATIONS -- what a DES was \
used for.

The section is titled "{title}", so the measurements it describes are most likely \
{property_hint}.
{defined}
Return TWO lists. Either may be empty; most sections fill only one. An empty list is \
the right answer for most sentences.

=== measurements ===

One record per (DES, property, number) the text states. A record needs ALL of:
  - a SPECIFIC DES named in the same sentence -- by its formula, its full component
    names, or a code the paper defines. "DESs", "these solvents", "the mixtures" or
    "binary DES systems" are NOT specific: such a sentence gives no record, however
    many numbers it contains. Never supply a DES the sentence does not name.
  - a number the sentence states for that DES.

  components     The DES's constituents as the sentence writes them, HBA first. Keep
                 abbreviations as written ("ChCl", "TBAB"); do not expand them.
                 Never include a drug, solute or substrate dissolved IN the DES.
  molar_ratio    As written ("1:2"), only if this sentence states it. Otherwise null.
  property       One of: {properties}.
  value          The number as written. null if the sentence only ranks or compares.
                 NEVER invent, estimate, average or convert a number.
  unit           The unit exactly as written. null for a dimensionless property.
  temperature_C  The measuring temperature, only if separately stated. For a melting
                 or boiling point the number IS the temperature: leave this null.
  ref_numbers    The bracketed citation nearest the number, as written, or null.
  source_text    The sentence or clause holding the DES and the number, copied
                 CHARACTER FOR CHARACTER, at most about 150 characters. Do not add a
                 citation that is not inside that span, and do not paraphrase.

When one sentence gives several DES and several values ("A and B have densities of x
and y, respectively"), emit one record per DES, each with its own value, in order.
Emit each (DES, property, value) ONCE, however many times the text repeats it.

Example -- illustration only, this sentence is NOT in the text below:
  "TPAB:Gly (1:3) has a viscosity of 47.3 mPa s at 30 C [999]"
  -> {{"components": ["TPAB", "Gly"], "molar_ratio": "1:3", "property": "Viscosity",
       "value": 47.3, "unit": "mPa s", "temperature_C": 30.0, "ref_numbers": "999",
       "source_text": "TPAB:Gly (1:3) has a viscosity of 47.3 mPa s at 30 C [999]"}}

=== applications ===

One record per (DES, use) the text states: a named DES used AS or FOR something.

  components     The DES's constituents as written, HBA first. Same rules as above.
  molar_ratio    As written, only if this sentence states it. Otherwise null.
  domain         What it was used for, in the paper's own words.
  role           solvent, catalyst, extractant, electrolyte, sorbent, ... or null.
  target         What the DES acted ON: the drug, substrate, feedstock, analyte or
                 material. This is NEVER a component of the DES.
  outcome, outcome_unit, outcome_metric
                 A yield / efficiency / recovery number if stated, else null.
  ref_numbers    The citation for this claim, or null.
  source_text    The clause stating it, copied character for character, short.

A DES merely named, characterised or measured is not an application. A sentence about
DES in general names no DES and gives no record.

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


class BudgetExceeded(RuntimeError):
    """The run tag's spend reached DES_BUDGET_USD. Every answer so far is cached."""


_USAGE_LOCK = threading.Lock()
_SPENT = None                      # running total for this process, seeded from the CSV


def _spent_so_far():
    global _SPENT
    if _SPENT is None:
        _SPENT = 0.0
        if config.USAGE_CSV.exists():
            import csv

            with open(config.USAGE_CSV, newline="") as handle:
                for row in csv.DictReader(handle):
                    try:
                        _SPENT += float(row.get("cost_usd") or 0)
                    except ValueError:
                        pass
    return _SPENT


def log_usage(backend, model, section_id, input_tokens, output_tokens, seconds,
              cache_read=0, cache_write=0, stop_reason=""):
    """Append one row to USAGE_CSV: what a fresh model call cost in tokens, $ and time.

    This is what makes models comparable on cost and speed, not just accuracy. Cached
    answers are never logged -- they cost nothing -- so the file is the real bill.
    """
    global _SPENT
    price_in, price_out = config.ANTHROPIC_PRICES.get(model, (0.0, 0.0)) \
        if backend == "anthropic" else (0.0, 0.0)
    # Cache reads bill at ~0.1x and writes at ~1.25x the input price.
    cost = ((input_tokens or 0) * price_in + (output_tokens or 0) * price_out
            + (cache_read or 0) * price_in * 0.1
            + (cache_write or 0) * price_in * 1.25) / 1e6
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "run_tag": config.RUN_TAG,
           "backend": backend, "model": model, "call": section_id,
           "input_tokens": input_tokens or 0, "output_tokens": output_tokens or 0,
           "cache_read_tokens": cache_read or 0, "cache_write_tokens": cache_write or 0,
           "seconds": round(seconds, 2), "cost_usd": round(cost, 6),
           "stop_reason": stop_reason}
    with _USAGE_LOCK:
        import csv

        config.USAGE_CSV.parent.mkdir(parents=True, exist_ok=True)
        new = not config.USAGE_CSV.exists()
        with open(config.USAGE_CSV, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if new:
                writer.writeheader()
            writer.writerow(row)
        _SPENT = _spent_so_far() + cost


def _strict_schema(schema):
    """A pydantic JSON schema made acceptable to Anthropic structured outputs.

    Every object gets additionalProperties: false, and pydantic's `default` keys are
    dropped (the grammar does not use them, and every field we send is required).
    """
    if isinstance(schema, dict):
        out = {k: _strict_schema(v) for k, v in schema.items() if k != "default"}
        if out.get("type") == "object" and "properties" in out:
            out.setdefault("additionalProperties", False)
            out["required"] = list(out["properties"])
        return out
    if isinstance(schema, list):
        return [_strict_schema(v) for v in schema]
    return schema


def _anthropic_request(prompt, schema_json):
    """The Messages API request for one extraction call. -> (method name, kwargs).

    Structured outputs (output_config.format), not a forced tool call: Sonnet 5.5 and
    Opus 5.5 reject tool_choice "tool" with a 400. Effort is the depth control on the
    5.x models (Opus 5.5 cannot switch thinking off; low effort keeps a labelling task
    cheap); Haiku 4.5 takes neither effort nor adaptive thinking, so it gets neither.
    The 5.x requests also opt into server-side refusal fallbacks.
    """
    model = config.ANTHROPIC_MODEL
    kwargs = {
        "model": model,
        "max_tokens": config.ANTHROPIC_MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }
    output_config = {"format": {"type": "json_schema", "schema": _strict_schema(schema_json)}}
    is_5x = any(model.startswith(m) for m in ("claude-opus-5", "claude-sonnet-5",
                                              "claude-fable-5"))
    if is_5x and config.ANTHROPIC_EFFORT:
        output_config["effort"] = config.ANTHROPIC_EFFORT
    kwargs["output_config"] = output_config
    if model in ("claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1"):
        kwargs["betas"] = ["server-side-fallback-2026-07-01"]
        kwargs["fallbacks"] = "default"
        return "beta", kwargs
    return "standard", kwargs


def call_llm(prompt, schema_json, backend=None, section_id=""):
    """Send one prompt, return the raw JSON string. Backend chosen by config/env."""
    backend = backend or config.LLM_BACKEND
    started = time.time()
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
        log_usage(backend, config.OLLAMA_MODEL, section_id,
                  getattr(response, "prompt_eval_count", 0) or 0,
                  getattr(response, "eval_count", 0) or 0, time.time() - started,
                  stop_reason=response.done_reason or "")
        return response.message.content

    if backend == "anthropic":
        import anthropic

        if config.BUDGET_USD and _spent_so_far() >= config.BUDGET_USD:
            raise BudgetExceeded(f"spent ${_spent_so_far():.2f} of the "
                                 f"${config.BUDGET_USD:.2f} budget for run "
                                 f"{config.RUN_TAG or '(untagged)'}")
        client = anthropic.Anthropic(max_retries=4)
        kind, kwargs = _anthropic_request(prompt, schema_json)
        if kind == "beta":
            message = client.beta.messages.create(**kwargs)
        else:
            message = client.messages.create(**kwargs)
        usage = message.usage
        log_usage(backend, config.ANTHROPIC_MODEL, section_id,
                  usage.input_tokens, usage.output_tokens, time.time() - started,
                  getattr(usage, "cache_read_input_tokens", 0) or 0,
                  getattr(usage, "cache_creation_input_tokens", 0) or 0,
                  message.stop_reason or "")
        if message.stop_reason == "refusal":
            print(f"    {section_id}: the model declined; treated as an empty answer")
            return "{}"
        if message.stop_reason == "max_tokens":
            print(f"    warning: {section_id} hit max_tokens; the JSON is probably cut off")
        for block in message.content:
            if block.type == "text":
                return block.text
        return "{}"

    raise ValueError(f"unknown LLM backend {backend!r} (expected 'ollama' or 'anthropic')")


def map_calls(fn, items, backend=None):
    """fn over items, in order. Parallel for the API backend; sequential for ollama.

    A local ollama serves one request at a time, so threads would only queue. The API
    serves many, and a paper's sections are independent.
    """
    backend = backend or config.LLM_BACKEND
    items = list(items)
    if backend != "anthropic" or config.ANTHROPIC_CONCURRENCY <= 1 or len(items) < 2:
        return [fn(item) for item in items]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=config.ANTHROPIC_CONCURRENCY) as pool:
        return list(pool.map(fn, items))


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
        "options": _ollama_options() if backend == "ollama" else
        {"max_tokens": config.ANTHROPIC_MAX_TOKENS, "effort": config.ANTHROPIC_EFFORT},
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

    raw = call_llm(prompt, schema_json, backend=backend, section_id=section_id)
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

    from . import store

    table_rows = store.read_all("mixtures")
    long_all = store.read_all("measurements")
    if not table_rows or not long_all:
        return {}

    table = pd.DataFrame(table_rows)
    components_by_row = {}
    for r in table.to_dict("records"):
        names = [r.get(f"Component_{i}") for i in (1, 2, 3)]
        names = [_norm_key(n) for n in names if isinstance(n, str) and n.strip()]
        components_by_row[r["Row_id"]] = frozenset(names)

    index = {}
    for r in long_all:
        # A range-only cell ("3200-3500") has no Value; there is nothing to match.
        if r.get("Value") is None or r.get("Value") == "":
            continue
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
    """What keeps this row out of the graph. First match wins; see schema.STATUS_ORDER.

    The quote is checked FIRST and the number is checked IN THE QUOTE. The number used
    to be checked anywhere in the section, and the quote not at all, so a row whose
    quote the model had paraphrased -- or invented -- passed as long as its number
    happened to occur somewhere in the section (10 of 14 accepted rows, once).
    """
    if row.value is None:
        return "qualitative"            # the text ranked, it did not measure
    if not row.quote_found:
        return "quote_not_found"        # the quote is not in the section
    if not row.verified:
        return "value_not_in_quote"     # the number is not in that quote
    if not row.components_resolved and not row.components_written:
        return "no_des_in_quote"        # the quote names no specific DES
    if row.unit and row.unit.startswith("!"):
        return "unhandled_unit"
    if row.duplicate_of:
        return "duplicate"              # a table already has it
    if row.component_status != "resolved":
        return "unresolved_components"  # we will not guess at a chemical name
    return "ok"


def property_for_title(title):
    """Which property a section title is about. -> name, or "".

    Keyword matching, because exact equality was one paper's table of contents: across
    the ten new papers `title in config.PROPERTY_SECTIONS` selected 3 sections out of
    203, and rejected "2.2.1. Melting Point of Deep Eutectic Solvents" for carrying a
    section number.
    """
    stripped = re.sub(r"^[\d.\s]+", "", str(title or "")).strip().lower()
    for name, pattern in config.PROPERTY_SECTION_KEYWORDS:
        if re.search(pattern, stripped):
            return name
    return ""


def is_application_section(title):
    """Does this section's title suggest it says what a DES was USED FOR? -> bool."""
    stripped = re.sub(r"^[\d.\s]+", "", str(title or "")).strip().lower()
    return any(re.search(p, stripped) for p in config.APPLICATION_SECTION_KEYWORDS)


def extract_section(section_id, title, text, paper_doi="", reference_map=None,
                    index=None, backend=None, allow_lookup=False, refresh_llm=False,
                    dictionary=None, resolve=None):
    """-> (measurements, applications, elapsed_seconds, was_cached) for one section."""
    name = property_for_title(title)
    hint = name or "any of the properties listed"
    defined = _dictionary_hint(dictionary)
    prompt = PROMPT.format(title=title, property_hint=hint, text=text,
                           properties=", ".join(config.PROPERTY_NAMES),
                           defined=(f"\nDES THIS PAPER DEFINES (use these codes when the "
                                    f"text uses them):\n{defined}\n" if defined else ""))

    started = time.time()
    raw, was_cached = cached_call_llm(prompt, LLMExtraction.model_json_schema(),
                                      section_id=section_id, backend=backend,
                                      refresh=refresh_llm)
    elapsed = time.time() - started

    config.RAW_LLM_DIR.mkdir(parents=True, exist_ok=True)
    (config.RAW_LLM_DIR / f"{section_id or 'section'}.json").write_text(raw, encoding="utf-8")

    try:
        parsed = LLMExtraction.model_validate_json(raw)
        drafts, application_drafts = parsed.measurements, parsed.applications
    except ValidationError as exc:
        hint_text = ""
        if not raw.rstrip().endswith("}"):
            hint_text = " -- output looks cut off; try a larger num_predict"
        print(f"    {section_id} {title!r}: output did not validate "
              f"({exc.errors()[0]['msg']}){hint_text}")
        return [], [], elapsed, was_cached

    normalised_text = re.sub(r"\s+", " ", text)
    rows, seen = [], set()
    for draft in drafts:
        quote = re.sub(r"\s+", " ", draft.source_text or "").strip()
        found = quote_found(quote, text)
        comp = grounded_composition(draft.components, quote, draft.value, dictionary,
                                    resolve, index)

        # Solubility describes a SOLUTE in the DES. A name the model put among the
        # components that is not part of any DES the paper defines is the solute.
        solute = ""
        if draft.property in config.SOLUTE_PROPERTIES and dictionary is not None:
            members = {normalise(c) for e in dictionary.mixtures.values()
                       for c in e.components}
            outsiders = [c for c in comp.components if normalise(c) not in members
                         and not is_water(c)]
            if outsiders and len(comp.components) - len(outsiders) >= 2:
                solute = "; ".join(outsiders)
                comp.components = [c for c in comp.components if c not in outsiders]

        # Units through units.py, as the table route does: K -> C, Pa s -> mPa s. A
        # unit that measures something else entirely keeps the row out of the graph.
        written_value, unit_raw = draft.value, draft.unit
        value, unit = draft.value, canonical_unit(draft.unit, draft.property)
        if draft.value is not None:
            conflict = units.unit_conflict(unit_raw or "", draft.property)
            converted, canonical, status = units.to_canonical(draft.value, unit_raw or "",
                                                              draft.property)
            if conflict or status:
                unit = f"!{unit_raw}"           # marks unhandled_unit for _status_for
            else:
                value, unit = converted, canonical
        temperature = draft.temperature_C
        if draft.property in config.PHASE_TRANSITION_PROPERTIES:
            temperature = None

        key = (normalise(quote), draft.property, f"{draft.value}",
               frozenset(normalise(c) for c in comp.components))
        if key in seen:
            continue                       # the same claim, emitted twice
        seen.add(key)

        resolved = [c for c in comp.components if c]
        unresolved = [w for w in comp.notes if w.startswith("unresolved:")]
        component_status = ("unresolved" if not resolved else
                            "partial" if unresolved else "resolved")
        ratio = comp.ratio_raw or None
        mixture = comp.label() if resolved and not unresolved else ""

        ref_numbers, ref_source = harvest_refs(draft, text)
        cited = sources(ref_numbers, reference_map, paper_doi)

        row = LLMMeasurement(
            components=";".join(draft.components),
            components_written=";".join(comp.components_written)[:200],
            components_source=comp.source or "",
            components_resolved=";".join(resolved),
            unresolved_components=";".join(u.split(":", 1)[1].strip() for u in unresolved),
            component_status=component_status,
            molar_ratio=ratio,
            Mixture=mixture,
            property=draft.property,
            value=value,
            unit=unit,
            value_as_written=written_value,
            unit_raw=unit_raw,
            temperature_C=temperature,
            Solute=solute,
            DES_code=comp.code,
            source_text=quote,
            quote_found=found,
            section_id=section_id,
            section_title=title,
            Source_ref_numbers=",".join(str(n) for n in ref_numbers),
            Source_DOIs=cited["doi"],
            Source_paper_keys=cited["key"],
            ref_source=ref_source,
            Paper_DOI=paper_doi,
            verified=number_in(written_value, quote) if written_value is not None else False,
        )
        stem = "+".join(sorted(resolved)) or "?"
        shown = f"{draft.value:g}" if draft.value is not None else "none"
        row.Measurement_key = f"P-{section_id}:{draft.property}:{stem}:{shown}"
        rows.append(row)

    applications = _application_rows(application_drafts, section_id, title,
                                     normalised_text, paper_doi, reference_map,
                                     index, allow_lookup, dictionary, resolve)
    return rows, applications, elapsed, was_cached


def _application_rows(drafts, section_id, title, normalised_text, paper_doi,
                      reference_map, index, allow_lookup, dictionary=None, resolve=None):
    """Prose application drafts -> list[ApplicationRow].

    The same guards as the measurement route: the quote must really be in the section;
    the DES must be named IN the quote (or be one the paper defines); a molar ratio is
    kept only if the quote states it -- the model was supplying ratios no text gave.
    """
    from .extract_applications import domain_of

    out, seen = [], set()
    for n, draft in enumerate(drafts, 1):
        quote = re.sub(r"\s+", " ", draft.source_text or "").strip()
        found = quote_found(quote, normalised_text)
        comp = grounded_composition(draft.components, quote, None, dictionary,
                                    resolve, index)
        stated = draft.molar_ratio and normalise(draft.molar_ratio).replace(" ", "") in \
            normalise(quote).replace(" ", "")
        if draft.molar_ratio and not stated and comp.ratio_raw == draft.molar_ratio:
            comp.ratio_raw, comp.ratio_molar = "", ""
        if stated and not comp.ratio_raw:
            comp.ratio_raw = draft.molar_ratio

        key = (normalise(quote), frozenset(normalise(c) for c in comp.components))
        if key in seen:
            continue
        seen.add(key)

        numbers = xml_utils.expand_ref_field(draft.ref_numbers or "")
        cited = sources(numbers, reference_map, paper_doi)
        (c1, c2, c3), _parts, flag = comp.slots()
        unresolved = [w for w in comp.notes if w.startswith("unresolved:")]
        if not found:
            status = "unverified"
        elif not comp.components:
            status = "no_des_in_quote"
        elif des_problem(comp, dictionary):
            status = "not_des"
        elif unresolved:
            status = "unresolved_components"
        else:
            status = "ok"
        domain, detail = domain_of(draft.domain, draft.target, quote)
        out.append(ApplicationRow(
            Application_key=f"P-{section_id}:{n:03d}",
            Paper_key=paper_doi, Paper_DOI=paper_doi,
            Domain=domain, Domain_detail=(draft.domain or detail)[:200],
            Table_caption="",
            Component_1=c1, Component_2=c2, Component_3=c3,
            Ratio_raw=comp.ratio_raw,
            Mixture=comp.label(),
            Components_written=";".join(comp.components_written),
            Component_flag=flag,
            Detail=" | ".join(
                f"{k}={v}" for k, v in (("role", draft.role), ("target", draft.target),
                                        ("metric", draft.outcome_metric)) if v)[:500],
            Extractor="prose", Section_id=section_id, Section_title=title[:120],
            Role=(draft.role or "")[:80], Target=(draft.target or "")[:200],
            Outcome=draft.outcome, Outcome_unit=(draft.outcome_unit or "")[:20],
            Outcome_metric=(draft.outcome_metric or "")[:80],
            source_text=quote[:300], quote_found=found, status=status,
            DES_code=comp.code, Ratio_molar=comp.ratio_molar,
            Water_fraction=comp.water_fraction,
            Source_ref_numbers=",".join(str(x) for x in numbers),
            Source_DOIs=cited["doi"], Source_paper_keys=cited["key"],
            Source_titles=cited["title"], Source_years=cited["year"],
        ))
    return out


# ---------- all sections ----------
def choose_sections(sections, only_property_sections=True):
    """Which sections are worth a model call. -> [(id, title, text)].

    Two filters over one list, not one: a property is discussed in a section named
    after it, an application in Results / Extraction / Pretreatment. Selecting only the
    first kind found 3 of the ten new papers' 203 sections.

    Sections shorter than MIN_SECTION_CHARS are dropped whatever their title -- one
    review here has 82 sections, many of them a single heading line.
    """
    chosen = []
    for sid, title, text in sections:
        if len(text or "") < config.MIN_SECTION_CHARS:
            continue
        if not only_property_sections or property_for_title(title) \
                or is_application_section(title):
            chosen.append((sid, title, text))
    return chosen


def run(sections, paper_doi="", reference_map=None, only_property_sections=True,
        backend=None, allow_lookup=True, refresh_llm=False, dictionary=None, resolve=None,
        paper_slug=""):
    """Extract from every routed prose section. -> (measurements, rejected, applications).

    `rejected` holds the rows with no number at all (the text ranked or compared), kept
    out of sections_llm.csv so that file holds only real candidates.
    """
    chosen = choose_sections(sections, only_property_sections)
    if not chosen:
        print("  no matching prose sections (use --all-sections to widen)")
        return [], [], []

    index = component_index()

    backend_name = backend or config.LLM_BACKEND
    detail = ""
    if backend_name == "ollama":
        detail = (f" ({config.OLLAMA_MODEL}, think={config.OLLAMA_THINK}, "
                  f"num_ctx={config.OLLAMA_NUM_CTX})")
    else:
        detail = f" ({config.ANTHROPIC_MODEL})"
    print(f"  {len(chosen)} section(s) via {backend_name}{detail}")

    def one(item):
        sid, title, text = item
        # The paper's slug in the call id, so usage.csv attributes every call to its
        # paper (section ids like "sec2.1" repeat across papers).
        call_id = f"{paper_slug}-{sid}" if paper_slug else sid
        return item, extract_section(
            call_id, title, text, paper_doi, reference_map=reference_map, index=index,
            backend=backend, allow_lookup=allow_lookup, refresh_llm=refresh_llm,
            dictionary=dictionary, resolve=resolve)

    rows, applications, total_seconds, hits = [], [], 0.0, 0
    for (sid, title, _text), (found, used_for, elapsed, was_cached) in \
            map_calls(one, chosen, backend_name):
        total_seconds += elapsed
        hits += was_cached
        ok = sum(1 for r in found if r.quote_found and r.verified)
        timing = "  cached" if was_cached else f"{elapsed:6.1f}s"
        print(f"    {sid} {title[:30]:<32} {timing}  "
              f"{len(found):>3} found, {ok:>3} quoted+verified"
              f"{f', {len(used_for)} application(s)' if used_for else ''}")
        rows += found
        applications += used_for

    mark_duplicates(rows)
    for n, row in enumerate(rows, 1):
        row.Row_id = f"P-{n:04d}"
        row.status = _status_for(row)
        if row.unit and row.unit.startswith("!"):
            row.unit = row.unit[1:] or None
    measured = [r for r in rows if r.status != "qualitative"]
    rejected = [r for r in rows if r.status == "qualitative"]

    counts = Counter(r.status for r in rows)
    print(f"\n  cache: {hits} hit, {len(chosen) - hits} fresh")
    print(f"  {len(rows)} records in {total_seconds / 60:.1f} min")
    for status in STATUS_ORDER:
        if counts.get(status):
            note = " -> graph" if status == "ok" else ""
            print(f"    {status:<24}{counts[status]:>4}{note}")

    if applications:
        mark_duplicate_applications(applications)
        used = Counter(a.status for a in applications)
        quoted = sum(1 for a in applications if a.quote_found)
        print(f"  applications from prose: {len(applications)} "
              f"({quoted} with the quote found in the section)")
        for status, n in used.most_common():
            print(f"    {status:<24}{n:>4}{' -> graph' if status == 'ok' else ''}")
    return measured, rejected, applications


def mark_duplicate_applications(applications):
    """Flag a prose application that restates one a table already gave. -> None.

    The table is authoritative for the same reason it is for measurements: it was read
    from columns, not from a sentence. Matched on the mixture and the domain, which is
    all two accounts of the same use are guaranteed to share.
    """
    from . import store

    try:
        existing = store.read_all("applications")
    except Exception:
        return
    seen = {}
    for row in existing:
        if (row.get("Extractor") or "table") != "table":
            continue
        key = (_norm_key(row.get("Mixture", "")), (row.get("Domain") or "").lower())
        seen.setdefault(key, row.get("Application_key", ""))
    for app in applications:
        key = (_norm_key(app.Mixture), (app.Domain or "").lower())
        if key in seen and app.status == "ok":
            app.duplicate_of = seen[key]
            app.status = "duplicate"


def _dictionary_hint(dictionary, limit=1500):
    """The paper's own DES, for the prompt, so the model can name them by code."""
    if dictionary is None or not dictionary.mixtures:
        return ""
    lines = []
    for comp in dictionary.mixtures.values():
        if comp.code:
            lines.append(f"  {comp.code} = {comp.label()}")
    text = "\n".join(lines)
    return text[:limit]


def all_formulas(quote):
    """Every DES formula in a quote, in reading order. -> [(position, [tokens])].

    `formula_clauses` keeps one formula per clause; a "respectively" sentence names
    several in one clause ("ChCl:Phenol and ChCl:O-cresol ... 1.097 and 1.078 g/cm3,
    respectively"), and their ORDER is what assigns the numbers.
    """
    out = []
    cleaned = _PARENS.sub(lambda m: " " * len(m.group(0)),
                          _CITATION_REF.sub(lambda m: " " * len(m.group(0)), quote))
    for match in re.finditer(r"[A-Za-z\[][\w\[\]\-,.+()]*?(?:\s?:\s?[A-Za-z\[][\w\[\]\-,.+]*)+",
                             cleaned):
        tokens = [t.strip(" ,.;-") for t in match.group(0).split(":")]
        if len(tokens) >= 2 and all(re.search(r"[A-Za-z]{2}", t) for t in tokens):
            out.append((match.start(), tokens[:4]))
    if not out:
        for raw, tokens in formula_clauses(quote):
            out.append((quote.find(raw), tokens))
    return out


_NUMBER_IN_TEXT = re.compile(r"(?<![\w.:])-?\d+(?:\.\d+)?(?![\w:]|\s*:\s*\d)")


def _value_numbers(quote):
    """The numbers in a quote that could be values: not ratios, not citations."""
    cleaned = _CITATION_REF.sub(" ", quote.replace("−", "-"))
    cleaned = re.sub(r"\d+(?:\.\d+)?(?:\s*:\s*\d+(?:\.\d+)?)+", " ", cleaned)
    return [float(m.group(0)) for m in _NUMBER_IN_TEXT.finditer(cleaned)]


def grounded_composition(model_components, quote, value, dictionary, resolve, index):
    """The DES a quote is about, read from the QUOTE. -> Composition.

    The model's own component list is used only for names that are actually written in
    the quote. When a sentence talks about DES in general, the model used to supply a
    specific one (ChCl:urea, ChCl:EG) that the sentence never mentions; now such a row
    has no components and is refused as `no_des_in_quote`.

    Order of evidence: a code the paper defines ("DES5", "CEM1"); a formula written in
    the quote ("ChCl:Urea (1:2)") -- for a "respectively" sentence, the formula whose
    position matches the value's position; the model's names, where the quote names them.
    """
    from .composition import Composition, compose, parse_mixture

    quote = str(quote or "")
    if dictionary is not None:
        for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]*\d+|\bDESs?\b", quote):
            comp = dictionary.lookup(token)
            if comp is not None:
                comp.source = "dictionary"
                return comp

    formulas = all_formulas(quote)
    if formulas:
        chosen = formulas[0][1]
        if len(formulas) > 1:
            numbers = _value_numbers(quote)
            respectively = re.search(r"\brespectively\b", quote, re.I)
            wanted = None if value is None else float(value)
            if respectively and wanted is not None and len(numbers) >= len(formulas):
                tail = numbers[-len(formulas):]
                for i, number in enumerate(tail):
                    if math.isclose(number, wanted, rel_tol=1e-6, abs_tol=1e-9):
                        chosen = formulas[i][1]
                        break
            else:
                # the formula nearest BEFORE the value
                position = _position_of(quote, value)
                before = [f for f in formulas if position is None or f[0] <= position]
                chosen = (before[-1] if before else formulas[0])[1]
        written = ":".join(chosen)
        ratio = re.search(r"\(\s*(\d+(?:\.\d+)?(?:\s*:\s*\d+(?:\.\d+)?)+)", quote)
        comp = parse_mixture(written, dictionary, resolve)
        if ratio and not comp.ratio_raw:
            comp = compose([written], ratio.group(1), "", dictionary=dictionary,
                           resolve=resolve)
        comp.source = "source_text"
        return _mark_unresolved(comp, index)

    q = normalise(quote)
    written_abbreviations = set()
    if dictionary is not None:
        for abbr, full in dictionary.aliases.items():
            if re.search(rf"(?<!\w){re.escape(abbr)}(?!\w)", quote, re.I):
                written_abbreviations.add(normalise(full))
    # A model name counts only if the quote writes it, or writes the paper's own
    # abbreviation for it. Nothing is added that the quote does not name.
    named = [c for c in (model_components or [])
             if c and (normalise(c) in q or normalise(c) in written_abbreviations)]
    if len(named) >= 2:
        comp = compose(named, "", "", dictionary=dictionary, resolve=resolve)
        comp.source = "model_names_in_quote"
        return _mark_unresolved(comp, index)
    return Composition(written=quote[:80], source="none")


def _position_of(quote, value):
    if value is None:
        return None
    for form in {f"{value:g}", f"{value}"}:
        i = quote.replace("−", "-").find(form)
        if i >= 0:
            return i
    return None


def _mark_unresolved(comp, index):
    """Note components that neither the paper nor the corpus can name."""
    if index is None:
        return comp
    from .enrich_components import resolve_component

    for written, name in zip(comp.components_written, comp.components):
        if written == name and not is_water(name):
            found, _how = resolve_component(name, index, allow_lookup=False)
            if not found:
                comp.note(f"unresolved: {name}")
    return comp
