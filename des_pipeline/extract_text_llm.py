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
import json
import re
import time

from pydantic import ValidationError

from . import config
from .schema import LLMExtraction, LLMMeasurement

PROMPT = """You extract deep eutectic solvent (DES) property measurements from a \
section of a chemistry review paper.

The section is titled "{title}", so the measurements it describes are most likely \
{property_hint}.

A valid measurement has:
  - at least two chemical component names (an HBA and an HBD, e.g. "choline chloride" \
and "urea"),
  - a numeric value for one of: Melting_point, Density, Viscosity, Conductivity, \
Surface_tension, Refractive_index.

Rules:
  - Only record values explicitly stated in the text. Never infer or calculate.
  - IGNORE author names, citation numbers, years and page numbers. They are not data.
  - Copy the exact sentence containing the value into source_text, verbatim.
  - If the section states no genuine DES measurements, return an empty list.

TEXT:
{text}
"""


# ---------- backends ----------
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
            options={
                "temperature": 0,                  # deterministic
                "num_ctx": config.OLLAMA_NUM_CTX,
                "num_predict": config.OLLAMA_NUM_PREDICT,
            },
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
def extract_section(section_id, title, text, review_doi="", backend=None):
    """-> (list[LLMMeasurement], elapsed_seconds) for one prose section."""
    hint = config.PROPERTY_SECTIONS.get(title, "any of the six properties")
    prompt = PROMPT.format(title=title, property_hint=hint, text=text)

    started = time.time()
    raw = call_llm(prompt, LLMExtraction.model_json_schema(), backend=backend)
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
        return [], elapsed

    rows = []
    for draft in drafts:
        components = normalize_components(draft.components, text)
        rows.append(LLMMeasurement(
            components=";".join(components),
            molar_ratio=draft.molar_ratio,
            property=draft.property,
            value=draft.value,
            unit=draft.unit,
            temperature_C=draft.temperature_C,
            source_text=draft.source_text,
            section_id=section_id,
            section_title=title,
            review_doi=review_doi,
            verified=verified(draft.value, text),
        ))
    return rows, elapsed


# ---------- all sections ----------
def run(sections, review_doi="", only_property_sections=True, backend=None):
    """Extract from every routed prose section. -> list[LLMMeasurement]."""
    chosen = [
        (sid, title, text) for sid, title, text in sections
        if not only_property_sections or title in config.PROPERTY_SECTIONS
    ]
    if not chosen:
        print("  no matching prose sections (use --all-sections to widen)")
        return []

    backend_name = backend or config.LLM_BACKEND
    detail = ""
    if backend_name == "ollama":
        detail = (f" ({config.OLLAMA_MODEL}, think={config.OLLAMA_THINK}, "
                  f"num_ctx={config.OLLAMA_NUM_CTX})")
    print(f"  {len(chosen)} section(s) via {backend_name}{detail}")

    rows, total_seconds = [], 0.0
    for sid, title, text in chosen:
        found, elapsed = extract_section(sid, title, text, review_doi, backend=backend)
        total_seconds += elapsed
        ok = sum(1 for r in found if r.verified)
        print(f"    {sid} {title[:30]:<32} {elapsed:6.1f}s  "
              f"{len(found):>3} found, {ok:>3} verified")
        rows += found

    total_ok = sum(1 for r in rows if r.verified)
    print(f"  {len(rows)} measurements, {total_ok} with a value present in the source text"
          f"  ({total_seconds / 60:.1f} min total)")
    return rows
