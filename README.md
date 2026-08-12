# DES Data Extraction

Centralise the extraction of relevant scientific data from open-access published papers
accessible from public repositories, and load it into a graph database.

The current scope is deep eutectic solvents (DES), starting from one paper:
`SadeghiDESReview.xml` (Omar & Sadeghi, *J. Mol. Liq.* **384** (2023) 121899), whose
Table 2 lists ~1500 DES with their measured physical properties.

## The pipeline

```
                    ┌─ floats/table        → extract_table       → wide + long CSV
 XML ─→ router.py ──┼─ floats/figure       → extract_figures     → figures.csv  (human)
                    ├─ tail/bibliography   → extract_references  → references.csv (Crossref)
                    └─ body//section       → extract_text_llm    → sections_llm.csv (review)

 component names ──────────────────────────→ enrich_components   → components.csv (PubChem)

 all CSVs ─────────────────────────────────→ build_graph         → Neo4j
```

Routing is by document *structure*, which is deterministic — the pipeline never asks an
LLM what kind of thing it is looking at. Reliability decreases down the list, so the
table route feeds the graph directly while the figure and prose routes produce review
worklists instead.

## Setup

```bash
conda activate GraphDataBase
pip install -r requirements.txt
```

Copy your secrets into a gitignored `.env` at the repository root:

```
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
CROSSREF_MAILTO=you@example.com     # Crossref's "polite pool"
DES_LLM_BACKEND=ollama              # or "anthropic" (then set ANTHROPIC_API_KEY)
```

## Running it

Every step is independent, so you can iterate on one without re-hitting Crossref or
PubChem. Both network steps cache to disk and resume.

```bash
python run_pipeline.py --steps route                    # inventory, no side effects
python run_pipeline.py --steps refs,table --no-network   # ~5 s, uses the cache
python run_pipeline.py --steps refs                      # Crossref, ~1 min
python run_pipeline.py --steps figures
python run_pipeline.py --steps text                      # LLM, needs `ollama serve`
python run_pipeline.py --steps aliases                   # abbreviation candidates
python run_pipeline.py --steps components --limit 20     # PubChem; drop --limit for all 494
python run_pipeline.py --steps graph --wipe
```

Order matters in two places: `table` and `text` both read the reference cache, so run
`refs` at least once first; and `text` runs before `components` so prose-only component
names exist by the time the PubChem lookup runs. `--steps all` already does both.

Prose measurements load into the graph by default; `--no-prose` skips them.

## What comes out

Everything lands in `data/` (gitignored).

| File | Rows | What it is |
|---|---|---|
| `table2_with_dois.csv` | 1535 | **one row per DES mixture** — components, ratios, the six properties as value/unit/temperature triples, plus the review's and every primary paper's bibliographic metadata |
| `measurements_long.csv` | 1646 | one row per measurement; derived from the above, and what the graph loader reads |
| `references.csv` | 343 | the bibliography, with Crossref DOIs, volume, issue and pages |
| `components.csv` | 494 | PubChem identifiers and pure-component properties |
| `figures.csv` | 11 | worklist for manual digitisation (WebPlotDigitizer) |
| `tables_unhandled.csv` | 1 | tables with no parser yet (Table 1) |
| `sections_llm.csv` | — | prose measurements, with `status` deciding which reach the graph |
| `alias_suggestions.csv` | — | abbreviation candidates from `--steps aliases` |

In `table2_with_dois.csv` the `Source_*` columns are multi-valued and positionally
aligned: the n-th DOI in `Source_DOIs` belongs to the n-th title in `Source_titles`.
Papers are separated by `|`; authors within one paper by `; `.

Don't open these in Excel — it silently rewrites the `Ratio_raw` column (`1:2` becomes
a time value).

### On the prose route

Prose measurements are first-class: they attach to the same `Mixture` and `Component`
nodes as table data and carry the same `REPORTED_IN` / `REVIEW_PAPER` provenance. But
they only get there if they survive five checks, applied in order — the first failure
wins and sets `status`:

| status | meaning | in graph |
|---|---|---|
| `qualitative` | no number; the text only ranked or compared | no |
| `unverified` | the value does not occur in the real section text | no |
| `duplicate` | Table 2 already has it (`duplicate_of` names the row) | no |
| `unresolved_components` | a chemical name we refused to guess at | no |
| `ok` | — | **yes** |

`verified` compares against the XML section text, not the model's own quoted sentence, so
a hallucination cannot validate itself.

**Abbreviations are the hard part, and they are resolved by lookup, never by guessing.**
The prose writes `ChCl:EG(1:2)`; Table 2 writes the names out in full; and the paper
defines none of its abbreviations — ChCl (50 uses), TBAB (47) and TBAC (36) are never
spelled out anywhere. Asking qwen3 to expand them gets roughly half wrong, and wrong in a
way that looks right: it reads `TBAB` as tetra**ethyl**ammonium bromide when the paper
means tetra**butyl**, `Gly` as glycine when this paper means glycolic acid. Four such
wrong answers are themselves real Table 2 entries, so checking against the table
vocabulary does not catch them — it legitimises them.

So `des_pipeline/component_aliases.json` is authoritative. It is human-edited, and every
entry is evidence-backed rather than recalled: either the paper defines it inline, or a
prose measurement's value and ratio match exactly one Table 2 row that names the
components in full. `--steps aliases` regenerates candidates for anything unresolved:

```
abbreviation             uses  candidate from Table 2         confidence
TBAB                        8  Tetrabutylammonium bromide     strong
LA                          1  Levulinic acid                 strong
```

Treat it as a proposal, not an answer — for `TBAC` the top-ranked candidate is *Glycerol*,
which is wrong. Anything you do not confirm stays out of the graph. Note that `LA` means
**both** lactic and levulinic acid in this paper depending on the section, which is why it
is deliberately absent from the alias file.

The alias file is **per-paper**. A second review needs its own.

A component named in prose but absent from Table 2 — `1,5-pentanediol` appears six times
in the text and nowhere in the table — is accepted only if **PubChem** recognises it, and
then flows into the enrichment step like any other component. A cheap word filter keeps
compound *classes* ("Amino acids", "Choline salt", "Tetraalkyl ammonium halides") and
non-chemicals ("HBD", "RCl") from being looked up at all.

Component *attribution* is still where the model is weakest, so read `sections_llm.csv`
before trusting it. For better quality set `DES_LLM_BACKEND=anthropic`.

Two ollama settings matter for this route beyond speed. `num_ctx` must fit prompt plus a
long answer (~7000 tokens); at 4096 or 8192 the JSON is cut off mid-object and the whole
section is lost. And qwen3 ships with `repeat_penalty = 1`, i.e. none, which let it
degenerate mid-quote — one run got stuck on `"...It was shown that the viscos"` and emitted
`0` until it hit the token cap. Hence `DES_OLLAMA_REPEAT_PENALTY=1.1` and the instruction
to keep quotes short.

Re-running `--steps text` after a prompt change leaves orphaned prose nodes, because keys
are derived from content. Reload with `--steps graph --wipe`.

**Thinking is deliberately off.** qwen3 is a reasoning model and ollama enables thinking
by default, but this pipeline reads only `message.content` and discards
`message.thinking` — so every reasoning token is wasted work. Measured on the Melting
point section: 678 s producing 22,305 characters of discarded reasoning for a
1,865-character answer, versus 158 s with `think=False`. Ollama also caps the context at
4096 tokens by default regardless of the model, and one section prompt is already ~2400,
which left too little room to generate.

| Env var | Default | Why |
|---|---|---|
| `DES_OLLAMA_THINK` | `0` | set to `1` to re-enable reasoning; ~4x slower |
| `DES_OLLAMA_NUM_CTX` | `16384` | ollama's own default of 4096 is far too small; prompt plus answer runs to ~7000 tokens |
| `DES_OLLAMA_NUM_PREDICT` | `8192` | runaway guard only — set it too low and the JSON is truncated mid-object and fails to parse |
| `DES_OLLAMA_REPEAT_PENALTY` | `1.1` | qwen3 defaults to 1 (none) and degenerates while copying long quotes |
| `DES_OLLAMA_KEEP_ALIVE` | `30m` | steps either side can run longer than ollama's 5-minute unload |

## The graph

```
(:Component {name, smiles, cas, cid, formula})
    -[:PART_OF {molar_ratio, role}]-> (:Mixture {name, ratio_raw, n_components})

(:Mixture) -[:HAS_DENSITY {temperature_C}]-> (:Density {key, value, unit, temperature_C})
           -[:HAS_MELTING_POINT]->  (:Melting_point ...)
           -[:HAS_VISCOSITY]->      (:Viscosity ...)
           -[:HAS_CONDUCTIVITY]->   (:Conductivity ...)
           -[:HAS_SURFACE_TENSION]->(:Surface_tension ...)
           -[:HAS_REFRACTIVE_INDEX]->(:Refractive_index ...)

(:Density) -[:REPORTED_IN {ref_numbers}]-> (:Paper {role:'primary'})   where the data came from
(:Density) -[:REVIEW_PAPER]-------------->  (:Paper {role:'review'})   where we read it
(:Mixture) -[:REPORTED_IN]/[:REVIEW_PAPER]-> (:Paper)
```

Every measurement, mixture and component carries `origin` — `'table'` or `'prose'` — so
the two sources stay separable. Prose measurements also carry `evidence`, the sentence
they were read from.

```cypher
MATCH (m:Mixture)-[:HAS_VISCOSITY]->(v) WHERE v.origin = 'table' RETURN v.value;
```

The two relationship types are the point: `REPORTED_IN` reaches the paper that actually
measured the value, `REVIEW_PAPER` the review that tabulated it. Papers are attached at
the mixture level too, because 862 of the 1535 rows report no numeric value at all and
would otherwise have no provenance.

Current size: 1460 mixtures, 494 components, 338 papers, 1646 measurements.

```cypher
// find the source paper for a viscosity, with structures
MATCH (a:Component)-[:PART_OF]->(m:Mixture)-[:HAS_VISCOSITY]->(v)-[:REPORTED_IN]->(p:Paper)
WHERE a.name = 'Choline chloride'
RETURN m.name, a.smiles, v.value, v.temperature_C, p.title, p.journal, p.year
LIMIT 10;
```

## Layout

```
run_pipeline.py          the driver
des_pipeline/
  config.py              paths, constants, credentials from .env
  xml_utils.py           lxml primitives — one copy of each
  schema.py              pydantic models; field order is CSV column order
  router.py              classify the document's parts
  extract_table.py       Table 2 → mixtures + measurements
  extract_references.py  bibliography → Crossref DOIs and metadata
  extract_figures.py     figures → human worklist
  extract_text_llm.py    prose → LLM + pydantic  [chemdataextractor hook]
  enrich_components.py   component names → PubChem / NIST
  build_graph.py         CSVs → Neo4j
```

### Superseded

`extract_data.py`, `process_paper.py`, `resolve_references.py`, `extract_LLM.py` and
`graph_Sadeghi.py` are the prototypes this pipeline replaces. They are kept for now so
their output can be diffed against the new CSVs; delete them once you are satisfied.

`analyse_paper.py`, `paper.py`, `section.py`, `table.py`, `figure.py`, `reference.py`,
`component.py`, `physical_property.py`, `process_property.py`,
`create_Sadeghi_graphdb.py`, `graph_import_csv.py` and `toy_graph.py` are a collaborator's
object model. Nothing in `des_pipeline/` imports them.

## Not implemented yet

1. Finding papers and fetching XML (pyalex / Unpaywall / publisher APIs).
2. PDF parsing (docling or similar) for papers with no XML.
3. Digitising the 11 figures.
4. chemdataextractor — there is a hook at `extract_text_llm.normalize_components`.
5. The text2cypher search agent.
