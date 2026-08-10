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
python run_pipeline.py --steps components --limit 20     # PubChem; drop --limit for all 494
python run_pipeline.py --steps text                      # LLM, needs `ollama serve`
python run_pipeline.py --steps graph --wipe
```

`table` reads the reference cache, so run `refs` at least once first.

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
| `sections_llm.csv` | — | prose measurements, flagged `needs_review` |

In `table2_with_dois.csv` the `Source_*` columns are multi-valued and positionally
aligned: the n-th DOI in `Source_DOIs` belongs to the n-th title in `Source_titles`.
Papers are separated by `|`; authors within one paper by `; `.

Don't open these in Excel — it silently rewrites the `Ratio_raw` column (`1:2` becomes
a time value).

### On the prose route

Routing one section at a time works, but local `qwen3` output is still weak. On the six
property sections it returned 34 measurements: values and components are broadly right
for viscosity and surface tension, but it invented two measurements outright for the
density section (components `A;B;C`, a sentence that is not in the paper) and folded
temperatures into the component list for melting point.

The `verified` column catches the fabrications — it checks each value against the real
XML section text, not against the model's own quoted sentence, so a hallucination cannot
validate itself. It flagged exactly those two rows and nothing else.

Component *attribution* is not checked and is where the model is weakest, so review
`sections_llm.csv` by hand before trusting it. If quality matters more than running
locally, set `DES_LLM_BACKEND=anthropic`.

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
| `DES_OLLAMA_NUM_CTX` | `8192` | ollama's own default of 4096 is too small here |
| `DES_OLLAMA_NUM_PREDICT` | `4096` | runaway guard only — anything near 1024 truncates the JSON and it fails to parse |
| `DES_OLLAMA_KEEP_ALIVE` | `30m` | the `components` step can run longer than ollama's 5-minute unload |

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
