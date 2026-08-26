# DES Data Extraction

Centralise the extraction of relevant scientific data from open-access published papers
accessible from public repositories, and load it into a graph database.

The current scope is deep eutectic solvents (DES). Every XML file in `xml/` is processed,
in two publisher formats and both article types:

| paper | format | type | contributes |
|---|---|---|---|
| Omar & Sadeghi, *J. Mol. Liq.* **384** (2023) | Elsevier | review | 1649 measurements from a ~1500-row table |
| Canela-Xandri *et al.*, *RSC Adv.* **15** (2025) | JATS/PMC | review | 91 **application** rows across 7 domains |
| Fan *et al.*, *Polymers* **16** (2024) | JATS/PMC | **research** | 144 of its own thermal-conductivity and viscosity measurements |

Nothing is hard-coded to a particular paper. A table's layout is discovered rather than
declared (see [Reading a table you have never seen](#reading-a-table-you-have-never-seen)),
outputs are partitioned per paper under `data/papers/<slug>/`, and every row carries the
`Paper_key` it came from.

## The pipeline

```
 xml/*.xml
    │
    ├─ dialects.detect ──→ one reader per publisher format (Elsevier, JATS/PMC)
    │
    ├─ tables      → profile_table (LLM) → extract_table    → mixtures + measurements
    ├─ figures     ─────────────────────→ extract_figures   → figures.csv     (human)
    ├─ bibliography ────────────────────→ extract_references → references.csv (Crossref)
    └─ sections    ─────────────────────→ extract_text_llm  → sections_llm.csv (review)

 component names ───────────────────────→ enrich_components → components.csv  (PubChem)
                                          component_properties → every value + its source

 data/papers/*/ ───────────────────────→ validate           → report card + review queue
                 └────────────────────→ build_graph        → Neo4j
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
python run_pipeline.py --steps components --limit 20     # PubChem; drop --limit for all
python run_pipeline.py --steps validate                  # report card
python run_pipeline.py --steps validate --review         # + interactive spot check
python run_pipeline.py --steps graph --wipe
```

`route,refs,table,figures,text` run **once per paper**, over every file in `xml/`.
`components,aliases,validate,graph` run once across the whole corpus, reading
`store.read_all()`. Restrict a run with `--papers <slug>,<slug>`, and use
`--continue-on-error` so one bad paper does not stop the rest.

Order matters in two places: `table` and `text` both read the reference cache, so run
`refs` at least once first; and `text` runs before `components` so prose-only component
names exist by the time the PubChem lookup runs. `--steps all` already does both.

Prose measurements load into the graph by default; `--no-prose` skips them.

### Getting a paper

```bash
python fetch_paper.py 10.1016/j.rechem.2024.101378 --name Yeow_2024
```

Tries the publisher's Crossref-declared text-mining link, then the Elsevier article API,
then Unpaywall. **An Elsevier API key alone is not enough for full text** — the article
endpoint returns `403 AUTHENTICATION_ERROR` for every article, open-access ones included,
while the abstract and search endpoints work on the same key. You need an institutional
token in `.env` as `ELSEVIER_INSTTOKEN` (ask your library) or a request from your
institution's network. Until then, download the XML by hand and drop it in `xml/`; the
pipeline only needs the file to exist.

## What comes out

Everything lands in `data/` (gitignored).

Per-paper files live under `data/papers/<slug>/`; corpus-wide ones sit in `data/`.
Counts below are for the Sadeghi paper.

| File | Rows | What it is |
|---|---|---|
| `papers/<slug>/mixtures.csv` | 1539 | **one row per DES mixture** — components, ratios, every property as a value/unit/temperature triple, plus the paper's and every primary source's bibliographic metadata |
| `papers/<slug>/measurements.csv` | 1649 | one row per measurement; what the graph loads |
| `papers/<slug>/references.csv` | 343 | the bibliography, with Crossref DOIs, volume, issue, pages |
| `papers/<slug>/figures.csv` | 11 | worklist for manual digitisation (WebPlotDigitizer) |
| `papers/<slug>/sections_llm.csv` | 138 | prose measurements, `status` deciding which reach the graph |
| `papers/<slug>/tables_unhandled.csv` | 1 | tables that produced no data, with the reason |
| `papers/<slug>/table_profiles.json` | — | how each table was read; hand-overridable |
| `papers.csv` | 1 | the corpus index, one row per paper |
| `components.csv` | 498 | PubChem identifiers and descriptors |
| `papers/<slug>/applications.csv` | 91 | what a DES was used FOR, with the row's own detail columns |
| `papers/<slug>/des_definitions.csv` | 12 | abbreviation → components, from the paper's own table |
| `papers/<slug>/reference_map.json` | — | that paper's resolved references; **per paper, never shared** |
| `component_properties.csv` | 3369 | every reported component property value, with its source |
| `duplicate_measurements.csv` | 15 | the same datum reported twice, with both values and `agree` |
| `review/queue.csv` | 42 | the prioritised human worklist |

Partitioning per paper means re-running one paper of a thousand rewrites only its own
directory, and it makes it structurally impossible for paper B to overwrite paper A.
Every row also carries `Paper_key`, so a concatenation is self-describing.

The `Source_*` columns are multi-valued and positionally aligned: the n-th DOI in
`Source_DOIs` belongs to the n-th title in `Source_titles`. Papers are separated by `|`;
authors within one paper by `; `.

Don't open these in Excel — it silently rewrites the `Ratio_raw` column (`1:2` becomes
a time value).

## Publisher formats

`dialects.py` is the only place a publisher's tag names appear. A dialect answers seven
questions about a document — its tables, figures, references, bibliography, sections,
glossary and identity — and everything downstream works with the normalised `Table` and
`Cell` structures. `detect()` raises on an unknown format rather than guessing, because
guessing is what once filed one paper's data under another paper's DOI.

| | Elsevier | JATS/PMC |
|---|---|---|
| tables | `floats/table`, `tgroup/row/entry` | `table-wrap`, `thead/tr`, `td`/`th` |
| spans | `@morerows`, `@namest`/`@nameend` | `@rowspan`, `@colspan` |
| legend | `legend` | `table-wrap-foot` |
| references | `bib-reference`, always labelled `[40]` | `ref-list/ref`, often unlabelled |
| metadata | `coredata` | `front/article-meta` |
| article type | not stated | `article-type`, authoritative |

Two JATS properties remove guesswork rather than adding it. References frequently carry
their own DOI (76/203 and 33/34 in the two new papers), which beats a fuzzy Crossref
match and skips the search entirely. And `article-type` says outright whether a paper is
a review or primary research — which decides whether its measurements attach to the
studies it cites or to itself. Elsevier states neither, so the paper type falls back to
title keywords and then to the model.

### References belong to one paper

Reference numbers are meaningful only inside the paper that printed them: Sadeghi's [40]
is a book chapter, Canela-Xandri's [40] is a 2001 ionic-liquids paper. The resolved-DOI
cache is therefore **per paper**, at `data/papers/<slug>/reference_map.json`, and `paper`
is a required argument on every function that touches it. The single shared cache it
replaced was keyed on the number alone; the first run of a second paper would have
inherited the first paper's DOIs wholesale and pointed every `REPORTED_IN` edge at the
wrong study.

What *is* shared is `data/crossref_cache.json`, keyed by DOI — a DOI means the same thing
in every paper, so the second review to cite a study pays nothing for it.

Citations resolve through `<xref rid="cit59">` wherever the format provides it, falling
back to parsing the printed `[40,42-44]` for Elsevier. The rid is a link the publisher
asserted; the printed number is a convention.

## Reading a table you have never seen

Nothing tells the pipeline that melting point is column 3. For each table an LLM is shown
a *card* — the caption, the header rows with row-spans expanded, the legend, and five
sample rows — and returns a column map. Deterministic code then reads all 1539 rows from
that map. **No cell value is ever seen by the model**, so nothing can be hallucinated,
rounded or unit-converted.

```
Table 2   des_properties   10 cols, 10 markers
  col 3  property  Melting_point     (˚C)      col 6  property  Conductivity     (mS·cm−1)
  col 4  property  Density           (g·cm−3)  col 7  property  Surface_tension  (mN·m−1)
  col 5  property  Viscosity         (mPa·s)   col 8  property  Refractive_index  nD
  markers a→40 °C  b→20 °C  c→60 °C  d→45 °C  e→30 °C  f→35 °C  g→50 °C  h→55 °C
  default temperature 25 °C, missing values "–" and "-"
```

That output was checked against the hard-coded constants it replaced: it reproduced the
column map and all eight temperature markers exactly.

Two checks make it safe, both against data we already hold rather than anything the model
asserts:

1. **The model must echo each column's header verbatim**, and we compare it against the
   printed header. A map shifted by one column announces itself.
2. **A column labelled as a property must actually contain numbers.** This caught a real
   error: asked about Table 1 ("Types of eutectics"), the model called the *Formula*
   column — holding `Cat+X− zMClx` — a melting point, echoing the header correctly while
   doing so.

3. **The printed unit must not contradict the property.** `Conductivity` (mS·cm⁻¹) and
   `Thermal_conductivity` (W·m⁻¹·K⁻¹) are one word apart and both numeric, so neither
   check above separates them. `config.PROPERTIES` carries a `unit_pattern` for each.

A table failing a check extracts nothing and is written to `tables_unhandled.csv` with
the reason. Missing data is visible; a silently mislabelled column is not.

Some mistakes are repaired rather than rejected, where rejecting would cost more than it
protects:

- **A temperature column that is really the property.** Told that a `T/K` column is a
  measurement *condition*, the model relabelled Sadeghi's `Tm (˚C)` column `condition` —
  and 332 melting points stopped being extracted while every other property in the table
  was re-stamped with the melting point as its temperature. Prompt wording did not fix
  it; a deterministic rule does, promoting any `condition` column whose header says
  `Tm`/`Tb`/melting/boiling back to a property.
- **A property column in an application table.** A yield or a water equivalent is a real
  quantity but not one in the vocabulary, so the model picks the nearest name — this
  corpus produced `Yield model substrate (%)` labelled `Refractive_index`. Nothing reads
  properties out of an application table, so the column becomes `context` and keeps its
  value instead of failing nine good rows.

### Layouts, and who decides

Fan's data tables are **paneled**: three repeats of `(T, value)` with the DES named in a
row spanning each panel, not in any column.

```
thead  T/ | λ/ | T/ | λ/ | T/ | λ/
       K  | W·m−1·K−1 | K | ...
tbody  [ChCl][Gl]3 (colspan=2) | [ChCl][Gl]4 (colspan=2) | [ChCl][LA]3 (colspan=2)
       294.59 | 0.2385 | 294.67 | 0.2456 | 295.17 | 0.2137
```

The layout is decided by **code, not the model** — asked twice about the same table it
answered "paneled" once and "wide" once, and the wide reading extracts nothing. The
evidence is objective and already in hand: a repeating group of column roles containing a
property, a row that names mixtures across a span, and no column holding a mixture name.
The model only has to get the column meanings right, and the card marks spanning rows
explicitly so it does not read the band row's chemical name as a "component" column.

`[ChCl][Gl]3` means nothing on its own, so a `des_definitions` table — Fan's Table 2 —
is read first and supplies the mapping to `Choline chloride:Glycerol (1:3)`. Same
authority model as `component_aliases.json`, except the paper defines it in its own
table so nobody writes it by hand.

### Components the caption states and no column lists

Canela-Xandri's Tables 1 and 7 print only the HBA, because the caption already says every
mixture is "PTSA based"; Tables 4 and 5 spell out `ChCl : PTSA`. Without the implied
component the same solvent becomes two unrelated mixtures. The profiler returns
`implied_components` **and `implied_from`**, the caption words that justify it, which is
checked against the caption and recorded in `table_profiles.json` for a human to confirm.

The check allows for a transposition, because papers misspell their own subject: this one
captions two of its seven tables "PSTA based DES" and the other five "PTSA". A fuzzy match
is reported as fuzzy rather than passed off as exact.

The profile is cached, and `data/papers/<slug>/table_profiles.json` is hand-overridable:
set `"source": "human"` and it is used verbatim, with a `card_sha256` so a re-downloaded
XML cannot be read through a stale hand-written map.

Because the legend is now the authority on footnote markers rather than a constant, a
superscript only means a temperature if *that table* says so — which is what stops the
charge sign in `[Br⁻]` being read as one.

## Validating it

```bash
python run_pipeline.py --steps validate            # report card
python run_pipeline.py --steps validate --review   # + interactive spot check
```

Five separate questions, deliberately kept apart:

**Fidelity — did we transcribe what the paper prints?** Every measurement records its
table, row and column, so the source cell is re-read and the value re-derived. Exhaustive,
automatic, no judgement. A mismatch is always our bug and is a blocker.

```
fidelity         1649/1649 re-read identically
```

This is what makes it safe to change the extractor: the profile-driven rewrite was checked
this way, and separately against the old extractor — 0 differing values across 1494 matched
rows, plus 4 rows recovered that the old code silently dropped.

**Skipped cells — did we miss anything the paper prints?** Fidelity only inspects values
that *were* extracted, so it is blind to one that was dropped. This walks every property
cell that had content and produced nothing, records why, and writes
`data/review/<slug>_skipped_cells.csv`.

```
cells skipped    38   (36 no numeric value, 2 unparseable)
```

Most refusals are correct — this paper prints `DT` for "reported at several temperatures",
which is not a number, and two cells hold a source typo (`1. 2201`) or two numbers at once
(`b140 84.5`), where dropping beats guessing. But it is the only place a *silently* lost
value can surface, and it is how one was found after fidelity had reported 1648/1648: a
footnote marker trailing its number (`0.688a`) instead of leading it.

**Plausibility — is what the paper prints physically sensible?** Bounds live in
`config.PROPERTIES` beside each property. A violation is *flagged, not dropped*: checking
the first paper's outliers against its table showed all of them were faithful
transcriptions — the paper really does print a melting point of 2298 °C, a viscosity of
325,000 mPa·s and a conductivity of 1548 mS/cm. So this catches the **source** being wrong
or using a different unit, which only a human can adjudicate. The graph stays a faithful
record of the literature and carries `plausible: false`; a training set can filter.

**Invariants.** Every row has a `Paper_key`; `Measurement_key` is unique across papers;
every cited paper key has a reference row; no measurement dangles.

**A human spot check** — the only way to get a real error rate. It samples **source rows**
deterministically and shows everything the pipeline did with each one, column by column:

```
 1/20  SadeghiDESReview  t0010 row 675   (1 value(s))
    Tetrabutylammonium bromide | 1,5-Propanediol | 1:3
      col 3  Melting_point     –               not reported
      col 4  Density           DT              no numeric value
      col 5  Viscosity         e183            -> 183.0 mPa*s @ 30.0C
      col 6  Conductivity      –               not reported
      col 7  Surface_tension   –               not reported
      col 8  Refractive_index  –               not reported
      [212] -> 10.3390/molecules19068011

    all correct? [y/n/?/q]
```

The row, not the value, is the unit of review, because a table row carries 2.45 values on
average and up to 6. Showing one sampled measurement beside a row containing several
invites the reviewer to mark a correct row wrong for the values it appeared to omit — that
happened, and produced a 3/5 "error rate" on data that was entirely correct. Printing every
column, including the ones that yielded nothing **and why**, makes "nothing was missed"
something you can see rather than assume. 20 screens now cover ~49 values instead of 20.

Verdicts persist in `data/review/<slug>_spotcheck.csv`, keyed on `<table>:<source row>`, and
are read back so a row is never asked about twice. Quit with `q` and resume later.

**The review queue** (`data/review/queue.csv`) is one prioritised worklist rather than six
CSVs, ranked by how much data hangs on each decision: a table that failed profiling
(affects the whole table) → an unresolved abbreviation (every row using it) → an
implausible value → a doubtful Crossref match → an unverified prose value.

A paper is **accepted** when fidelity is 100%, at least one table profiled or all were
explicitly irrelevant, the spot check found nothing wrong, and under 2% of values are
implausible.

### Deduplication across papers

Every measurement carries a `Dedup_key` over its resolved components, ratio, property,
value and *primary source DOI* — so the same original measurement copied into two reviews
is recognised as one datum. `duplicate_measurements.csv` records every group with both
values and an `agree` column, because collapsing them silently would destroy the evidence
that two sources concur, or hide that they do not.

It already earns its keep on one paper: **15 groups**, where the review's own table lists
the same datum twice.

**The graph holds one node per datum, not per report of it.** Property nodes key on
`Dedup_key`, with a `REVIEW_PAPER`/`REPORTED_IN` edge to every paper that reports it and
the contributing `Measurement_key`s kept in `member_keys`, so a merged node is still
traceable to the exact cells it came from. 1793 measurement rows load as 1778 nodes.
Otherwise counting measurements would count publications.

Merging is **refused when the values disagree**: both nodes stay and the pair goes to the
review queue, because two papers printing different numbers for the same datum is a
finding, not noise.

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

**Abbreviations are resolved from the paper's own words, by lookup, never by guessing.**

Two separate things go wrong if you let the model name the chemicals. It expands
abbreviations incorrectly — reading `ChAc` (choline **acetate**) as "Choline chloride",
`TBAC` (…**chloride**) as "Tetrabutylammonium bromide", `NaCl` as "Tetrabutylammonium
bromide". And because those wrong answers are themselves real Table 2 entries, checking
against the table vocabulary does not catch them; it legitimises them. This was live: 15
distinct wrong mappings across 27 of 138 rows.

So the pipeline reads the DES formula out of the quoted `source_text` and resolves *that*:

```
source_text   "ChAc:EG(1:2, 23 °C) [113]"
              -> components_written  ChAc;EG
              -> components_resolved Choline acetate;Ethylene glycol
              -> components_source   source_text
```

The model's own list is used only when the quote contains no formula at all (an ordinal
sentence like `[BF4]− (67 °C) > acetate (18 °C) > …`). When one sentence ranks several
DESs, the clause containing the row's number wins, which is what stops a value being
pinned to the wrong solvent.

`des_pipeline/component_aliases.json` is the authority for what an abbreviation means. An
entry earns its place two ways only: the paper **defines it in its own text**
("choline acetate (ChAc)"), or it is one of a handful of universally-used ones (`ChCl`,
`ChAc`, `ChBr`, `EG`). Nothing else. An abbreviation that is not there does not resolve,
and rows using it stay out of the graph.

### Adding a paper

1. `--steps text` — extract (cached, so re-runs are free).
2. `--steps aliases` — proposes definitions two independent ways: every "Full name (ABBR)"
   the paper writes, matched with the Schwartz–Hearst algorithm, **and** for anything still
   unresolved, the Table 2 components whose measurements match the prose value.
3. Paste the entries you believe into `component_aliases.json`, with the paper's DOI in
   `source`.
4. Re-run `--steps text` for **every** paper — free, because responses are cached — so
   earlier papers are re-assessed against the grown map. Then reload the graph.

Read the `verdict` column. `conflicts_with_table` is the one that earns the tool its keep:

```
CONFLICT  LA: the text says 'Lactic acid', but this paper's own Table 2 data
          says 'Levulinic acid'. Leave it out unless you can tell which is which.
```

That is real — this paper defines `ChCl:Lactic acid (LA)` inline, yet its density section
writes `TBAB:LA (1:4, 1.1031)`, which matches Table 2 row T2-0625 *Levulinic acid* exactly
while TBAB+Lactic acid has zero rows. `LA` is therefore in the file with a `null` name,
which keeps both readings out.

Expect the graph to be conservative. On this paper only 1 of 138 prose rows is loaded: 84
state no number, 28 restate Table 2, and 25 use an abbreviation the paper never defines.
That is the intended trade — missing beats wrong — and later papers are expected to define
the rest.

A component named in prose but absent from Table 2 — `1,5-pentanediol` appears six times
in the text and nowhere in the table — is accepted only if **PubChem** recognises it, and
then flows into the enrichment step like any other component. A cheap word filter keeps
compound *classes* ("Amino acids", "Choline salt", "Tetraalkyl ammonium halides") and
non-chemicals ("HBD", "RCl") from being looked up at all.

### Component properties

Each component is looked up in PubChem for identifiers and descriptors (SMILES, InChI,
formula, MW, H-bond donor/acceptor counts, TPSA, rotatable bonds, formal charge, XLogP,
complexity) and in the NIST WebBook for phase-change data. All of it lands on the
`(:Component)` node.

**Hydrates** need their own route. Reviews write them the way a chemist does —
`FeCl3·6H2O` — and PubChem's name index accepts neither that nor `FeCl3 hexahydrate`, only
the fully spelled `iron(III) chloride hexahydrate`. So 15 of this paper's 18 hydrates
resolved to nothing at all. `_pubchem_hydrate` splits the water off, resolves the anhydrous
parent, and retries as `<synonym> hexahydrate`.

The water is not incidental — ChCl:FeCl3·6H2O is a different solvent from ChCl:FeCl3 — so
nothing here ever falls back to the anhydrous salt. Every hit is checked by **formula**:
it is accepted only if it is the parent plus exactly *n* waters. That is what makes
appending a word to a synonym safe, and it is why `KCr(SO4)2·10H2O` still resolves to
nothing: PubChem has chrome alum only as the dodecahydrate, and recording the wrong
hydration state would be worse than the gap. 16 of 18 now resolve; the two that do not are
that one and `Lanthanide nitrate six hydrate`, which names a class of elements rather than
a compound.

`matched_name` records the query that actually matched whenever it was not the name as
written, so a fuzzy PubChem hit stays auditable. It also drives the CAS lookup, which
matters: `Manganese chloride` returns CAS 7773-01-5, the *anhydrous* salt, where
`Manganese chloride tetrahydrate` returns 13446-34-9 — and CAS is what the NIST query uses.

PubChem usually reports a property **more than once, from different sources** — 208 of the
256 components with text do. The scalar on the node is just the first parseable value;
`component_properties.csv` and the graph keep them all:

```
Ammonium thiocyanate
  Melting Point: 320 °F (USCG, 1999)   CAMEO Chemicals                 -> 160.0 C
  Melting Point: 149.6 °C              Hazardous Substances Data Bank  -> 149.6 C
  Melting Point: 149.6 °C              PAC Chemical Database (DOE)     -> 149.6 C
  Boiling Point: Solid decomposes      CAMEO Chemicals                 -> qualifier
  Density: 1.3057 g/mL                 Hazardous Substances Data Bank  -> 1.3057
```

The LLM reads those lines, but **it returns a line number, never text and never a converted
unit**. That keeps every claim checkable: attribution comes from PubChem's own `Reference`
map rather than from the model, unit conversion happens in Python, and verification is
exact — the number must occur character-for-character in the line *we* fetched. Records
that fail land in the CSV with a `status` and stay out of the graph:

| status | meaning |
|---|---|
| `qualitative` | the line hedges — "Solid decomposes", "Sublimes" |
| `unverified` | the number is not on the line |
| `different_substance` | the value is for PEG 400, not for this component |
| `unhandled_unit` | a density in lb/gal or "relative density (water = 1)" |
| `ok` | loaded |

The pass is one call per component so the model never sees two substances at once, and it
is cached like the prose route, so re-running is free.

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

Responses are cached in `raw_responses/cache/`, keyed on the prompt, model and sampling
options — but deliberately **not** on the alias file, since resolution happens after the
call. So a second `--steps text` costs seconds instead of 35 minutes, which is what makes
re-assessing earlier papers practical. Editing `PROMPT` invalidates every entry; the run
prints `cache: 6 hit, 0 fresh` so that is impossible to miss. `--refresh-llm` forces a
re-call.

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

           -[:HAS_THERMAL_CONDUCTIVITY]->(:Thermal_conductivity ...)

(:Mixture) -[:USED_IN {detail, paper_key}]-> (:Application {domain})

(:Density) -[:REPORTED_IN {ref_numbers}]-> (:Paper {role:'primary'})   where the data came from
(:Density) -[:REVIEW_PAPER]-------------->  (:Paper {role:'review'})   where we read it
(:Mixture) -[:REPORTED_IN]/[:REVIEW_PAPER]-> (:Paper)
```

**A research paper is its own source.** A review's measurements point at the studies it
cites; Fan reports its own, so its 144 measurements carry `REPORTED_IN → Fan` and no
`REVIEW_PAPER` edge at all. Every `role` used to be hard-coded to `review`, which for a
primary paper records a review of nobody and loses the citation entirely.

### Schema reference

What each field means, then the generated inventory. Three things to know before writing
a query — all three have caught me out:

1. **A property label holds two different kinds of thing.** `(:Melting_point)` is 1182
   *pure-component* values (hanging off `(:Component)`) and 328 *mixture* values (off
   `(:Mixture)`). `origin` is the discriminator — `component`, `table` or `prose` — and a
   bare `MATCH (m:Melting_point)` returns both. The `component`-origin ones additionally
   carry `component`, `raw_string`, `extractor`, `qualifier` and `pressure`.
2. **`REPORTED_IN` has two ranges.** To `(:Paper)` it means a study; to `(:Source)` it
   means a database or an attribution within one (PubChem, NIST, HSDB, CAMEO). Same
   relationship type, different meaning — filter on the target label.
3. **Some properties exist for only one kind of thing.** All 976 `Boiling_point` nodes are
   pure components; no table in the corpus reports a DES boiling point. And
   `(:Refractive_index)` has **no `unit` property at all**, because it is dimensionless —
   `RETURN n.unit` on it returns null, correctly.

**Field meanings**

| field | on | what it is |
|---|---|---|
| `key` | property, Paper | the merge key. For a property node, `Dedup_key` when the datum is reported more than once, else `<slug>:<table>:<row>:<Property>`. For a Paper, the DOI, or `<paper>#refN` when Crossref found none |
| `name` | Component, Mixture | the merge key. `Mixture.name` is `"Choline chloride:Urea (1:2)"`, built from resolved component names so the same solvent from two papers is one node |
| `origin` | all | which route produced it: `table`, `prose`, `component` (a database lookup) or `application` |
| `value`, `unit`, `temperature_C` | property | the measurement. `unit` is canonicalised (`g*cm^-3`, `mPa*s`, `W*m^-1*K^-1`), not as the paper printed it |
| `dedup_key`, `member_keys`, `n_reports` | property | deduplication. `n_reports > 1` means several papers report this datum; `member_keys` lists the source cells it was merged from |
| `plausible`, `plausibility_note` | property | outside the physical range in `config.PROPERTIES`. Flagged, never dropped — usually the *source* is wrong, which only a human can adjudicate |
| `locus`, `evidence` | property | where in the paper. `locus` is the table row; `evidence` is the quoted sentence, for prose-route values |
| `role`, `extracted` | Paper | `review` or `primary`, and whether we extracted from it (3) or only cite it (567) |
| `match_score`, `title_agreement`, `raw` | Paper | how much to trust the Crossref match: its score, the XML-vs-Crossref title overlap, and the original citation string |
| `matched_name` | Component | the query PubChem actually matched, when it was not the name as written — `FeCl3·6H2O` → `FERRIC CHLORIDE hexahydrate` |
| `cid`, `cas`, `smiles`, `inchikey`, `formula` | Component | identifiers from PubChem; `377` of 565 components resolved |
| `tpsa`, `xlogp`, `complexity`, `h_bond_*`, `rotatable_bond_count`, `formal_charge` | Component | descriptors, for ML features |
| `melting_point_C`, `boiling_point_C`, `density_g_cm3` | Component | a convenience scalar — the *first* parseable value. The full multi-source set is on the property nodes |
| `molar_ratio`, `role` | `PART_OF` | the component's share of the mixture, and whether it is the HBA or HBD |
| `data_source` | `HAS_*` | which database within PubChem asserted a component property |
| `domain`, `detail` | Application, `USED_IN` | the use (`esterification`, `biodiesel`, …) and the row's own columns flattened as `alcohol=MeOH \| acid=Lauric`, since they differ per table |

**Inventory**

| label | n | properties |
|---|---|---|
| `:Mixture` | 1546 | `component_flag`, `n_components`, `name`, `origin`, `ratio_flag`, `ratio_raw`, `row_id` |
| `:Melting_point` | 1510 | `component`, `dedup_key`, `evidence`, `extractor`, `key`, `locus`, `member_keys`, `mixture`, `n_reports`, `origin`, `plausibility_note`, `plausible`, `pressure`, `property`, `qualifier`, `raw_string`, `temperature_C`, `unit`, `value` |
| `:Density` | 1116 | `component`, `dedup_key`, `evidence`, `extractor`, `key`, `locus`, `member_keys`, `mixture`, `n_reports`, `origin`, `plausibility_note`, `plausible`, `pressure`, `property`, `qualifier`, `raw_string`, `temperature_C`, `unit`, `value` |
| `:Boiling_point` | 976 | `component`, `extractor`, `key`, `origin`, `pressure`, `property`, `qualifier`, `raw_string`, `temperature_C`, `unit`, `value` |
| `:Paper` | 570 | `authors`, `doi`, `extracted`, `issue`, `journal`, `key`, `match_score`, `pages`, `raw`, `ref_number`, `role`, `title`, `title_agreement`, `volume`, `year` |
| `:Component` | 565 | `boiling_point_C`, `cas`, `cid`, `complexity`, `density_g_cm3`, `formal_charge`, `formula`, `h_bond_acceptor_count`, `h_bond_donor_count`, `inchikey`, `matched_name`, `melting_point_C`, `molecular_weight`, `name`, `origin`, `rotatable_bond_count`, `smiles`, `tpsa`, `xlogp` |
| `:Viscosity` | 505 | `dedup_key`, `evidence`, `key`, `locus`, `member_keys`, `mixture`, `n_reports`, `origin`, `plausibility_note`, `plausible`, `property`, `temperature_C`, `unit`, `value` |
| `:Conductivity` | 208 | `dedup_key`, `evidence`, `key`, `locus`, `member_keys`, `mixture`, `n_reports`, `origin`, `plausibility_note`, `plausible`, `property`, `temperature_C`, `unit`, `value` |
| `:Refractive_index` | 169 | `dedup_key`, `evidence`, `key`, `locus`, `member_keys`, `mixture`, `n_reports`, `origin`, `plausibility_note`, `plausible`, `property`, `temperature_C`, `value` |
| `:Surface_tension` | 115 | `dedup_key`, `evidence`, `key`, `locus`, `member_keys`, `mixture`, `n_reports`, `origin`, `plausibility_note`, `plausible`, `property`, `temperature_C`, `unit`, `value` |
| `:Thermal_conductivity` | 60 | `dedup_key`, `evidence`, `key`, `locus`, `member_keys`, `mixture`, `n_reports`, `origin`, `plausibility_note`, `plausible`, `property`, `temperature_C`, `unit`, `value` |
| `:Source` | 14 | `kind`, `name` |
| `:Application` | 7 | `domain`, `example_caption` |

| relationship | n | endpoints | properties |
|---|---|---|---|
| `REPORTED_IN` | 9973 | `(:Melting_point)`→`(:Source)` 2364<br>`(:Boiling_point)`→`(:Source)` 1952<br>`(:Mixture)`→`(:Paper)` 1697<br>`(:Density)`→`(:Source)` 1444<br>…8 more | `ref_numbers` |
| `PART_OF` | 3310 | `(:Component)`→`(:Mixture)` 3310 | `molar_ratio`, `role` |
| `REVIEW_PAPER` | 3104 | `(:Mixture)`→`(:Paper)` 1463<br>`(:Viscosity)`→`(:Paper)` 420<br>`(:Density)`→`(:Paper)` 394<br>`(:Melting_point)`→`(:Paper)` 328<br>…4 more | — |
| `HAS_MELTING_POINT` | 1510 | `(:Component)`→`(:Melting_point)` 1182<br>`(:Mixture)`→`(:Melting_point)` 328 | `data_source`, `temperature_C` |
| `HAS_DENSITY` | 1116 | `(:Component)`→`(:Density)` 722<br>`(:Mixture)`→`(:Density)` 394 | `data_source`, `temperature_C` |
| `HAS_BOILING_POINT` | 976 | `(:Component)`→`(:Boiling_point)` 976 | `data_source` |
| `HAS_VISCOSITY` | 504 | `(:Mixture)`→`(:Viscosity)` 504 | `temperature_C` |
| `HAS_CONDUCTIVITY` | 208 | `(:Mixture)`→`(:Conductivity)` 208 | `temperature_C` |
| `HAS_REFRACTIVE_INDEX` | 169 | `(:Mixture)`→`(:Refractive_index)` 169 | `temperature_C` |
| `USED_IN` | 118 | `(:Mixture)`→`(:Application)` 118 | `detail`, `implied_components`, `key`, `paper_key`, `source_row`, `table_id` |
| `HAS_SURFACE_TENSION` | 115 | `(:Mixture)`→`(:Surface_tension)` 115 | `temperature_C` |
| `HAS_THERMAL_CONDUCTIVITY` | 60 | `(:Mixture)`→`(:Thermal_conductivity)` 60 | `temperature_C` |

#### Regenerating this section

The inventory above is read out of the database, so it cannot drift from what is actually
loaded. Regenerate it after adding a property or a node type:

```bash
python tools/dump_schema.py > /tmp/schema.md    # then paste over the tables above
```

Or ad hoc, against a running database:

```cypher
CALL db.schema.nodeTypeProperties()
YIELD nodeLabels, propertyName
RETURN nodeLabels, collect(propertyName) AS properties;

MATCH (a)-[r]->(b)
RETURN type(r) AS rel, labels(a)[0] AS from, labels(b)[0] AS to, count(*) AS n
ORDER BY rel, n DESC;
```

### Applications

`(:Application {domain})` is one node per use — `esterification`, `biodiesel`,
`biomass_fractionation`, `metal_processing`, `fuel_cleaning`, `materials_synthesis`,
`reaction_medium` — taken from the table caption by keyword, deterministically.

The per-row detail rides on the `USED_IN` relationship, because the columns differ from
table to table (alcohol and acid here, biomass source and lignin yield there, material and
product elsewhere) and a union of every column any paper might print would be a schema
nobody could read. Those columns need no per-table code: they are whatever the profiler
labelled `context`, named by the `context_field` it already assigns.

```cypher
MATCH (c:Component)-[:PART_OF]->(m:Mixture)-[u:USED_IN]->(a:Application {domain:'esterification'})
RETURN m.name, u.detail
```

**Every** reference becomes a `Paper`, whether or not Crossref matched it. Nodes merge on
`key` — the DOI when there is one, otherwise `<review_doi>#refN` — because merging on a
null DOI would silently collapse all the unmatched references into one node. Each carries
`match_score`, `title_agreement` (XML title vs Crossref title overlap) and `raw`, so you
can judge a match rather than having it silently dropped. Four references here have no
DOI and are cited by 53 table rows between them.

```cypher
MATCH (p:Paper) WHERE p.doi IS NULL RETURN p.key, p.ref_number, p.raw;
MATCH (p:Paper) WHERE p.title_agreement < 0.5 RETURN p.doi, p.match_score, p.title;
```

Pure-component properties use the same labels, hanging off the `Component` rather than the
`Mixture`, with one node per reported value and each pointing at whoever reported it:

```
(:Component)-[:HAS_MELTING_POINT {data_source}]->
    (:Melting_point {origin:'component', value, unit, qualifier, pressure, raw_string})
        -[:REPORTED_IN]-> (:Source {name:'CAMEO Chemicals', kind:'attribution'})
        -[:REPORTED_IN]-> (:Source {name:'pubchem', kind:'database'})
```

Every measurement, mixture and component carries `origin` — `'table'`, `'prose'` or
`'component'` — so the sources stay separable. Prose measurements also carry `evidence`,
the sentence they were read from; component properties carry `raw_string`.

**Adding a property** is one entry in `config.PROPERTIES`. It flows automatically into the
pydantic `Literal`, the Neo4j label, the `HAS_<PROPERTY>` relationship and the uniqueness
constraint. `table2_column` and `pugview` say where (if anywhere) that property shows up in
the paper's table and in PubChem — boiling point has a PubChem heading but no table column,
which is exactly why the two lists are separate.

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

0. **A second dialect family.** Wiley and Springer are neither Elsevier nor JATS;
   `detect()` will raise on them by name, which is the intended failure.
1. Finding papers and fetching XML (pyalex / Unpaywall / publisher APIs).
2. PDF parsing (docling or similar) for papers with no XML.
3. Digitising the 11 figures.
4. chemdataextractor — there is a hook at `extract_text_llm.normalize_components`.
5. The text2cypher search agent.
6. **Cross-paper mixture normalisation.** Two papers naming the same solvent in a
   different component order produce two `Mixture` nodes. A merged measurement attaches
   to both, so nothing is lost, but the mixtures themselves do not yet merge.
7. **Component resolution for application rows.** `Alcohol` + `Acid` + an implied PTSA is
   four components against a three-slot schema; those rows carry `Component_flag =
   quaternary+`.
