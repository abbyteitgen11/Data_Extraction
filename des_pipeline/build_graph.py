"""
Load the CSVs into Neo4j.

Graph shape (each property gets its own node label, so you can query
``MATCH (m:Mixture)-[:HAS_DENSITY]->(d)`` directly):

    (:Component {name, smiles, cas, cid, formula})
        -[:PART_OF {molar_ratio, role}]-> (:Mixture {key, name, ratio_raw, ...})

    (:Mixture) -[:HAS_DENSITY]-> (:Density {key, value, unit, temperature_C})
        ... and HAS_MELTING_POINT, HAS_VISCOSITY, HAS_CONDUCTIVITY,
            HAS_SURFACE_TENSION, HAS_REFRACTIVE_INDEX, HAS_THERMAL_CONDUCTIVITY

    (:Density)  -[:REPORTED_IN {role, ref_numbers}]-> (:Paper)
    (:Mixture)  -[:REPORTED_IN {role}]--------------> (:Paper)

    (:Mixture)  -[:HAS_APPLICATION]-> (:ApplicationRecord {key, member_keys, source_text})
                                          -[:IN_DOMAIN]-> (:Application {domain})
                                          -[:REPORTED_IN {role}]-> (:Paper)

Papers are attached at the mixture level as well as the measurement level because
862 of the 1535 table rows report no numeric value at all — without the mixture
edges those DES would have no provenance in the graph.

Two rules keep the schema honest:

  * **One identity per node type, chosen so identical things collide.** Component and
    Source merge on `name`, Application on `domain`, Paper on `key` (its DOI, or
    "<paper>#refN" when Crossref found none — merging on a null DOI would silently
    collapse every unmatched reference into one node). Mixture and the property nodes
    merge on a *content hash*: a mixture's printed name depends on the order a table
    listed its components, so the same solvent had two names and became two nodes.

  * **Each fact is written once.** `temperature_C` lives on the property node and NOT
    on the HAS_* relationship. It used to be on both — identical wherever both existed,
    but missing from 383 relationships whose node had one, so a query on the
    relationship silently lost data. For the same reason `HAS_*.data_source` is gone
    (all 2880 duplicated an attribution `(:Source)` the property already points at), a
    component's identifiers are written only by COMPONENTS_CYPHER, and `role` sits on
    the REPORTED_IN edge rather than being inferred from the Paper it points at.

`role` is on the edge, not only on `(:Paper)`, because a paper's own type cannot say
whether it MEASURED a particular datum or merely tabulated it — a review that also
publishes original measurements needs both answers about itself.

Loads are idempotent: every node has a deterministic key, so re-running updates
rather than duplicates. ``--wipe`` is therefore optional.
"""
import pandas as pd

from . import config
from .extract_table import _hash, mixture_key
from .config import PROPERTY_NAMES

# Written onto (:Component) as attributes, never as property nodes: they are DFT values
# for one isolated molecule in the gas phase, not measurements of a substance.
QM9_FIELDS = ("qm9_id", "qm9_n_matches", "qm9_contested", "qm9_dipole_D",
              "qm9_polarizability_a0_3", "qm9_homo_eV", "qm9_lumo_eV", "qm9_gap_eV",
              "qm9_r2_a0_2", "qm9_zpve_eV", "qm9_cv_cal_mol_K")

CONSTRAINTS = [
    "CREATE CONSTRAINT component_name IF NOT EXISTS "
    "FOR (c:Component) REQUIRE c.name IS UNIQUE",
    # On `key`, not `name`: a mixture's printed name depends on the order a table
    # listed its components, so the same solvent can have two names and must not
    # become two nodes. `key` hashes the component SET plus the ratio.
    "CREATE CONSTRAINT mixture_key IF NOT EXISTS "
    "FOR (m:Mixture) REQUIRE m.key IS UNIQUE",
    "CREATE INDEX mixture_name_index IF NOT EXISTS FOR (m:Mixture) ON (m.name)",
    "CREATE CONSTRAINT paper_key IF NOT EXISTS "
    "FOR (p:Paper) REQUIRE p.key IS UNIQUE",
    "CREATE INDEX paper_doi_index IF NOT EXISTS FOR (p:Paper) ON (p.doi)",
    "CREATE CONSTRAINT source_name IF NOT EXISTS "
    "FOR (s:Source) REQUIRE s.name IS UNIQUE",
] + [
    f"CREATE CONSTRAINT {prop.lower()}_key IF NOT EXISTS "
    f"FOR (x:{prop}) REQUIRE x.key IS UNIQUE"
    for prop in PROPERTY_NAMES
]


def _driver():
    from neo4j import GraphDatabase

    if not config.NEO4J_PASSWORD:
        raise SystemExit(
            "NEO4J_PASSWORD is not set. Put it in a .env file at the repository root:\n"
            "    NEO4J_PASSWORD=your-password"
        )
    driver = GraphDatabase.driver(
        config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )
    driver.verify_connectivity()
    return driver


def _read(path):
    """Read a CSV with NaN turned into None, so `IS NOT NULL` behaves in Cypher."""
    df = pd.read_csv(path)
    return df.astype(object).where(pd.notna(df), None)


def _split(value):
    """Split a '|'-joined Source_* cell into a list."""
    if not value:
        return []
    return [v for v in str(value).split(config.SOURCE_SEP) if v]


def _str(value):
    """Stringify a CSV cell without pandas' float artefacts (2002.0 -> '2002')."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _canonical(name, mapping):
    """The surviving spelling for a component name. -> str.

    `mapping` comes from data/review/component_duplicates.csv and contains ONLY groups
    a human marked `yes`, so an unreviewed or doubted duplicate simply passes through
    unchanged. The CSVs keep whatever the paper wrote; identity is resolved here.
    """
    if not name:
        return name
    return mapping.get(str(name).strip(), name)


# ---------- turn the wide CSV into the nested shape Cypher wants ----------
def _paper_rows(references):
    """One record per unique paper key.

    Keyed on `key`, not `doi`, so the references Crossref could not match still get
    a node. Two references can legitimately share a DOI, so the count is lower than
    the number of references.
    """
    papers = []
    seen = set()
    for r in references:
        key = r.get("key")
        if not key or key in seen:
            continue
        seen.add(key)
        papers.append({
            "key": key,
            "doi": r.get("doi"),
            "match_score": r.get("match_score"),
            "title_agreement": r.get("title_agreement"),
            "raw": r.get("raw") or "",
            "authors": r.get("authors") or "",
            "title": r.get("title") or "",
            "journal": r.get("journal") or "",
            "volume": _str(r.get("volume")),
            "issue": _str(r.get("issue")),
            "pages": _str(r.get("pages")),
            "year": _str(r.get("year")),
        })
    return papers


def _mixture_rows(table, components_by_name, roles=None, canonical=None):
    """One record per mixture, with components pre-filtered into a nested list.

    Doing the null-filtering in Python (rather than three FOREACH blocks in Cypher)
    is what stops a null Component_3 from becoming a `Component {name: null}` node.
    """
    canonical = canonical or {}
    rows, by_key, rekeyed = [], {}, {}
    for r in table:
        comps = []
        for i, role in ((1, "HBA"), (2, "HBD"), (3, "HBD")):
            name = r.get(f"Component_{i}")
            if not name:
                continue
            extra = components_by_name.get(name, {})
            name = _canonical(name, canonical)
            comps.append({
                "name": name,
                "ratio": r.get(f"Ratio_component_{i}"),
                "role": role,
                # The paper never gives SMILES; these come from enrich_components.
                "smiles": r.get(f"Component_{i}_SMILES") or extra.get("smiles"),
                "cas": extra.get("cas"),
                "cid": int(extra["cid"]) if extra.get("cid") is not None else None,
                "formula": extra.get("molecular_formula"),
            })
        key = mixture_key([c["name"] for c in comps], r.get("Ratio_raw"))
        # measurements.csv carries a Mixture_key computed at extraction time from the
        # ORIGINAL spellings. Canonicalising here changes the key, so record the
        # translation -- without it the measurement rows MERGE fresh Mixture nodes
        # under the stale keys, and the count goes UP instead of down.
        if r.get("Mixture_key"):
            rekeyed[r["Mixture_key"]] = key
        # Every table row that produced this mixture, not just whichever came first:
        # `row_id` held one of them and read as though it were the only one.
        by_key.setdefault(key, []).append(r["Row_id"])
        rows.append({
            "member_keys": by_key[key],
            "key": key,
            "mixture": r["Mixture"],
            "ratio_raw": r.get("Ratio_raw") or "",
            "ratio_flag": r.get("Ratio_flag") or "",
            "component_flag": r.get("Component_flag") or "",
            "components": comps,
            "paper_doi": r["Paper_DOI"],
            # A research paper is its own source, so its mixtures get REPORTED_IN
            # rather than REVIEW_PAPER; the Cypher branches on these two lists.
            "review_papers": ([] if (roles or {}).get(r.get("Paper_key")) == "primary"
                              else [r.get("Paper_key") or r["Paper_DOI"]]),
            "source_keys": (_split(r.get("Source_paper_keys")) or
                            ([r["Paper_key"]]
                             if (roles or {}).get(r.get("Paper_key")) == "primary"
                             else [])),
            "ref_numbers": r.get("Source_ref_numbers") or "",
        })
    return rows, rekeyed


def _measurement_rows(long_rows, roles=None, rekeyed=None):
    """One row per DISTINCT datum, not per report of it. -> list[dict].

    Two reviews tabulating the same primary measurement are reporting one fact, and
    the graph should hold one node for it with an edge to each paper -- otherwise
    counting measurements counts publications. `Dedup_key` already identifies them:
    same component set, ratio, property, value, temperature and originating study.

    The key is the dedup hash whether or not anything merged, so the field has one
    format rather than two. Values cannot disagree within a group: the value is part of
    the hash, so two different numbers produce two different groups.
    """
    roles = roles or {}
    groups = {}
    for r in long_rows:
        # Rows with no dedup key are their own group, keyed uniquely so they never merge.
        key = r.get("Dedup_key") or _hash([r["Measurement_key"]])
        groups.setdefault(key, []).append(r)

    return [_one_measurement(members[0], members, roles, key, rekeyed or {})
            for key, members in groups.items()]


def _one_measurement(first, members, roles, node_key, rekeyed=None):
    """Assemble one property node from every row that reports the same value."""
    review_papers, primary_papers, refs, mixtures, member_keys = [], [], [], [], []
    for m in members:
        member_keys.append(m["Measurement_key"])
        pair = {"key": (rekeyed or {}).get(m.get("Mixture_key"),
                                            m.get("Mixture_key") or ""),
                "name": m.get("Mixture") or ""}
        if pair["key"] and pair not in mixtures:
            mixtures.append(pair)
        containing = m.get("Paper_key") or m.get("Paper_DOI") or ""
        # A review reports someone else's measurement; a research paper reports its
        # own, so it is the primary source rather than a review of nobody.
        target = primary_papers if roles.get(containing) == "primary" else review_papers
        if containing and containing not in target:
            target.append(containing)
        for k in _split(m.get("Source_paper_keys")):
            if k not in primary_papers:
                primary_papers.append(k)
        if m.get("Source_ref_numbers"):
            refs.append(str(m["Source_ref_numbers"]))

    return {
        "key": node_key,
        # The mixture this describes. `subject` is the display name of whichever thing
        # the node hangs off -- a mixture here, a component on the database route --
        # and `origin` says which kind it is.
        "subject": first["Mixture"],
        "mixtures": mixtures,
        "property": first["Property"],
        "value": first["Value"],
        "unit": first.get("Unit"),
        "temperature_C": first.get("Temperature_C"),
        "origin": "table",
        "source_text": "",
        "plausible": bool(first.get("plausible", True)),
        "plausibility_note": first.get("plausibility_note") or "",
        "dedup_key": first.get("Dedup_key") or "",
        "member_keys": member_keys,
        "review_papers": review_papers,
        "source_keys": primary_papers,
        "ref_numbers": ",".join(refs),
    }


def _int(value):
    """CSV ints arrive as floats when the column has any NaN. 1.0 -> 1."""
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _component_rows(components_by_name, canonical=None):
    """components.csv -> the scalar properties that belong on (:Component).

    Kept as its own pass rather than threaded through _mixture_rows and
    _prose_mixture_rows: those build the nested component list for two different
    Cypher statements, and eleven more keys in each is a lot of duplication for
    values that depend only on the component's name.
    """
    canonical = canonical or {}
    merged_into = {}
    for written, target in canonical.items():
        merged_into.setdefault(target, []).append(written)

    # One row per SURVIVING name. Where several spellings collapse, the row whose own
    # name is the canonical one wins; otherwise the first is used, since they describe
    # the same substance and carry the same PubChem data.
    chosen = {}
    for name, c in components_by_name.items():
        if c.get("lookup_status") != "ok":
            continue
        target = _canonical(name, canonical)
        if target not in chosen or name == target:
            chosen[target] = c

    return [{
        "name": name,
        "merged_from": sorted(merged_into.get(name, [])),
        "canonical_source": "review" if merged_into.get(name) else "",
        # Identifiers are written HERE and nowhere else. They used to be set inside the
        # mixture loops, which only table components pass through, so the 69 components
        # that arrive via the applications route never got a CID even though
        # components.csv had one for them.
        "cid": _int(c.get("cid")),
        "smiles": c.get("smiles"),
        "cas": c.get("cas"),
        "formula": c.get("molecular_formula"),
        # How PubChem was actually reached, when it was not by this name -- a hydrate
        # resolved as "FERRIC CHLORIDE hexahydrate". Without it on the node, the only
        # record of a fuzzy match lives in a CSV and the graph looks more certain than
        # it is.
        "matched_name": c.get("matched_name") or "",
        "inchikey": c.get("inchikey"),
        "molecular_weight": c.get("molecular_weight"),
        "h_bond_donor_count": _int(c.get("h_bond_donor_count")),
        "h_bond_acceptor_count": _int(c.get("h_bond_acceptor_count")),
        "tpsa": c.get("tpsa"),
        "rotatable_bond_count": _int(c.get("rotatable_bond_count")),
        "formal_charge": _int(c.get("formal_charge")),
        "xlogp": c.get("xlogp"),
        "complexity": c.get("complexity"),
        "melting_point_C": c.get("melting_point_C"),
        "boiling_point_C": c.get("boiling_point_C"),
        "density_g_cm3": c.get("density_g_cm3"),
        # Computed, gas-phase, single-molecule. Deliberately node attributes rather
        # than property nodes -- see des_pipeline/qm9.py.
        **{f: c.get(f) for f in QM9_FIELDS},
    } for name, c in chosen.items()]


def _component_property_rows(properties, canonical=None):
    """component_properties.csv -> one record per measurement, only the loadable ones.

    The name is canonicalised for the same reason the mixture rows are: the Cypher
    MATCHes an existing Component by name, so a row still carrying an absorbed spelling
    matches nothing and its property node is silently never created. That cost 699
    measurements the first time the merge ran.
    """
    canonical = canonical or {}
    return [{
        # Hash here rather than trusting the CSV's `key`: rows written before this
        # module hashed carry the member key in that column, and a node's key must be
        # the same shape whichever vintage of CSV it was loaded from.
        "key": _hash([r.get("member_key") or r["key"]]),
        "member_keys": [r.get("member_key") or r["key"]],
        "name": _canonical(r["name"], canonical),
        "property": r["property"],
        "value": r["value"],
        "unit": r.get("unit") or "",
        "temperature_C": r.get("temperature_C"),
        "pressure_kPa": r.get("pressure_kPa"),
        "pressure_raw": r.get("pressure_raw") or "",
        "condition_note": r.get("condition_note") or "",
        "qualifier": r.get("qualifier") or "",
        "data_source": r.get("data_source") or "",
        "source_db": r.get("source_db") or "",
        "source_text": r.get("source_text") or "",
        "extractor": r.get("extractor") or "",
    } for r in properties if r.get("status") == "ok"]


def _prose_mixture_key(row):
    """The Mixture key for a prose row, from the same inputs _prose_mixture_rows uses."""
    names = [n.strip() for n in str(row.get("components_resolved") or "").split(";")
             if n.strip()]
    return mixture_key(names, row.get("molar_ratio"))


def _prose_measurement_rows(prose):
    """Prose rows in the same shape as _measurement_rows, so they share the Cypher."""
    return [{
        "key": _hash([r["Measurement_key"]]),
        "subject": r["Mixture"],
        # The prose CSV has no Mixture_key column, so derive it exactly as
        # _prose_mixture_rows does -- otherwise this creates a second, keyless node
        # for a mixture that already exists.
        "mixtures": [{"key": _prose_mixture_key(r), "name": r["Mixture"]}],
        "property": r["property"],
        "value": r["value"],
        "unit": r.get("unit"),
        "temperature_C": r.get("temperature_C"),
        "origin": "prose",
        "plausible": True, "plausibility_note": "", "dedup_key": "",
        "member_keys": [r["Measurement_key"]],
        "source_text": r.get("source_text") or "",   # the sentence it was read from
        "paper_doi": r["Paper_DOI"],
        "source_keys": _split(r.get("Source_paper_keys")),
        "ref_numbers": r.get("Source_ref_numbers") or "",
    } for r in prose]


def _prose_mixture_rows(prose, components_by_name, canonical=None):
    """One record per distinct prose Mixture, components pre-split and null-filtered."""
    from . import xml_utils

    seen, rows = set(), []
    for r in prose:
        mixture = r.get("Mixture")
        if not mixture or mixture in seen:
            continue
        seen.add(mixture)
        names = [_canonical(n.strip(), canonical or {})
                 for n in str(r.get("components_resolved") or "").split(";") if n.strip()]
        ratios = xml_utils.split_ratio(r.get("molar_ratio"), n=len(names) or 1)
        comps = []
        for i, name in enumerate(names):
            extra = components_by_name.get(name, {})
            comps.append({
                "name": name,
                "ratio": ratios[i] if i < len(ratios) else None,
                "role": "HBA" if i == 0 else "HBD",
                "smiles": extra.get("smiles"),
                "cas": extra.get("cas"),
                "cid": int(extra["cid"]) if extra.get("cid") is not None else None,
                "formula": extra.get("molecular_formula"),
            })
        rows.append({
            "member_keys": [r.get("Row_id") or ""],
            "key": mixture_key(names, r.get("molar_ratio")),
            "mixture": mixture,
            "ratio_raw": r.get("molar_ratio") or "",
            "components": comps,
            "paper_doi": r["Paper_DOI"],
            "source_keys": _split(r.get("Source_paper_keys")),
            "ref_numbers": r.get("Source_ref_numbers") or "",
        })
    return rows


# ---------- Cypher ----------
PAPERS_CYPHER = """
UNWIND $papers AS p
MERGE (paper:Paper {key: p.key})
  SET paper.doi = p.doi, paper.authors = p.authors,
      paper.title = p.title, paper.journal = p.journal, paper.volume = p.volume,
      paper.issue = p.issue, paper.pages = p.pages, paper.year = p.year,
      paper.match_score = p.match_score, paper.title_agreement = p.title_agreement,
      paper.raw = p.raw, paper.role = 'primary'
"""

# The papers we extracted FROM. Written after PAPERS_CYPHER so that a paper which is
# both cited by one review and extracted by us keeps its own role and metadata rather
# than the thinner reference-list version.
CORPUS_CYPHER = """
UNWIND $corpus AS p
MERGE (paper:Paper {key: p.key})
  SET paper.doi = p.doi, paper.authors = p.authors, paper.title = p.title,
      paper.journal = p.journal, paper.volume = p.volume, paper.issue = p.issue,
      paper.pages = p.pages, paper.year = p.year, paper.role = p.role,
      paper.extracted = true
"""

MIXTURES_CYPHER = """
UNWIND $rows AS row
MERGE (mix:Mixture {key: row.key})
  SET mix.name = row.mixture, mix.member_keys = row.member_keys,
      mix.ratio_raw = row.ratio_raw,
      mix.ratio_flag = row.ratio_flag, mix.component_flag = row.component_flag,
      mix.n_components = size(row.components), mix.origin = 'table'

FOREACH (c IN row.components |
  MERGE (comp:Component {name: c.name})
    ON CREATE SET comp.origin = 'table'
  MERGE (comp)-[part:PART_OF]->(mix)
    SET part.molar_ratio = c.ratio, part.role = c.role
)

FOREACH (k IN row.review_papers |
  MERGE (review:Paper {key: k})
    ON CREATE SET review.doi = k
  MERGE (mix)-[rep:REPORTED_IN]->(review)
    SET rep.role = 'review'
)

FOREACH (k IN row.source_keys |
  MERGE (src:Paper {key: k})
  MERGE (mix)-[rep:REPORTED_IN]->(src)
    SET rep.role = 'primary', rep.ref_numbers = row.ref_numbers
)
"""

# What a DES was used FOR, shaped exactly like a measurement: one node per reported
# record, carrying its own key, member_keys and source_text and pointing at the papers
# that report it. `(:Application)` stays as the small domain vocabulary so "which
# solvents do esterification" is still one hop from the domain.
#
# It used to be an edge, which meant the per-row detail could not reach a Paper: the
# domain node aggregated every paper across all 118 rows, so "which paper reported THIS
# use" had no answer.
APPLICATIONS_CYPHER = """
UNWIND $rows AS row
MERGE (app:Application {domain: row.domain})
  ON CREATE SET app.example_caption = row.caption

MERGE (mix:Mixture {key: row.mixture_key})
  ON CREATE SET mix.name = row.mixture, mix.ratio_raw = row.ratio_raw,
                mix.origin = 'application'

FOREACH (c IN row.components |
  MERGE (comp:Component {name: c.name})
    ON CREATE SET comp.origin = 'application'
  MERGE (comp)-[part:PART_OF]->(mix)
    SET part.role = c.role,
        // Marks a component the caption implied rather than any column listing it.
        // Keeps asserted chemistry distinguishable from chemistry we read.
        part.inferred = c.inferred
)

MERGE (rec:ApplicationRecord {key: row.key})
  SET rec.domain = row.domain, rec.source_text = row.source_text,
      rec.member_keys = row.member_keys, rec.subject = row.mixture,
      rec.origin = 'application'
MERGE (mix)-[:HAS_APPLICATION]->(rec)
MERGE (rec)-[:IN_DOMAIN]->(app)

FOREACH (k IN row.review_papers |
  MERGE (review:Paper {key: k})
  MERGE (rec)-[r:REPORTED_IN]->(review)
    SET r.role = 'review', r.ref_numbers = row.ref_numbers
)

FOREACH (k IN row.source_keys |
  MERGE (src:Paper {key: k})
  MERGE (rec)-[r:REPORTED_IN]->(src)
    SET r.role = 'primary', r.ref_numbers = row.ref_numbers
)
"""


def _application_rows(applications, roles=None, canonical=None):
    """applications.csv -> the shape APPLICATIONS_CYPHER wants."""
    roles = roles or {}
    rows = []
    for r in applications:
        # A component the caption implied is marked on its PART_OF edge rather than
        # repeated as a string on every record -- 118 identical copies of "PTSA" said
        # nothing about WHICH component was inferred.
        implied = {_norm(n) for n in str(r.get("Implied_components") or "").split(";")
                   if n.strip()}
        comps = []
        for i, role in ((1, "HBA"), (2, "HBD"), (3, "HBD")):
            name = r.get(f"Component_{i}")
            if name and str(name).strip() and str(name) != "nan":
                comps.append({"name": _canonical(str(name).strip(), canonical or {}),
                              "role": role,
                              "inferred": _norm(name) in implied})
        if not comps or not r.get("Mixture"):
            continue
        containing = r.get("Paper_key") or ""
        is_primary = roles.get(containing) == "primary"
        rows.append({
            "key": r["Application_key"],
            "mixture_key": mixture_key([c["name"] for c in comps], r.get("Ratio_raw")),
            "domain": r.get("Domain") or "unspecified",
            "caption": r.get("Table_caption") or "",
            "mixture": r["Mixture"],
            "ratio_raw": r.get("Ratio_raw") or "",
            "components": comps,
            "source_text": r.get("Detail") or "",
            # The key IS "<slug>:<table>:<row>", so paper/table/row are not repeated.
            "member_keys": [r["Application_key"]],
            "ref_numbers": r.get("Source_ref_numbers") or "",
            "review_papers": [] if is_primary else ([containing] if containing else []),
            "source_keys": (_split(r.get("Source_paper_keys")) or
                            ([containing] if is_primary and containing else [])),
        })
    return rows


def _norm(name):
    """Compare component names ignoring case, spacing and a leading coefficient."""
    import re

    return re.sub(r"^\d+", "", re.sub(r"\W+", "", str(name or "")).lower())


# One block per property. The label cannot be parameterised in Cypher without
# APOC, so it is substituted from config.PROPERTY_NAMES — our own constant, never
# user input.
PROSE_MIXTURES_CYPHER = """
UNWIND $rows AS row
MERGE (mix:Mixture {key: row.key})
  ON CREATE SET mix.name = row.mixture, mix.member_keys = row.member_keys,
                mix.ratio_raw = row.ratio_raw,
                mix.n_components = size(row.components), mix.origin = 'prose'

FOREACH (c IN row.components |
  MERGE (comp:Component {name: c.name})
    ON CREATE SET comp.origin = 'prose'
  MERGE (comp)-[part:PART_OF]->(mix)
    SET part.molar_ratio = c.ratio, part.role = c.role
)

MERGE (review:Paper {key: row.paper_doi})
  ON CREATE SET review.doi = row.paper_doi
MERGE (mix)-[rev:REPORTED_IN]->(review)
  SET rev.role = 'review'

FOREACH (k IN row.source_keys |
  MERGE (src:Paper {key: k})
  MERGE (mix)-[rep:REPORTED_IN]->(src)
    SET rep.role = 'primary', rep.ref_numbers = row.ref_numbers
)
"""

COMPONENTS_CYPHER = """
UNWIND $rows AS r
MATCH (c:Component {name: r.name})
  SET c.merged_from = r.merged_from,
      c.canonical_source = r.canonical_source,
      c.cid = r.cid,
      c.smiles = r.smiles,
      c.cas = r.cas,
      c.formula = r.formula,
      c.matched_name = r.matched_name,
      c.inchikey = r.inchikey,
      c.molecular_weight = r.molecular_weight,
      c.h_bond_donor_count = r.h_bond_donor_count,
      c.h_bond_acceptor_count = r.h_bond_acceptor_count,
      c.tpsa = r.tpsa,
      c.rotatable_bond_count = r.rotatable_bond_count,
      c.formal_charge = r.formal_charge,
      c.xlogp = r.xlogp,
      c.complexity = r.complexity,
      c.qm9_id = r.qm9_id, c.qm9_n_matches = r.qm9_n_matches,
      c.qm9_contested = r.qm9_contested,
      c.qm9_dipole_D = r.qm9_dipole_D,
      c.qm9_polarizability_a0_3 = r.qm9_polarizability_a0_3,
      c.qm9_homo_eV = r.qm9_homo_eV, c.qm9_lumo_eV = r.qm9_lumo_eV,
      c.qm9_gap_eV = r.qm9_gap_eV, c.qm9_r2_a0_2 = r.qm9_r2_a0_2,
      c.qm9_zpve_eV = r.qm9_zpve_eV, c.qm9_cv_cal_mol_K = r.qm9_cv_cal_mol_K,
      c.melting_point_C = r.melting_point_C,
      c.boiling_point_C = r.boiling_point_C,
      c.density_g_cm3 = r.density_g_cm3
"""

# One node per reported value, each reaching the source that reported it. Uses the
# same property-as-label idiom and the same `origin` discriminator as the table and
# prose routes, so a new property in config.PROPERTIES needs no new Cypher here.
COMPONENT_PROPERTIES_CYPHER = """
UNWIND $rows AS r
MATCH (c:Component {{name: r.name}})
MERGE (p:{label} {{key: r.key}})
  SET p.value = r.value, p.unit = r.unit, p.temperature_C = r.temperature_C,
      p.pressure_kPa = r.pressure_kPa, p.pressure_raw = r.pressure_raw,
      p.condition_note = r.condition_note, p.qualifier = r.qualifier,
      p.property = r.property, p.subject = r.name,
      p.source_text = r.source_text, p.extractor = r.extractor,
      p.member_keys = r.member_keys, p.origin = 'component'
MERGE (c)-[:HAS_{rel}]->(p)

MERGE (db:Source {{name: r.source_db}})
  ON CREATE SET db.kind = 'database'
MERGE (p)-[:REPORTED_IN]->(db)

FOREACH (_ IN CASE WHEN r.data_source <> '' THEN [1] ELSE [] END |
  MERGE (s:Source {{name: r.data_source}})
    ON CREATE SET s.kind = 'attribution'
  MERGE (p)-[:REPORTED_IN]->(s)
)
"""

MEASUREMENTS_CYPHER = """
UNWIND $rows AS r
MERGE (m:{label} {{key: r.key}})
  SET m.value = r.value, m.unit = r.unit, m.temperature_C = r.temperature_C,
      m.property = r.property, m.subject = r.subject,
      m.origin = r.origin, m.source_text = r.source_text,
      m.plausible = r.plausible, m.plausibility_note = r.plausibility_note,
      m.dedup_key = r.dedup_key, m.member_keys = r.member_keys

// Attached by name, because the mixture rows have already created every mixture and
// two papers can name the same one differently. `temperature_C` is deliberately NOT
// copied onto the relationship -- it describes the measurement, and one writer means
// the two can never disagree.
FOREACH (mx IN r.mixtures |
  MERGE (mix:Mixture {{key: mx.key}})
    ON CREATE SET mix.name = mx.name
  MERGE (mix)-[:HAS_{rel}]->(m)
)

FOREACH (k IN r.review_papers |
  MERGE (review:Paper {{key: k}})
    ON CREATE SET review.doi = k
  MERGE (m)-[rep:REPORTED_IN]->(review)
    SET rep.role = 'review'
)

FOREACH (k IN r.source_keys |
  MERGE (src:Paper {{key: k}})
  MERGE (m)-[rep:REPORTED_IN]->(src)
    SET rep.role = 'primary', rep.ref_numbers = r.ref_numbers
)
"""


def build(wipe=False, include_prose=True):
    from . import store

    table = store.read_all("mixtures")
    long_rows = store.read_all("measurements")
    references = store.read_all("references")

    # Only rows the prose route judged loadable: a real number, present in the
    # section text, not already in Table 2, and every component name resolved.
    prose = []
    if include_prose:
        prose_all = store.read_all("sections_llm")
        if prose_all:
            prose = [r for r in prose_all if r.get("status") == "ok"]
            print(f"  {len(prose)} prose measurement(s) of {len(prose_all)} rows "
                  f"passed the status checks")

    components_by_name = {}
    if config.COMPONENTS_CSV.exists():
        components_by_name = {
            c["name"]: c for c in _read(config.COMPONENTS_CSV).to_dict("records")
        }
        print(f"  joining {len(components_by_name)} enriched components")
    else:
        print("  no components.csv yet — run --steps components to add SMILES/CAS")

    # Every DOI a measurement points at must have a Paper node, or the MERGE below
    # would invent an empty one.
    cited = {k for r in table for k in _split(r.get("Source_paper_keys"))}
    cited |= {k for r in prose for k in _split(r.get("Source_paper_keys"))}
    known = {r["key"] for r in references if r.get("key")}
    missing = cited - known
    assert not missing, (f"{len(missing)} cited paper keys are absent from "
                         f"references.csv: {list(missing)[:3]}")

    component_properties = []
    if config.COMPONENT_PROPERTIES_CSV.exists():
        raw = _read(config.COMPONENT_PROPERTIES_CSV)
        if not raw.empty and "status" in raw.columns:
            component_properties = raw.to_dict("records")
            loadable = sum(1 for r in component_properties if r.get("status") == "ok")
            print(f"  {loadable} component property records of {len(component_properties)} "
                  f"passed the status checks")

    # Which of the papers we extracted from report their own measurements. A research
    # article is the primary source for its data; a review is not.
    roles = {p["Paper_key"]: p.get("role", "review") for p in store.papers()}
    corpus = [{"key": p["Paper_key"], "doi": p.get("Paper_DOI") or "",
               "authors": p.get("authors") or "", "title": p.get("title") or "",
               "journal": p.get("journal") or "", "volume": str(p.get("volume") or ""),
               "issue": str(p.get("issue") or ""), "pages": str(p.get("pages") or ""),
               "year": str(p.get("year") or ""), "role": p.get("role", "review")}
              for p in store.papers()]

    from . import duplicates

    # Only groups a human marked `yes` in data/review/component_duplicates.csv. An
    # empty file means nothing merges, which is the safe default.
    canonical = duplicates.canonical_map()
    if canonical:
        print(f"  merging {len(canonical)} component spelling(s) into "
              f"{len(set(canonical.values()))} canonical name(s)")

    papers = _paper_rows(references)
    applications = _application_rows(store.read_all("applications"), roles, canonical)
    mixtures, rekeyed = _mixture_rows(table, components_by_name, roles, canonical)
    measurements = _measurement_rows(long_rows, roles, rekeyed)
    prose_mixtures = _prose_mixture_rows(prose, components_by_name, canonical)
    prose_measurements = _prose_measurement_rows(prose)
    component_scalars = _component_rows(components_by_name, canonical)
    property_records = _component_property_rows(component_properties, canonical)

    driver = _driver()
    try:
        if wipe:
            driver.execute_query("MATCH (n) DETACH DELETE n", database_=config.NEO4J_DATABASE)
            print("  wiped the database")
        # Constraints from earlier versions of this schema. `--wipe` deletes nodes but
        # not constraints, so these survive forever unless dropped by name -- and
        # `mixture_name` actively blocks the new key-based identity, because two
        # spellings of one solvent must now be allowed to share a node.
        for obsolete in ("paper_doi",          # Paper identity moved doi -> key
                         "mixture_name",       # Mixture identity moved name -> key
                         "meas_id",            # :Measurement and :Property are labels
                         "property_name"):     # from a design that no longer exists
            driver.execute_query(f"DROP CONSTRAINT {obsolete} IF EXISTS",
                                 database_=config.NEO4J_DATABASE)
        for statement in CONSTRAINTS:
            driver.execute_query(statement, database_=config.NEO4J_DATABASE)

        driver.execute_query(PAPERS_CYPHER, papers=papers, database_=config.NEO4J_DATABASE)
        driver.execute_query(CORPUS_CYPHER, corpus=corpus, database_=config.NEO4J_DATABASE)
        extracted_reviews = sum(1 for p in corpus if p["role"] == "review")
        print(f"  papers: {len(papers)} cited + {len(corpus)} extracted "
              f"({extracted_reviews} review, {len(corpus) - extracted_reviews} primary)")

        driver.execute_query(MIXTURES_CYPHER, rows=mixtures, database_=config.NEO4J_DATABASE)
        print(f"  mixtures: {len(mixtures)} rows")

        if applications:
            driver.execute_query(APPLICATIONS_CYPHER, rows=applications,
                                 database_=config.NEO4J_DATABASE)
            domains = sorted({a["domain"] for a in applications})
            print(f"  applications: {len(applications)} rows across "
                  f"{len(domains)} domain(s) -- {', '.join(domains)}")

        if prose_mixtures:
            # Separate statement using ON CREATE SET, so a prose row can never
            # overwrite the identity of a mixture Table 2 already created.
            driver.execute_query(PROSE_MIXTURES_CYPHER, rows=prose_mixtures,
                                 database_=config.NEO4J_DATABASE)
            print(f"  prose mixtures: {len(prose_mixtures)}")

        for prop in PROPERTY_NAMES:
            rows = [m for m in measurements if m["property"] == prop]
            extra = [m for m in prose_measurements if m["property"] == prop]
            if not rows and not extra:
                continue
            driver.execute_query(
                MEASUREMENTS_CYPHER.format(label=prop, rel=prop.upper()),
                rows=rows + extra,
                database_=config.NEO4J_DATABASE,
            )
            suffix = f"  (+{len(extra)} prose)" if extra else ""
            print(f"  {prop:<18} {len(rows):>5}{suffix}")

        # Component data last: MATCH, not MERGE, so the 133 names that never resolved
        # cannot become orphan Component nodes.
        if component_scalars:
            driver.execute_query(COMPONENTS_CYPHER, rows=component_scalars,
                                 database_=config.NEO4J_DATABASE)
            print(f"  component scalars: {len(component_scalars)}")

        for prop in PROPERTY_NAMES:
            rows = [r for r in property_records if r["property"] == prop]
            if not rows:
                continue
            driver.execute_query(
                COMPONENT_PROPERTIES_CYPHER.format(label=prop, rel=prop.upper()),
                rows=rows, database_=config.NEO4J_DATABASE)
            print(f"  component {prop:<18} {len(rows):>5}")

        _report(driver)
    finally:
        driver.close()


def _report(driver):
    def query(cypher):
        records, _, _ = driver.execute_query(cypher, database_=config.NEO4J_DATABASE)
        return records

    print("\n  nodes:")
    for r in query("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY n DESC"):
        print(f"    {r['label']:<18} {r['n']:>6}")

    print("  relationships:")
    for r in query("MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS n ORDER BY n DESC"):
        print(f"    {r['type']:<18} {r['n']:>6}")

    print("  by origin:")
    for label, cypher in (
        ("measurements", "MATCH (n) WHERE n.origin IS NOT NULL AND NOT n:Mixture "
                         "AND NOT n:Component RETURN n.origin AS origin, count(*) AS n"),
        ("mixtures", "MATCH (n:Mixture) RETURN n.origin AS origin, count(*) AS n"),
        ("components", "MATCH (n:Component) RETURN n.origin AS origin, count(*) AS n"),
    ):
        counts = {r["origin"]: r["n"] for r in query(cypher)}
        summary = ", ".join(f"{k or 'unset'} {v}" for k, v in sorted(counts.items(),
                                                                    key=lambda kv: -kv[1]))
        print(f"    {label:<14} {summary}")

    sources = query("MATCH (s:Source)<-[:REPORTED_IN]-() "
                    "RETURN s.name AS name, s.kind AS kind, count(*) AS n "
                    "ORDER BY n DESC LIMIT 6")
    if sources:
        print("  top property sources:")
        for r in sources:
            print(f"    {r['name'][:44]:<46}{r['kind']:<14}{r['n']:>6}")

    no_doi = query("MATCH (p:Paper) WHERE p.doi IS NULL RETURN count(p) AS n")[0]["n"]
    print(f"    papers without a DOI: {no_doi} (Crossref found no match; "
          f"they still carry match_score and raw)")
