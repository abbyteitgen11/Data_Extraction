"""
Load the CSVs into Neo4j.

Graph shape (each property gets its own node label, so you can query
``MATCH (m:Mixture)-[:HAS_DENSITY]->(d)`` directly):

    (:Component {name, smiles, cas, cid, formula})
        -[:PART_OF {molar_ratio, role}]-> (:Mixture {name, ratio_raw, ...})

    (:Mixture) -[:HAS_DENSITY {temperature_C}]-> (:Density {key, value, unit, temperature_C})
        ... and HAS_MELTING_POINT, HAS_VISCOSITY, HAS_CONDUCTIVITY,
            HAS_SURFACE_TENSION, HAS_REFRACTIVE_INDEX

    (:Density)  -[:REPORTED_IN {ref_numbers}]-> (:Paper {role:'primary'})   the original source
    (:Density)  -[:REVIEW_PAPER]-------------> (:Paper {role:'review'})     the review

    (:Mixture)  -[:REPORTED_IN]--------------> (:Paper {role:'primary'})
    (:Mixture)  -[:REVIEW_PAPER]-------------> (:Paper {role:'review'})

Papers are attached at the mixture level as well as the measurement level because
862 of the 1535 table rows report no numeric value at all — without the mixture
edges those DES would have no provenance in the graph.

Loads are idempotent: every node has a deterministic key, so re-running updates
rather than duplicates. ``--wipe`` is therefore optional.
"""
import pandas as pd

from . import config
from .config import PROPERTY_NAMES

CONSTRAINTS = [
    "CREATE CONSTRAINT component_name IF NOT EXISTS "
    "FOR (c:Component) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT mixture_name IF NOT EXISTS "
    "FOR (m:Mixture) REQUIRE m.name IS UNIQUE",
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
            "ref_number": r.get("ref_number"),
            "authors": r.get("authors") or "",
            "title": r.get("title") or "",
            "journal": r.get("journal") or "",
            "volume": _str(r.get("volume")),
            "issue": _str(r.get("issue")),
            "pages": _str(r.get("pages")),
            "year": _str(r.get("year")),
        })
    return papers


def _mixture_rows(table, components_by_name, roles=None):
    """One record per mixture, with components pre-filtered into a nested list.

    Doing the null-filtering in Python (rather than three FOREACH blocks in Cypher)
    is what stops a null Component_3 from becoming a `Component {name: null}` node.
    """
    rows = []
    for r in table:
        comps = []
        for i, role in ((1, "HBA"), (2, "HBD"), (3, "HBD")):
            name = r.get(f"Component_{i}")
            if not name:
                continue
            extra = components_by_name.get(name, {})
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
        rows.append({
            "row_id": r["Row_id"],
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
    return rows


def _measurement_rows(long_rows, roles=None):
    """One row per DISTINCT datum, not per report of it. -> list[dict].

    Two reviews tabulating the same primary measurement are reporting one fact, and
    the graph should hold one node for it with an edge to each paper -- otherwise
    counting measurements counts publications. `Dedup_key` already identifies them:
    same component set, ratio, property, value, temperature and originating study.

    Merging is refused when the values disagree. Two papers printing different numbers
    for what should be the same datum is a finding, not noise, so both nodes stay and
    `validate.duplicate_report` puts the pair in front of a human.
    """
    roles = roles or {}
    groups = {}
    for r in long_rows:
        # Rows with no dedup key are their own group, keyed uniquely so they never merge.
        key = r.get("Dedup_key") or f"__{r['Measurement_key']}"
        groups.setdefault(key, []).append(r)

    rows = []
    for dedup_key, members in groups.items():
        values = {round(float(m["Value"]), 6) for m in members if m.get("Value") is not None}
        if len(members) > 1 and len(values) > 1:
            # A real disagreement: keep every report separate and let review decide.
            for m in members:
                rows.append(_one_measurement(m, [m], roles, m["Measurement_key"]))
            continue
        node_key = (f"dedup:{dedup_key}" if len(members) > 1
                    else members[0]["Measurement_key"])
        rows.append(_one_measurement(members[0], members, roles, node_key))
    return rows


def _one_measurement(first, members, roles, node_key):
    """Assemble one property node from every row that reports the same value."""
    review_papers, primary_papers, refs, mixtures, member_keys = [], [], [], [], []
    for m in members:
        member_keys.append(m["Measurement_key"])
        if m.get("Mixture") and m["Mixture"] not in mixtures:
            mixtures.append(m["Mixture"])
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
        "mixture": first["Mixture"],
        "mixtures": mixtures,
        "property": first["Property"],
        "value": first["Value"],
        "unit": first.get("Unit"),
        "temperature_C": first.get("Temperature_C"),
        "locus": first.get("Locus") or "",
        "origin": "table",
        "evidence": "",
        "plausible": bool(first.get("plausible", True)),
        "plausibility_note": first.get("plausibility_note") or "",
        "dedup_key": first.get("Dedup_key") or "",
        "member_keys": member_keys,
        "n_reports": len(members),
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


def _component_rows(components_by_name):
    """components.csv -> the scalar properties that belong on (:Component).

    Kept as its own pass rather than threaded through _mixture_rows and
    _prose_mixture_rows: those build the nested component list for two different
    Cypher statements, and eleven more keys in each is a lot of duplication for
    values that depend only on the component's name.
    """
    return [{
        "name": name,
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
    } for name, c in components_by_name.items() if c.get("lookup_status") == "ok"]


def _component_property_rows(properties):
    """component_properties.csv -> one record per measurement, only the loadable ones."""
    return [{
        "key": r["key"],
        "name": r["name"],
        "property": r["property"],
        "value": r["value"],
        "unit": r.get("unit") or "",
        "temperature_C": r.get("temperature_C"),
        "pressure": r.get("pressure") or "",
        "qualifier": r.get("qualifier") or "",
        "data_source": r.get("data_source") or "",
        "source_db": r.get("source_db") or "",
        "raw_string": r.get("raw_string") or "",
        "extractor": r.get("extractor") or "",
    } for r in properties if r.get("status") == "ok"]


def _prose_measurement_rows(prose):
    """Prose rows in the same shape as _measurement_rows, so they share the Cypher."""
    return [{
        "key": r["Measurement_key"],
        "mixture": r["Mixture"],
        "property": r["property"],
        "value": r["value"],
        "unit": r.get("unit"),
        "temperature_C": r.get("temperature_C"),
        "locus": f"{r.get('section_id') or ''} {r.get('section_title') or ''}".strip(),
        "origin": "prose",
        "plausible": True, "plausibility_note": "", "dedup_key": "",
        "evidence": r.get("source_text") or "",     # the sentence it was read from
        "paper_doi": r["Paper_DOI"],
        "source_keys": _split(r.get("Source_paper_keys")),
        "ref_numbers": r.get("Source_ref_numbers") or "",
    } for r in prose]


def _prose_mixture_rows(prose, components_by_name):
    """One record per distinct prose Mixture, components pre-split and null-filtered."""
    from . import xml_utils

    seen, rows = set(), []
    for r in prose:
        mixture = r.get("Mixture")
        if not mixture or mixture in seen:
            continue
        seen.add(mixture)
        names = [n.strip() for n in str(r.get("components_resolved") or "").split(";") if n.strip()]
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
            "row_id": r.get("Row_id") or "",
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
  SET paper.doi = p.doi, paper.ref_number = p.ref_number, paper.authors = p.authors,
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
MERGE (mix:Mixture {name: row.mixture})
  SET mix.row_id = row.row_id, mix.ratio_raw = row.ratio_raw,
      mix.ratio_flag = row.ratio_flag, mix.component_flag = row.component_flag,
      mix.n_components = size(row.components), mix.origin = 'table'

FOREACH (c IN row.components |
  MERGE (comp:Component {name: c.name})
    ON CREATE SET comp.origin = 'table'
    SET comp.smiles  = coalesce(c.smiles, comp.smiles),
        comp.cas     = coalesce(c.cas, comp.cas),
        comp.cid     = coalesce(c.cid, comp.cid),
        comp.formula = coalesce(c.formula, comp.formula)
  MERGE (comp)-[part:PART_OF]->(mix)
    SET part.molar_ratio = c.ratio, part.role = c.role
)

FOREACH (k IN row.review_papers |
  MERGE (review:Paper {key: k})
    ON CREATE SET review.doi = k
  MERGE (mix)-[:REVIEW_PAPER]->(review)
)

FOREACH (k IN row.source_keys |
  MERGE (src:Paper {key: k})
  MERGE (mix)-[rep:REPORTED_IN]->(src)
    SET rep.ref_numbers = row.ref_numbers
)
"""

# What a DES was used FOR. The domain is a node so "which solvents do esterification"
# is one hop; the row's own columns -- alcohol, acid, biomass source, yield -- ride on
# the relationship, because they differ from table to table and would otherwise force
# an Application node per row with nothing shared between them.
APPLICATIONS_CYPHER = """
UNWIND $rows AS row
MERGE (app:Application {domain: row.domain})
  ON CREATE SET app.example_caption = row.caption

MERGE (mix:Mixture {name: row.mixture})
  ON CREATE SET mix.ratio_raw = row.ratio_raw, mix.origin = 'application'

FOREACH (c IN row.components |
  MERGE (comp:Component {name: c.name})
    ON CREATE SET comp.origin = 'application'
  MERGE (comp)-[part:PART_OF]->(mix)
    SET part.role = c.role
)

MERGE (mix)-[used:USED_IN {key: row.key}]->(app)
  SET used.detail = row.detail, used.paper_key = row.paper_key,
      used.table_id = row.table_id, used.source_row = row.source_row,
      used.implied_components = row.implied_components

FOREACH (k IN row.review_papers |
  MERGE (review:Paper {key: k})
  MERGE (app)-[:REVIEW_PAPER]->(review)
)

FOREACH (k IN row.source_keys |
  MERGE (src:Paper {key: k})
  MERGE (app)-[:REPORTED_IN]->(src)
)
"""


def _application_rows(applications, roles=None):
    """applications.csv -> the shape APPLICATIONS_CYPHER wants."""
    roles = roles or {}
    rows = []
    for r in applications:
        comps = []
        for i, role in ((1, "HBA"), (2, "HBD"), (3, "HBD")):
            name = r.get(f"Component_{i}")
            if name and str(name).strip() and str(name) != "nan":
                comps.append({"name": str(name).strip(), "role": role})
        if not comps or not r.get("Mixture"):
            continue
        containing = r.get("Paper_key") or ""
        is_primary = roles.get(containing) == "primary"
        rows.append({
            "key": r["Application_key"],
            "domain": r.get("Domain") or "unspecified",
            "caption": r.get("Table_caption") or "",
            "mixture": r["Mixture"],
            "ratio_raw": r.get("Ratio_raw") or "",
            "components": comps,
            "detail": r.get("Detail") or "",
            "implied_components": r.get("Implied_components") or "",
            "paper_key": containing,
            "table_id": r.get("Table_id") or "",
            "source_row": _int(r.get("Source_row")),
            "review_papers": [] if is_primary else ([containing] if containing else []),
            "source_keys": (_split(r.get("Source_paper_keys")) or
                            ([containing] if is_primary and containing else [])),
        })
    return rows


# One block per property. The label cannot be parameterised in Cypher without
# APOC, so it is substituted from config.PROPERTY_NAMES — our own constant, never
# user input.
PROSE_MIXTURES_CYPHER = """
UNWIND $rows AS row
MERGE (mix:Mixture {name: row.mixture})
  ON CREATE SET mix.row_id = row.row_id, mix.ratio_raw = row.ratio_raw,
                mix.n_components = size(row.components), mix.origin = 'prose'

FOREACH (c IN row.components |
  MERGE (comp:Component {name: c.name})
    ON CREATE SET comp.origin = 'prose'
  SET comp.smiles  = coalesce(c.smiles, comp.smiles),
      comp.cas     = coalesce(c.cas, comp.cas),
      comp.cid     = coalesce(c.cid, comp.cid),
      comp.formula = coalesce(c.formula, comp.formula)
  MERGE (comp)-[part:PART_OF]->(mix)
    SET part.molar_ratio = c.ratio, part.role = c.role
)

MERGE (review:Paper {key: row.paper_doi})
  ON CREATE SET review.doi = row.paper_doi
MERGE (mix)-[:REVIEW_PAPER]->(review)

FOREACH (k IN row.source_keys |
  MERGE (src:Paper {key: k})
  MERGE (mix)-[rep:REPORTED_IN]->(src)
    SET rep.ref_numbers = row.ref_numbers
)
"""

COMPONENTS_CYPHER = """
UNWIND $rows AS r
MATCH (c:Component {name: r.name})
  SET c.matched_name = r.matched_name,
      c.inchikey = r.inchikey,
      c.molecular_weight = r.molecular_weight,
      c.h_bond_donor_count = r.h_bond_donor_count,
      c.h_bond_acceptor_count = r.h_bond_acceptor_count,
      c.tpsa = r.tpsa,
      c.rotatable_bond_count = r.rotatable_bond_count,
      c.formal_charge = r.formal_charge,
      c.xlogp = r.xlogp,
      c.complexity = r.complexity,
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
      p.pressure = r.pressure, p.qualifier = r.qualifier,
      p.property = r.property, p.component = r.name,
      p.raw_string = r.raw_string, p.extractor = r.extractor,
      p.origin = 'component'
MERGE (c)-[has:HAS_{rel}]->(p)
  SET has.data_source = r.data_source

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
      m.property = r.property, m.mixture = r.mixture, m.locus = r.locus,
      m.origin = r.origin, m.evidence = r.evidence,
      m.plausible = r.plausible, m.plausibility_note = r.plausibility_note,
      m.dedup_key = r.dedup_key,
      m.member_keys = r.member_keys, m.n_reports = r.n_reports

// A merged datum attaches to every naming of the mixture, since two papers can order
// the components differently while meaning the same solvent.
FOREACH (name IN r.mixtures |
  MERGE (mix:Mixture {{name: name}})
  MERGE (mix)-[has:HAS_{rel}]->(m)
    SET has.temperature_C = r.temperature_C
)

FOREACH (k IN r.review_papers |
  MERGE (review:Paper {{key: k}})
    ON CREATE SET review.doi = k
  MERGE (m)-[:REVIEW_PAPER]->(review)
)

FOREACH (k IN r.source_keys |
  MERGE (src:Paper {{key: k}})
  MERGE (m)-[rep:REPORTED_IN]->(src)
    SET rep.ref_numbers = r.ref_numbers
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

    papers = _paper_rows(references)
    applications = _application_rows(store.read_all("applications"), roles)
    mixtures = _mixture_rows(table, components_by_name, roles)
    measurements = _measurement_rows(long_rows, roles)
    prose_mixtures = _prose_mixture_rows(prose, components_by_name)
    prose_measurements = _prose_measurement_rows(prose)
    component_scalars = _component_rows(components_by_name)
    property_records = _component_property_rows(component_properties)

    driver = _driver()
    try:
        if wipe:
            driver.execute_query("MATCH (n) DETACH DELETE n", database_=config.NEO4J_DATABASE)
            print("  wiped the database")
        # Paper identity moved from doi to key; drop the old constraint or the new
        # one cannot be created on a database built by an earlier version.
        driver.execute_query("DROP CONSTRAINT paper_doi IF EXISTS",
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
