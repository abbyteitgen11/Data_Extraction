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
    "CREATE CONSTRAINT paper_doi IF NOT EXISTS "
    "FOR (p:Paper) REQUIRE p.doi IS UNIQUE",
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
    """One record per unique primary DOI. Two references can share a DOI."""
    papers = []
    seen = set()
    for r in references:
        doi = r.get("doi")
        if not doi or doi in seen:
            continue
        seen.add(doi)
        papers.append({
            "doi": doi,
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


def _mixture_rows(table, components_by_name):
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
            "review_doi": r.get("Review_DOI") or config.REVIEW_DOI,
            "source_dois": _split(r.get("Source_DOIs")),
            "ref_numbers": r.get("Source_ref_numbers") or "",
        })
    return rows


def _measurement_rows(long_rows):
    return [{
        "key": r["Measurement_key"],
        "mixture": r["Mixture"],
        "property": r["Property"],
        "value": r["Value"],
        "unit": r.get("Unit"),
        "temperature_C": r.get("Temperature_C"),
        "locus": r.get("Locus") or "",
        "review_doi": r.get("Review_DOI") or config.REVIEW_DOI,
        "source_dois": _split(r.get("Source_DOIs")),
        "ref_numbers": r.get("Source_ref_numbers") or "",
    } for r in long_rows]


# ---------- Cypher ----------
PAPERS_CYPHER = """
UNWIND $papers AS p
MERGE (paper:Paper {doi: p.doi})
  SET paper.ref_number = p.ref_number, paper.authors = p.authors,
      paper.title = p.title, paper.journal = p.journal, paper.volume = p.volume,
      paper.issue = p.issue, paper.pages = p.pages, paper.year = p.year,
      paper.role = 'primary'
"""

REVIEW_CYPHER = """
MERGE (paper:Paper {doi: $review.doi})
  SET paper.authors = $review.authors, paper.title = $review.title,
      paper.journal = $review.journal, paper.volume = $review.volume,
      paper.issue = $review.issue, paper.year = $review.year,
      paper.role = 'review'
"""

MIXTURES_CYPHER = """
UNWIND $rows AS row
MERGE (mix:Mixture {name: row.mixture})
  SET mix.row_id = row.row_id, mix.ratio_raw = row.ratio_raw,
      mix.ratio_flag = row.ratio_flag, mix.component_flag = row.component_flag,
      mix.n_components = size(row.components)

FOREACH (c IN row.components |
  MERGE (comp:Component {name: c.name})
    SET comp.smiles  = coalesce(c.smiles, comp.smiles),
        comp.cas     = coalesce(c.cas, comp.cas),
        comp.cid     = coalesce(c.cid, comp.cid),
        comp.formula = coalesce(c.formula, comp.formula)
  MERGE (comp)-[part:PART_OF]->(mix)
    SET part.molar_ratio = c.ratio, part.role = c.role
)

MERGE (review:Paper {doi: row.review_doi})
MERGE (mix)-[:REVIEW_PAPER]->(review)

FOREACH (d IN row.source_dois |
  MERGE (src:Paper {doi: d})
  MERGE (mix)-[rep:REPORTED_IN]->(src)
    SET rep.ref_numbers = row.ref_numbers
)
"""

# One block per property. The label cannot be parameterised in Cypher without
# APOC, so it is substituted from config.PROPERTY_NAMES — our own constant, never
# user input.
MEASUREMENTS_CYPHER = """
UNWIND $rows AS r
MATCH (mix:Mixture {{name: r.mixture}})
MERGE (m:{label} {{key: r.key}})
  SET m.value = r.value, m.unit = r.unit, m.temperature_C = r.temperature_C,
      m.property = r.property, m.mixture = r.mixture, m.locus = r.locus
MERGE (mix)-[has:HAS_{rel}]->(m)
  SET has.temperature_C = r.temperature_C

MERGE (review:Paper {{doi: r.review_doi}})
MERGE (m)-[:REVIEW_PAPER]->(review)

FOREACH (d IN r.source_dois |
  MERGE (src:Paper {{doi: d}})
  MERGE (m)-[rep:REPORTED_IN]->(src)
    SET rep.ref_numbers = r.ref_numbers
)
"""


def build(wipe=False, include_llm=False):
    table = _read(config.TABLE_CSV).to_dict("records")
    long_rows = _read(config.LONG_CSV).to_dict("records")
    references = _read(config.REFERENCES_CSV).to_dict("records")

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
    cited = {d for r in table for d in _split(r.get("Source_DOIs"))}
    known = {r["doi"] for r in references if r.get("doi")}
    missing = cited - known
    assert not missing, f"{len(missing)} cited DOIs are absent from references.csv: {list(missing)[:3]}"

    review_row = table[0] if table else {}
    review = {
        "doi": review_row.get("Review_DOI") or config.REVIEW_DOI,
        "authors": review_row.get("Review_authors") or "",
        "title": review_row.get("Review_title") or "",
        "journal": review_row.get("Review_journal") or "",
        "volume": _str(review_row.get("Review_volume")),
        "issue": _str(review_row.get("Review_issue")),
        "year": _str(review_row.get("Review_year")),
    }

    papers = _paper_rows(references)
    mixtures = _mixture_rows(table, components_by_name)
    measurements = _measurement_rows(long_rows)

    driver = _driver()
    try:
        if wipe:
            driver.execute_query("MATCH (n) DETACH DELETE n", database_=config.NEO4J_DATABASE)
            print("  wiped the database")
        for statement in CONSTRAINTS:
            driver.execute_query(statement, database_=config.NEO4J_DATABASE)

        driver.execute_query(PAPERS_CYPHER, papers=papers, database_=config.NEO4J_DATABASE)
        driver.execute_query(REVIEW_CYPHER, review=review, database_=config.NEO4J_DATABASE)
        print(f"  papers: {len(papers)} primary + 1 review")

        driver.execute_query(MIXTURES_CYPHER, rows=mixtures, database_=config.NEO4J_DATABASE)
        print(f"  mixtures: {len(mixtures)} rows")

        for prop in PROPERTY_NAMES:
            rows = [m for m in measurements if m["property"] == prop]
            if not rows:
                continue
            driver.execute_query(
                MEASUREMENTS_CYPHER.format(label=prop, rel=prop.upper()),
                rows=rows,
                database_=config.NEO4J_DATABASE,
            )
            print(f"  {prop:<18} {len(rows):>5}")

        if include_llm and config.SECTIONS_LLM_CSV.exists():
            _load_llm(driver)

        _report(driver)
    finally:
        driver.close()


def _load_llm(driver):
    """Optional: load the prose measurements, tagged so they stay distinguishable."""
    rows = _read(config.SECTIONS_LLM_CSV)
    if rows.empty or "verified" not in rows.columns:
        print("  no prose measurements to load")
        return
    rows = rows[rows["verified"] == True].to_dict("records")   # noqa: E712 - pandas mask
    if not rows:
        print("  no verified prose measurements to load")
        return
    for prop in PROPERTY_NAMES:
        subset = [{
            "key": f"LLM:{r['section_id']}:{prop}:{r['value']}",
            "value": r["value"], "unit": r.get("unit"),
            "temperature_C": r.get("temperature_C"),
            "components": r.get("components") or "",
            "section": r.get("section_title") or "",
            "review_doi": r.get("review_doi") or config.REVIEW_DOI,
        } for r in rows if r["property"] == prop]
        if not subset:
            continue
        driver.execute_query(f"""
            UNWIND $rows AS r
            MERGE (m:{prop} {{key: r.key}})
              SET m.value = r.value, m.unit = r.unit, m.temperature_C = r.temperature_C,
                  m.property = '{prop}', m.source = 'prose', m.needs_review = true,
                  m.components_text = r.components, m.section = r.section
            MERGE (review:Paper {{doi: r.review_doi}})
            MERGE (m)-[:REVIEW_PAPER]->(review)
        """, rows=subset, database_=config.NEO4J_DATABASE)
    print(f"  loaded {len(rows)} prose measurements (flagged needs_review)")


def _report(driver):
    print("\n  nodes:")
    records, _, _ = driver.execute_query(
        "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY n DESC",
        database_=config.NEO4J_DATABASE,
    )
    for r in records:
        print(f"    {r['label']:<18} {r['n']:>6}")
    print("  relationships:")
    records, _, _ = driver.execute_query(
        "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS n ORDER BY n DESC",
        database_=config.NEO4J_DATABASE,
    )
    for r in records:
        print(f"    {r['type']:<18} {r['n']:>6}")
