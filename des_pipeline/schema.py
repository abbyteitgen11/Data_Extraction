"""
Pydantic models — the single source of truth for every CSV the pipeline writes.

Field order here *is* column order in the CSV, and every field is a scalar, so
``model_dump()`` -> DataFrame -> ``to_csv`` needs no flattening step. Anything
naturally list-shaped (authors, reference numbers, DOIs) is stored pre-joined:

    within one paper   authors are joined with "; "
    across papers      Source_* columns are joined with config.SOURCE_SEP ("|")

The Source_* columns are positionally aligned: the n-th DOI in Source_DOIs
belongs to the n-th title in Source_titles, and so on.
"""
from typing import Literal, Optional, get_args

from pydantic import BaseModel, Field

from .config import PROPERTY_NAMES

PropertyName = Literal[
    "Melting_point", "Density", "Viscosity",
    "Conductivity", "Surface_tension", "Refractive_index",
]

# Guard against the two lists drifting apart — this is the bug that hid all 332
# melting points from the graph for months.
assert set(get_args(PropertyName)) == set(PROPERTY_NAMES), (
    "schema.PropertyName and config.PROPERTIES disagree"
)


class MixtureRow(BaseModel):
    """One row of Table 2 = one DES mixture. -> data/table2_with_dois.csv"""

    Row_id: str                          # "T2-0001"; stable key, survives name collisions

    # --- identity (unchanged from the original table2_full.csv) ---
    Component_1: Optional[str] = None
    Component_1_SMILES: Optional[str] = None
    Component_2: Optional[str] = None
    Component_2_SMILES: Optional[str] = None
    Component_3: Optional[str] = None
    Component_3_SMILES: Optional[str] = None
    Ratio_component_1: Optional[float] = None
    Ratio_component_2: Optional[float] = None
    Ratio_component_3: Optional[float] = None
    Ratio_raw: str = ""
    Mixture: str = ""
    Ratio_flag: str = ""
    Component_flag: str = ""
    DOI: str = ""                        # the review; kept for backwards compatibility
    Ref: str = ""                        # the raw citation cell, e.g. "1,26-28"

    # --- the six properties, as value/unit/temperature triples ---
    Melting_point: Optional[float] = None
    Units_melting_point: Optional[str] = None
    Temperature_melting_point: Optional[float] = None
    Density: Optional[float] = None
    Units_density: Optional[str] = None
    Temperature_density: Optional[float] = None
    Viscosity: Optional[float] = None
    Units_viscosity: Optional[str] = None
    Temperature_viscosity: Optional[float] = None
    Conductivity: Optional[float] = None
    Units_conductivity: Optional[str] = None
    Temperature_conductivity: Optional[float] = None
    Surface_tension: Optional[float] = None
    Units_surface_tension: Optional[str] = None
    Temperature_surface_tension: Optional[float] = None
    Refractive_index: Optional[float] = None
    Units_refractive_index: Optional[str] = None
    Temperature_refractive_index: Optional[float] = None

    # --- the review paper that contains the table ---
    Review_DOI: str = ""
    Review_authors: str = ""
    Review_title: str = ""
    Review_journal: str = ""
    Review_volume: str = ""
    Review_issue: str = ""
    Review_year: str = ""

    # --- the primary papers the data actually came from ("|"-joined) ---
    Source_ref_numbers: str = ""
    Source_DOIs: str = ""
    Source_authors: str = ""
    Source_titles: str = ""
    Source_journals: str = ""
    Source_volumes: str = ""
    Source_issues: str = ""
    Source_pages: str = ""
    Source_years: str = ""


class MeasurementRow(BaseModel):
    """One property value. Derived from MixtureRow. -> data/measurements_long.csv

    This is what build_graph.py loads, and the natural shape for ML training later.
    """

    Measurement_key: str                 # "T2-0007:Density" — unique, makes loads idempotent
    Row_id: str
    Mixture: str
    Property: PropertyName
    Value: float
    Unit: Optional[str] = None
    Temperature_C: Optional[float] = None
    Source: str = "Table 2"              # provenance: where in the paper
    Locus: str = ""                      # provenance: "row 7"
    Source_ref_numbers: str = ""
    Source_DOIs: str = ""
    Review_DOI: str = ""


class ReferenceRow(BaseModel):
    """One bibliography entry. -> data/references.csv"""

    ref_number: int
    authors: str = ""
    title: str = ""
    journal: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""
    year: str = ""
    doi: Optional[str] = None
    match_score: Optional[float] = None
    metadata_source: str = "xml"         # "crossref" once the enrich pass has run
    title_agreement: Optional[float] = None   # XML vs Crossref word overlap; low = suspect
    raw: str = ""


class FigureRow(BaseModel):
    """One figure, flagged for manual digitisation. -> data/figures.csv"""

    figure_id: str
    label: str = ""
    caption: str = ""
    cited_ref_numbers: str = ""          # refs named in the caption, e.g. "6,12"
    image_link: str = ""
    review_doi: str = ""
    status: str = "needs_human"
    tool: str = "webplotdigitizer"
    extracted_csv: str = ""              # fill this in once you have digitised the plot


class TableRow(BaseModel):
    """A table the pipeline has no parser for. -> data/tables_unhandled.csv"""

    table_id: str
    label: str = ""
    caption: str = ""
    n_rows: int = 0
    status: str = "needs_human"


class ComponentRow(BaseModel):
    """External data for one DES component. -> data/components.csv"""

    name: str                            # exactly as written in the paper
    cid: Optional[int] = None
    cas: Optional[str] = None
    smiles: Optional[str] = None
    inchi: Optional[str] = None
    inchikey: Optional[str] = None
    molecular_formula: Optional[str] = None
    molecular_weight: Optional[float] = None
    h_bond_donor_count: Optional[int] = None
    h_bond_acceptor_count: Optional[int] = None
    melting_point_C: Optional[float] = None
    boiling_point_C: Optional[float] = None
    density_g_cm3: Optional[float] = None
    property_comments: str = ""          # the raw PubChem strings, which are messy
    sources: str = ""                    # "pubchem;nist"
    lookup_status: str = ""              # ok | not_found | error


class LLMMeasurement(BaseModel):
    """One measurement claimed in prose. -> data/sections_llm.csv

    Only rows with status == "ok" are loaded into the graph. See STATUS_ORDER below.
    """

    Row_id: str = ""                     # "P-0001"; for humans, not a graph key
    Measurement_key: str = ""            # content-derived, so re-loads are idempotent

    # --- identity ---
    components: str = ""                 # ";"-joined, exactly as the model wrote them
    components_resolved: str = ""        # ";"-joined canonical Table-2 names
    unresolved_components: str = ""      # the ones we refused to guess at
    component_status: str = ""           # resolved | partial | unresolved
    molar_ratio: Optional[str] = None
    Mixture: str = ""                    # "A:B (1:2)" — same convention as Table 2

    # --- the measurement ---
    property: PropertyName
    value: Optional[float] = None        # None when the text only ranks or compares
    unit: Optional[str] = None           # canonicalised to the Table 2 spelling
    unit_raw: Optional[str] = None       # exactly as the model wrote it
    temperature_C: Optional[float] = None

    # --- provenance ---
    source_text: str = ""                # the model's quote
    quote_found: bool = False            # ...and whether it is really in the section
    section_id: str = ""
    section_title: str = ""
    Source_ref_numbers: str = ""         # harvested from the citation, e.g. "113"
    Source_DOIs: str = ""                # "|"-joined, same convention as the table route
    ref_source: str = ""                 # agreed | text | llm | none
    Review_DOI: str = ""

    # --- verdicts ---
    verified: bool = False               # does `value` occur in the real section text?
    duplicate_of: str = ""               # a table Measurement_key, e.g. "T2-0303:Density"
    duplicate_kind: str = ""             # components+value | value_only
    status: str = "ok"                   # see STATUS_ORDER


# What keeps a prose row out of the graph, in precedence order. First match wins.
# Only "ok" is loaded.
STATUS_ORDER = ("qualitative", "unverified", "duplicate", "unresolved_components", "ok")


class LLMExtraction(BaseModel):
    """Container so the model can return a list. Its JSON schema constrains the call."""

    measurements: list["LLMMeasurementDraft"]


class LLMMeasurementDraft(BaseModel):
    """What we ask the model for — deliberately smaller than LLMMeasurement.

    Every field is REQUIRED BUT NULLABLE, i.e. `Field(description=...)` with no
    default. This is load-bearing: in pydantic v2 `Optional[X] = None` leaves the
    field out of the schema's `required` list, so the grammar lets the model skip
    the key entirely — which is exactly why molar_ratio, unit and temperature_C
    came back 0/59 populated. With no default the model must emit the key and
    decide between a value and null.

    The descriptions are dropped by ollama when it converts this schema to a GBNF
    grammar (they do reach the anthropic backend, which passes input_schema
    through). So the same instructions are repeated in extract_text_llm.PROMPT.
    """

    components: list[str] = Field(
        description="Chemicals in the DES, HBA first. Full chemical names, never "
                    "abbreviations: 'Choline chloride', not 'ChCl'.")
    molar_ratio: Optional[str] = Field(
        description="Mixing ratio exactly as written, e.g. '1:2', '1:1.5', '1:1:1'.")
    property: PropertyName = Field(
        description="Which of the six properties this number is.")
    value: Optional[float] = Field(
        description="The number stated in the text. null when the text only ranks or "
                    "compares DESs without giving a number. Never invent one.")
    unit: Optional[str] = Field(
        description="Unit as written, e.g. '°C', 'g·cm-3', 'mPa·s', 'mS·cm-1'.")
    temperature_C: Optional[float] = Field(
        description="Measurement temperature in Celsius, if stated separately.")
    ref_numbers: Optional[str] = Field(
        description="The bracketed citation this number is attributed to, e.g. '113' "
                    "or '73,80'. The nearest one, not every citation in the sentence.")
    source_text: str = Field(
        description="The clause containing the number, verbatim and SHORT -- at most "
                    "about 150 characters, not a whole paragraph.")


LLMExtraction.model_rebuild()
