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

from pydantic import BaseModel

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
    """A measurement the LLM claims to have found in prose. -> data/sections_llm.csv

    Never loaded into the graph without --include-llm. Prose numbers are far less
    trustworthy than the table.
    """

    components: str = ""                 # ";"-joined, e.g. "choline chloride;urea"
    molar_ratio: Optional[str] = None
    property: PropertyName
    value: float
    unit: Optional[str] = None
    temperature_C: Optional[float] = None
    source_text: str = ""                # the model's quote — checked against the real text
    section_id: str = ""
    section_title: str = ""
    review_doi: str = ""
    verified: bool = False               # does `value` actually occur in the XML section?
    status: str = "needs_review"


class LLMExtraction(BaseModel):
    """Container so the model can return a list. Its JSON schema constrains the call."""

    measurements: list["LLMMeasurementDraft"]


class LLMMeasurementDraft(BaseModel):
    """What we ask the model for — deliberately smaller than LLMMeasurement.

    Fields we already know (section, DOI, verification) are filled in afterwards
    rather than asked for, so the model has less room to invent.
    """

    components: list[str]
    molar_ratio: Optional[str] = None
    property: PropertyName
    value: float
    unit: Optional[str] = None
    temperature_C: Optional[float] = None
    source_text: str


LLMExtraction.model_rebuild()
