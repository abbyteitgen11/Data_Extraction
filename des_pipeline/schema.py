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
    "Melting_point", "Boiling_point", "Density", "Viscosity",
    "Conductivity", "Surface_tension", "Refractive_index",
]

# Guard against the two lists drifting apart — this is the bug that hid all 332
# melting points from the graph for months.
assert set(get_args(PropertyName)) == set(PROPERTY_NAMES), (
    "schema.PropertyName and config.PROPERTIES disagree"
)


class MixtureRow(BaseModel):
    """One row of Table 2 = one DES mixture. -> data/table2_with_dois.csv"""

    Row_id: str                          # "<slug>:<table>:0001" -- scoped, so a second
                                         # paper's table cannot overwrite this one
    Paper_key: str = ""
    Table_id: str = ""                   # provenance: which table
    Source_row: Optional[int] = None     # provenance: which row of it

    # --- identity ---
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
    Source_col_melting_point: Optional[int] = None
    Density: Optional[float] = None
    Units_density: Optional[str] = None
    Temperature_density: Optional[float] = None
    Source_col_density: Optional[int] = None
    Viscosity: Optional[float] = None
    Units_viscosity: Optional[str] = None
    Temperature_viscosity: Optional[float] = None
    Source_col_viscosity: Optional[int] = None
    Conductivity: Optional[float] = None
    Units_conductivity: Optional[str] = None
    Temperature_conductivity: Optional[float] = None
    Source_col_conductivity: Optional[int] = None
    Surface_tension: Optional[float] = None
    Units_surface_tension: Optional[str] = None
    Temperature_surface_tension: Optional[float] = None
    Source_col_surface_tension: Optional[int] = None
    Boiling_point: Optional[float] = None
    Units_boiling_point: Optional[str] = None
    Temperature_boiling_point: Optional[float] = None
    Source_col_boiling_point: Optional[int] = None
    Refractive_index: Optional[float] = None
    Units_refractive_index: Optional[str] = None
    Temperature_refractive_index: Optional[float] = None
    Source_col_refractive_index: Optional[int] = None

    # --- the review paper that contains the table ---
    Paper_DOI: str = ""
    Paper_authors: str = ""
    Paper_title: str = ""
    Paper_journal: str = ""
    Paper_volume: str = ""
    Paper_issue: str = ""
    Paper_year: str = ""

    # --- the primary papers the data actually came from ("|"-joined) ---
    Source_ref_numbers: str = ""
    Source_DOIs: str = ""
    Source_paper_keys: str = ""
    Source_authors: str = ""
    Source_titles: str = ""
    Source_journals: str = ""
    Source_volumes: str = ""
    Source_issues: str = ""
    Source_pages: str = ""
    Source_years: str = ""
    Context: str = ""                    # any role="context" column the profile named


class MeasurementRow(BaseModel):
    """One property value. Derived from MixtureRow. -> data/measurements_long.csv

    This is what build_graph.py loads, and the natural shape for ML training later.
    """

    Measurement_key: str                 # "<Row_id>:<Property>" -- unique across papers
    Row_id: str
    Paper_key: str = ""
    Paper_DOI: str = ""
    Table_id: str = ""                   # provenance, so validate can re-read the cell
    Source_row: Optional[int] = None
    Source_col: Optional[int] = None
    Mixture: str = ""
    Property: PropertyName
    Value: float
    Unit: Optional[str] = None
    Temperature_C: Optional[float] = None
    Source: str = "Table 2"              # provenance: where in the paper
    Locus: str = ""                      # provenance: "row 7"
    Source_ref_numbers: str = ""
    Source_DOIs: str = ""
    Source_paper_keys: str = ""
    Dedup_key: str = ""                  # same primary datum reported by another paper
    plausible: bool = True               # within the property's physical range?
    plausibility_note: str = ""


class ReferenceRow(BaseModel):
    """One bibliography entry. -> data/references.csv"""

    ref_number: int
    key: str = ""                        # the DOI, or "<paper_doi>#refN" when unmatched
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
    paper_doi: str = ""
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
    tpsa: Optional[float] = None            # topological polar surface area, A^2
    rotatable_bond_count: Optional[int] = None
    formal_charge: Optional[int] = None     # pubchempy calls this `charge`
    xlogp: Optional[float] = None           # often null for salts and ionic species
    complexity: Optional[float] = None
    melting_point_C: Optional[float] = None
    boiling_point_C: Optional[float] = None
    density_g_cm3: Optional[float] = None
    property_comments: str = ""          # the raw PubChem strings, which are messy
    sources: str = ""                    # "pubchem;nist"
    lookup_status: str = ""              # ok | not_found | error


class ColumnSpec(BaseModel):
    """What one column of a table means. The model labels; it never reads values."""

    index: int = Field(
        description="0-based column number, exactly as numbered in the header below.")
    header: str = Field(
        description="The header text at that index, copied character-for-character. "
                    "Do not tidy or translate it.")
    role: Literal["component", "ratio", "reference", "property",
                  "condition", "context", "ignore"] = Field(
        description="What the column holds. 'component' = a chemical name; 'ratio' = "
                    "the mixing ratio; 'reference' = a citation into the bibliography; "
                    "'property' = a measured physical property; 'condition' = a "
                    "temperature or pressure the measurements were made at; 'context' "
                    "= descriptive non-numeric data; 'ignore' = anything else.")
    property: Optional[PropertyName] = Field(
        description="Which property, when role is 'property'. null otherwise. If the "
                    "column is a property that is not in the list, use role 'ignore'.")
    unit_as_written: Optional[str] = Field(
        description="The unit exactly as the header or sub-header prints it, e.g. "
                    "'g cm-3', '(mPa s)'. null when the table states none.")
    component_role: Optional[Literal["HBA", "HBD", "either"]] = Field(
        description="For a component column: hydrogen-bond acceptor, donor, or either.")
    context_field: Optional[str] = Field(
        description="For a context column, a short snake_case name for what it "
                    "records, e.g. natural_source, target_compound, technique.")
    multi_valued: bool = Field(
        description="True when one cell of this column holds several values, e.g. on "
                    "separate lines or separated by slashes.")


class FootnoteMarker(BaseModel):
    """One marker the table's own legend defines, e.g. superscript 'a' = 40 C."""

    marker: str = Field(
        description="The marker exactly as printed: 'a', 'i', 'us', 'C'.")
    meaning: Literal["temperature", "ratio_basis", "stability",
                     "not_reported", "other"] = Field(
        description="What the marker signifies.")
    temperature_C: Optional[float] = Field(
        description="The temperature in Celsius, when meaning is 'temperature'.")
    note: str = Field(description="The legend's own wording for this marker.")


class TableProfile(BaseModel):
    """How to read one table. Produced by the LLM, validated, then used by code.

    This is what replaces the hard-coded column indices: the model reads a table's
    caption, headers, legend and a few sample rows and says what each column means,
    and deterministic code then extracts every cell. No data value is ever seen by
    the model, so nothing can be hallucinated, rounded or unit-converted.
    """

    relevant: bool = Field(
        description="False when the table carries no deep-eutectic-solvent data.")
    record_type: Literal["des_properties", "des_application",
                         "component_properties", "other"] = Field(
        description="des_properties = measured physical properties of DES mixtures; "
                    "des_application = a DES used FOR something (source, target, "
                    "technique); component_properties = properties of single pure "
                    "compounds; other = anything else.")
    reason: str = Field(description="One sentence on why, for a human reading the log.")
    header_row_count: int = Field(
        description="How many of the leading rows are header rather than data.")
    columns: list[ColumnSpec] = Field(
        description="Exactly one entry per column index, in order, 0..N-1.")
    footnote_markers: list[FootnoteMarker] = Field(
        description="One entry per marker the legend defines. Empty when there is none.")
    missing_value_tokens: list[str] = Field(
        description="Strings printed in place of a value, e.g. '-', 'n/a', 'nd', or a "
                    "legend-defined token such as 'DT'. Not footnote markers, which "
                    "annotate a value that is present.")
    default_temperature_C: Optional[float] = Field(
        description="The temperature unmarked values were measured at, if the caption "
                    "or legend states one. null otherwise.")


class ComponentPropertyRow(BaseModel):
    """One property value for one component, from ONE named source.

    Long format: the single best guess stays a scalar on ComponentRow, and this is
    where the disagreements live. PubChem routinely reports the same property twice
    from different sources -- ammonium thiocyanate has a melting point of 320 F from
    CAMEO and 149.6 C from elsewhere -- and the scalar path keeps only the first.
    """

    key: str                             # content-derived, so graph loads are idempotent
    name: str                            # joins to ComponentRow.name
    cid: Optional[int] = None
    property: PropertyName

    # --- the value, canonicalised, and as the source actually wrote it ---
    value: Optional[float] = None        # C for temperatures, g*cm^-3 for density
    unit: str = ""
    value_as_written: Optional[float] = None
    unit_as_written: str = ""            # "F", "K", "g/mL", ...

    # --- the conditions it was measured under ---
    temperature_C: Optional[float] = None   # a CONDITION ("1.3057 @ 25 C"), not the value
    pressure: str = ""                      # "760 mm Hg"
    qualifier: str = ""                     # approximate|greater_than|less_than|decomposes|sublimes
    applies_to: str = ""                    # "PEG 400", "dl-Form" -- a DIFFERENT substance

    # --- where it came from ---
    data_source: str = ""                # PubChem SourceName, e.g. "CAMEO Chemicals"
    source_record: str = ""              # PubChem Reference Name
    source_record_matches: bool = True   # ...and does it name THIS component? (advisory)
    source_db: str = "pubchem"           # pubchem | nist
    extractor: str = "llm"               # llm | regex
    raw_string: str = ""                 # the source line, verbatim

    verified: bool = False               # does value_as_written occur in raw_string?
    status: str = "ok"


# What keeps a component property out of the graph, in precedence order. Only "ok" loads.
COMPONENT_PROPERTY_STATUS_ORDER = (
    "qualitative",          # the line hedges: "Solid decomposes", "Sublimes"
    "unverified",           # the number is not on the line -- the model invented it
    "different_substance",  # the value is for PEG 400, not for this component
    "unhandled_unit",       # lb/gal, "Relative density (water = 1)"
    "ok",
)


class ComponentPropertyExtraction(BaseModel):
    """Container so the model can return a list. Its JSON schema constrains the call."""

    values: list["ComponentPropertyDraft"]


class ComponentPropertyDraft(BaseModel):
    """What we ask the model for, reading numbered PubChem property lines.

    It returns a LINE NUMBER, never copied text and never a converted unit. That is
    what makes the result checkable: we already hold the line, so attribution comes
    from PubChem's own Reference map, conversion is done in Python, and verification
    is exact -- the number must appear character-for-character in the line we looked
    up rather than in a quote the model wrote for itself.

    Every field is required-but-nullable (Field with no default). `Optional[X] = None`
    would drop the field out of the schema's `required` list and ollama's grammar
    would then let the model skip the key entirely.
    """

    line: int = Field(
        description="The number of the line this value came from. Copy it exactly.")
    property: PropertyName = Field(
        description="Melting_point, Boiling_point or Density -- the line's own heading.")
    value: Optional[float] = Field(
        description="The number AS WRITTEN, in the unit written on the line. Never "
                    "convert. Never average a range -- emit low and high as two "
                    "records. Never invent. null when the line states no number.")
    unit: Optional[str] = Field(
        description="Unit as written: 'C', 'F', 'K', 'g/cm3', 'g/mL'. null if absent.")
    temperature_C: Optional[float] = Field(
        description="The temperature the value was measured AT ('1.3057 @ 25 C' -> 25). "
                    "For a melting or boiling point the number IS the temperature: put "
                    "it in value and leave this null.")
    pressure: Optional[str] = Field(
        description="Pressure as written, e.g. '760 mm Hg'. null if not stated.")
    qualifier: Optional[str] = Field(
        description="null, or one of: approximate, greater_than, less_than, "
                    "decomposes, sublimes.")
    applies_to: Optional[str] = Field(
        description="null if the value is for this substance. Otherwise the other "
                    "substance, grade, form or isomer it belongs to, as written: "
                    "'PEG 400', 'dl-Form', 'solution'.")


ComponentPropertyExtraction.model_rebuild()


class LLMMeasurement(BaseModel):
    """One measurement claimed in prose. -> data/sections_llm.csv

    Only rows with status == "ok" are loaded into the graph. See STATUS_ORDER below.
    """

    Row_id: str = ""                     # "P-0001"; for humans, not a graph key
    Measurement_key: str = ""            # content-derived, so re-loads are idempotent

    # --- identity ---
    components: str = ""                 # ";"-joined, exactly as the model wrote them
    components_written: str = ""         # the tokens the paper's own quote wrote
    components_source: str = ""          # source_text | model -- which one won
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
    Source_paper_keys: str = ""           # what the graph merges Paper nodes on
    ref_source: str = ""                 # agreed | text | llm | none
    Paper_DOI: str = ""

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
