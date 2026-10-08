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
from typing import Literal, Optional

from pydantic import BaseModel, Field, create_model, model_validator

from .config import PROPERTY_NAMES

# Derived, not written out. This used to be a hand-typed Literal beside a hand-typed
# list of MixtureRow fields, with two asserts to catch them drifting apart from
# config.PROPERTIES -- and that drift is the bug that hid all 332 melting points from
# the graph for months. Generating both from the one declaration makes the drift
# impossible rather than merely detected, which is what lets a property be added to
# config.PROPERTIES and nowhere else.
PropertyName = Literal[tuple(PROPERTY_NAMES)]        # type: ignore[valid-type]


def _property_triples():
    """The four fields every property contributes to a wide mixture row.

    value / unit / temperature / source column, named as they always were, so the CSV
    column names and every `getattr(row, f"Units_{suffix}")` in the pipeline are
    unchanged. Only the typing-out by hand is gone.
    """
    fields = {}
    for name in PROPERTY_NAMES:
        suffix = name.lower()
        fields[name] = (Optional[float], None)
        fields[f"Units_{suffix}"] = (Optional[str], None)
        fields[f"Temperature_{suffix}"] = (Optional[float], None)
        fields[f"Source_col_{suffix}"] = (Optional[int], None)
    return fields


class _MixtureRowFields(BaseModel):
    """Everything on a mixture row except the per-property triples.

    Those are appended by `create_model` below, from config.PROPERTIES, so they end up
    grouped at the end of the CSV instead of spelled out here eight times over.
    """

    Row_id: str                          # "<slug>:<table>:0001" -- scoped, so a second
                                         # paper's table cannot overwrite this one
    Mixture_key: str = ""                # hash(component set + ratio): what the DES IS,
                                         # independent of the order a table printed it in
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


MixtureRow = create_model("MixtureRow", __base__=_MixtureRowFields,
                          **_property_triples())
MixtureRow.__doc__ = """One row of a property table = one DES mixture. -> mixtures.csv

Wide: a value/unit/temperature/source-column quadruple per property in
config.PROPERTIES, appended by create_model so the vocabulary has one declaration.
`to_measurements` derives the long view the graph and any ML training want.
"""


# How a measurement's temperature was established, best evidence first. There is no
# "assumed 25 C" any more: `config.DEFAULT_TEMP` used to supply one and it invented a
# condition the paper never stated.
TEMPERATURE_SOURCES = ("cell_inline", "marker", "condition_column", "column_header",
                       "caption_default", "unstated")

# What keeps a table measurement out of the graph, in precedence order. Same shape as
# COMPONENT_PROPERTY_STATUS_ORDER below, for the same reason: the row is always written
# to the CSV, and the status is what decides whether it loads.
MEASUREMENT_STATUS_ORDER = (
    "unhandled_unit",     # the printed unit names a different quantity (cSt, spec. grav.)
    "ambiguous_basis",    # Heat_capacity or Polarity with no basis readable from the unit
    "no_solute",          # Solubility with nothing saying what dissolved
    "range_only",         # the cell states a range, not a value -- never averaged
    "ok",
)


class MeasurementRow(BaseModel):
    """One property value. Derived from MixtureRow. -> data/measurements_long.csv

    This is what build_graph.py loads, and the natural shape for ML training later.
    """

    Measurement_key: str                 # "<Row_id>:<Property>" -- unique across papers
    Row_id: str
    Mixture_key: str = ""                # which Mixture node this hangs off
    Paper_key: str = ""
    Paper_DOI: str = ""
    Table_id: str = ""                   # provenance, so validate can re-read the cell
    Source_row: Optional[int] = None
    Source_col: Optional[int] = None
    Mixture: str = ""
    Property: PropertyName
    Value: Optional[float] = None        # canonical unit; null for a range-only cell
    Unit: Optional[str] = None

    # --- what the table actually printed, before conversion ---
    # The pair matters: check_fidelity compares Value_as_written against the cell, so it
    # tests the TRANSCRIPTION. Comparing the converted value would test the converter
    # instead, and a wrong conversion would then look like a fidelity pass.
    Value_as_written: Optional[float] = None
    Unit_as_written: str = ""
    Header_scale: Optional[float] = None    # the "10-3" a header wrote in front of rho
    Uncertainty: Optional[float] = None     # from "1.0101 ± 0.02"
    Value_low: Optional[float] = None       # from "3200-3500"; never averaged
    Value_high: Optional[float] = None
    Qualifier: str = ""                     # "<", ">", "~"

    Temperature_C: Optional[float] = None
    Temperature_source: str = "unstated"    # one of TEMPERATURE_SOURCES

    # Solubility is a property of a solute IN the DES; without the solute the number
    # says nothing and must not merge with another solute's through Dedup_key.
    Solute: str = ""
    Basis: str = ""                         # per_gram | per_mole | ET30 | pi_star | ...

    Source: str = ""                     # provenance: where in the paper
    Source_ref_numbers: str = ""
    Source_DOIs: str = ""
    Source_paper_keys: str = ""
    Dedup_key: str = ""                  # same primary datum reported by another paper
    plausible: bool = True               # within the property's physical range?
    plausibility_note: str = ""
    status: str = "ok"                   # see MEASUREMENT_STATUS_ORDER


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
    match_basis: str = ""                # inline_doi | title | journal_volume_page |
                                         # insufficient. A low score means something
                                         # different for each, and title_agreement
                                         # cannot be computed without a title at all.
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
    matched_name: str = ""               # the query PubChem actually matched, when it
                                         # was not `name` verbatim -- e.g. a hydrate
                                         # reached as "FERRIC CHLORIDE hexahydrate"
    synonyms: str = ""                   # PubChem's own names for this compound, ";"-
                                         # joined. Kept because it is the only evidence
                                         # that separates a naming variant ("H2O" IS a
                                         # synonym of water) from a mis-resolution
                                         # ("PEG600" is NOT a synonym of diethylene
                                         # glycol, though PubChem returned that CID)

    # --- QM9: DFT descriptors for ONE ISOLATED MOLECULE IN THE GAS PHASE ---
    # Computed, never measured, and not a property of a DES. Units are the ones
    # torch_geometric returns (eV, not the raw files' Hartree), so they are in the
    # field names. See des_pipeline/qm9.py.
    qm9_id: str = ""                     # the GDB id it matched
    qm9_n_matches: Optional[int] = None  # >1 means the skeleton was ambiguous
    qm9_contested: str = ""              # set INSTEAD of a match: other component names
                                         # sharing this InChIKey, so the ID is in doubt
    qm9_dipole_D: Optional[float] = None
    qm9_polarizability_a0_3: Optional[float] = None
    qm9_homo_eV: Optional[float] = None
    qm9_lumo_eV: Optional[float] = None
    qm9_gap_eV: Optional[float] = None
    qm9_r2_a0_2: Optional[float] = None
    qm9_zpve_eV: Optional[float] = None
    qm9_cv_cal_mol_K: Optional[float] = None


# The two asserts that used to sit here -- one guarding PropertyName against
# config.PROPERTIES, one guarding MixtureRow's hand-written triples against it -- are
# gone because both are now generated from that one declaration. A guard that detects
# drift is worth having; not being able to drift is better.


class ApplicationRow(BaseModel):
    """One row of a table saying what a DES was USED FOR. -> applications.csv

    `Detail` is deliberately a flat "key=value | key=value" string rather than typed
    columns: the fields differ from table to table (alcohol/acid here, biomass source
    and lignin yield there), and inventing a union of every column any paper might
    print would be a schema nobody could read. The graph unpacks it onto the
    relationship, where heterogeneity costs nothing.
    """

    Application_key: str
    Paper_key: str = ""
    Paper_DOI: str = ""
    Table_id: str = ""                   # "" for a prose row; Section_id carries it
    Source_row: Optional[int] = None
    Domain: str = ""                     # esterification, biodiesel, ...
    Table_caption: str = ""
    Component_1: Optional[str] = None
    Component_2: Optional[str] = None
    Component_3: Optional[str] = None
    Ratio_raw: str = ""
    Mixture: str = ""
    Component_flag: str = ""
    Components_written: str = ""         # as the table printed them, before resolution
    Implied_components: str = ""         # taken from the caption, not from a column
    Detail: str = ""

    # --- where this row came from, and how much to trust it ---
    # A table constrains the model with columns; running prose does not, so a prose
    # application can confuse the solvent with what the solvent acted on and nothing
    # structural stops it. Keeping the origin on the row means a query can exclude them
    # in one clause instead of having to guess.
    Extractor: str = "table"             # table | prose
    Section_id: str = ""                 # prose only
    Section_title: str = ""              # prose only
    Role: str = ""                        # solvent | catalyst | extractant | ...
    Target: str = ""                      # what the DES acted ON
    Outcome: Optional[float] = None       # a yield / efficiency / recovery, when stated
    Outcome_unit: str = ""
    Outcome_metric: str = ""              # what the number measures, as written
    source_text: str = ""                 # the verbatim clause, for a prose row
    quote_found: bool = True              # ...and is it really in the section? (prose)
    duplicate_of: str = ""                # a table Application_key this restates
    status: str = "ok"                    # ok | unverified | duplicate | unresolved_components

    Source_ref_numbers: str = ""
    Source_DOIs: str = ""
    Source_paper_keys: str = ""
    Source_titles: str = ""
    Source_years: str = ""


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


# What a table can hold. A LIST of these, not one of them: `record_type` was a single
# value, and `extract_tables` skipped anything that was not des_properties while
# `extract_applications` took only des_application -- so a table with a composition and
# measured properties lost whichever half the label did not name, and `repair()` went
# further and demoted the property columns outright.
ContentType = Literal["des_properties", "des_application", "des_definitions",
                      "component_properties", "other"]


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
    content_types: list[ContentType] = Field(
        description="EVERY kind of data this table holds -- a list, because one table "
                    "often holds more than one. A table giving each DES a composition "
                    "AND a measured density AND the yield it achieved is all three of "
                    "des_definitions, des_properties and des_application. Do not pick "
                    "the single best label. Empty only when the table holds none of "
                    "them, in which case use [\"other\"].")
    layout: Literal["wide_per_mixture", "paneled_by_mixture"] = Field(
        description="wide_per_mixture = one data row per DES, properties in fixed "
                    "columns. paneled_by_mixture = the SAME column pattern repeats "
                    "across the table (e.g. T, value, T, value, T, value) and the DES "
                    "each group belongs to is named in a spanning row above its data, "
                    "not in a column.")
    reason: str = Field(description="One sentence on why, for a human reading the log.")
    header_row_count: int = Field(
        description="How many of the leading rows are header rather than data.")
    implied_components: list[str] = Field(
        description="Components every row contains that NO column lists, stated in the "
                    "caption instead -- a caption reading 'PTSA based DES' with only an "
                    "HBA column means PTSA is in every mixture. Empty unless the caption "
                    "actually says so. Never guess from the paper's subject.")
    implied_from: str = Field(
        description="The exact caption words that justify implied_components, so a human "
                    "can check the inference. Empty when there are none.")
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

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_record_type(cls, data):
        """Read a profile written before `content_types` existed. -> data.

        Every table_profiles.json on disk stores the old single `record_type`, and one of
        them -- Sadeghi's -- is pinned `source: "human"` precisely so it is never
        re-derived. Dropping that on the floor would silently re-profile the one table
        whose profile has been checked by hand, which is the operation that once stamped
        407 measurements at 0 C.
        """
        if not isinstance(data, dict):
            return data
        if data.get("content_types"):
            return data
        legacy = data.get("record_type")
        if legacy:
            data = {**data, "content_types": [legacy]}
        return data

    @property
    def record_type(self):
        """The first content type, for log lines that want one word. Read-only.

        Deliberately not writable: code that has to DECIDE something must ask
        `"des_properties" in profile.content_types`, because a table can be two things.
        """
        return self.content_types[0] if self.content_types else "other"


class ComponentPropertyRow(BaseModel):
    """One property value for one component, from ONE named source.

    Long format: the single best guess stays a scalar on ComponentRow, and this is
    where the disagreements live. PubChem routinely reports the same property twice
    from different sources -- ammonium thiocyanate has a melting point of 320 F from
    CAMEO and 149.6 C from elsewhere -- and the scalar path keeps only the first.
    """

    key: str                             # hash(member_key); one shape across all routes
    member_key: str = ""                 # the source record: "<component>:<property>:<value>:<data_source>"
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
    pressure_kPa: Optional[float] = None    # normalised, like temperature_C
    pressure_raw: str = ""                  # "760 mm Hg", as the source wrote it
    condition_note: str = ""                # anything in the pressure slot that is NOT a
                                            # pressure: "closed capillary, rapid heating"
    qualifier: str = ""                     # approximate|greater_than|less_than|decomposes|sublimes
    applies_to: str = ""                    # "PEG 400", "dl-Form" -- a DIFFERENT substance

    # --- where it came from ---
    data_source: str = ""                # PubChem SourceName, e.g. "CAMEO Chemicals"
    source_record: str = ""              # PubChem Reference Name
    source_record_matches: bool = True   # ...and does it name THIS component? (advisory)
    source_db: str = "pubchem"           # pubchem | nist
    extractor: str = "llm"               # llm | regex
    source_text: str = ""                # the source line, verbatim

    verified: bool = False               # does value_as_written occur in source_text?
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
    """Container so the model can return lists. Its JSON schema constrains the call.

    Two lists, because a paper says what a DES IS and what it was USED FOR in the same
    prose. Applications used to be found only when a table tabulated them, which meant
    they were found in one paper of the three and missed in every review that simply
    discusses them.
    """

    measurements: list["LLMMeasurementDraft"]
    applications: list["LLMApplicationDraft"]


class LLMApplicationDraft(BaseModel):
    """One "this DES was used for X" claim in prose.

    Mirrors the fields the table route produces so both feed one ApplicationRow writer.
    Every field is required-but-nullable for the same grammar reason as the measurement
    draft: `Optional[X] = None` drops the key out of the schema's `required` list and
    ollama's grammar then lets the model skip it entirely.
    """

    components: list[str] = Field(
        description="Chemicals in the DES, HBA first. Full chemical names, never "
                    "abbreviations: 'Choline chloride', not 'ChCl'.")
    molar_ratio: Optional[str] = Field(
        description="Mixing ratio exactly as written, e.g. '1:2'. null if not stated.")
    domain: str = Field(
        description="What the DES was used for, in the paper's own words: "
                    "'lignin extraction', 'biodiesel purification', 'CO2 capture'.")
    role: Optional[str] = Field(
        description="What the DES itself did: solvent, catalyst, extractant, "
                    "electrolyte, pretreatment, antisolvent. null if not stated.")
    target: Optional[str] = Field(
        description="What the DES acted ON -- the substrate, feedstock, analyte or "
                    "material. This is NOT part of the solvent: in 'ChCl:urea used to "
                    "extract lignin from birch', the DES is choline chloride and urea "
                    "and the target is lignin. null if not stated.")
    outcome: Optional[float] = Field(
        description="The number quantifying how well it worked -- a yield, efficiency "
                    "or recovery. null when the text gives none. Never invent one.")
    outcome_unit: Optional[str] = Field(
        description="Unit of that number as written, usually '%'. null if none.")
    outcome_metric: Optional[str] = Field(
        description="What the number measures, as written: 'lignin yield', 'extraction "
                    "efficiency', 'removal rate'. null if there is no number.")
    ref_numbers: Optional[str] = Field(
        description="The bracketed citation this claim is attributed to, e.g. '113' or "
                    "'73,80'. The nearest one, not every citation in the sentence.")
    source_text: str = Field(
        description="The clause stating this, verbatim and SHORT -- at most about 150 "
                    "characters, not a whole paragraph.")


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
