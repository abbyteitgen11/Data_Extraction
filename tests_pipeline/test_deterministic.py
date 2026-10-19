"""
Tests for the deterministic parts of the pipeline -- everything that does not need a
model, a network or a database. Every case is taken from a real cell or sentence in one
of the thirteen validation papers, named in the test.

    python tests_pipeline/test_deterministic.py        # no dependencies beyond the pipeline
    pytest tests_pipeline                              # also works, if pytest is installed

Not in tests/ because .gitignore excludes that name.
"""
import json
import math
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from des_pipeline import config, units  # noqa: E402
from des_pipeline.composition import compose, des_problem, parse_mixture  # noqa: E402
from des_pipeline.dialects import Cell, Table  # noqa: E402
from des_pipeline.paper import ascii_slug  # noqa: E402
from des_pipeline.paper_dictionary import PaperDictionary  # noqa: E402
from des_pipeline.schema import ColumnSpec, LLMMeasurement, TableProfile  # noqa: E402
from des_pipeline.textmatch import number_in, quote_found  # noqa: E402

ALIASES = {"chcl": "Choline chloride", "gly": "Glycerol", "eg": "Ethylene glycol",
           "la": "Lactic acid", "pgl": "Propylene glycol", "l-pro": "L-Proline"}


def resolve(name):
    return ALIASES.get(str(name).strip().lower())


# ---------- composition: every cell format from the review ----------
def test_whole_des_in_one_cell():                       # Anđelović Table 2
    c = parse_mixture("ChCl:Gly (1:2)", resolve=resolve)
    assert c.components == ["Choline chloride", "Glycerol"]
    assert c.ratio_raw == "1:2" and c.ratio_molar == "1:2"


def test_ternary_with_fractional_part():                 # Anđelović Table 2
    c = parse_mixture("ChCl:PG:ZnCl2 (1:4:0.02)", resolve=resolve)
    assert len(c.components) == 3 and c.ratio_molar == "1:4:0.02"


def test_slash_separated_components_and_racemate():     # Fan Table 2 / Sadeghi
    assert parse_mixture("Glycerol/Ethylene glycol").components == \
        ["Glycerol", "Ethylene glycol"]
    assert parse_mixture("D/L-Proline").components == ["D/L-Proline"]
    assert parse_mixture("ChCl/U").components == ["ChCl", "U"]          # Azad Table 2


def test_hyphen_between_abbreviations_only():
    assert parse_mixture("ChCl-Urea (1:2)", resolve=resolve).components == \
        ["Choline chloride", "Urea"]
    assert parse_mixture("L-Proline").components == ["L-Proline"]
    assert parse_mixture("N-methylurea").components == ["N-methylurea"]


def test_coefficient_on_a_name():                        # Canela-Xandri Table 4
    c = parse_mixture("ChCl : 2PTSA", resolve=resolve)
    assert c.components == ["Choline chloride", "PTSA"] and c.ratio_raw == "1:2"


def test_several_ratios_are_not_resolved_to_one():       # Marco-Velasco glossary
    c = parse_mixture("[BMIM][Br]/DEG (2:1, 1:1, 1:2)")
    assert c.ratio_raw == "2:1, 1:1, 1:2" and c.ratio_molar == ""


def test_water_percent_becomes_a_component_with_computed_ratio():   # Marco-Velasco T3
    c = compose(["ChCl", "Lactic Acid"], "1:1 +20% water", "Molar ratio", resolve=resolve)
    assert c.components[-1] == "Water"
    assert c.ratio_raw == "1:1"                          # what the paper wrote, kept
    assert c.ratio_source == "computed"
    # 20 wt% water on 1 ChCl (139.62) + 1 LA (90.08): water mass = 229.7/4 g = 3.19 mol
    water = c.parts[-1]
    assert math.isclose(water, 0.2 / 0.8 * (139.62 + 90.08) / 18.015, rel_tol=0.01)


def test_water_in_a_component_cell():                    # Marco-Velasco "4% water"
    c = compose(["ChCl:LA", "4% water"], "1:2", "Molar ratio", resolve=resolve)
    assert "Water" in c.components and c.water_fraction == 0.04


def test_bare_number_under_hba_hbd_header():             # Solcan Table 1
    c = compose(["L-Pro", "PGL"], "2", "HBA:HBD Molar Ratio", water_text="30",
                water_header="Water (%)", resolve=resolve)
    assert c.ratio_raw == "2:1" and c.ratio_source == "computed"   # water added
    assert c.components == ["L-Proline", "Propylene glycol", "Water"]


def test_two_ratio_columns_are_one_ratio():              # Fan Table 2 "Molar Ratio"
    c = compose(["Choline chloride", "Glycerol"], "1:3", "Molar Ratio")
    assert c.ratio_molar == "1:3"


def test_footnote_letter_and_abbreviation_are_cleaned():  # Huang T1, Alotaibi T1
    c = parse_mixture("1-methyl-3-octyl imidazolium chloride a:1-undecanol b")
    assert c.components == ["1-methyl-3-octyl imidazolium chloride", "1-undecanol"]
    c = compose(["Choline Chloride (ChCl)", "Ethylene Glycol (EG)"], "1:4", "ratio")
    assert c.components == ["Choline Chloride", "Ethylene Glycol"]
    assert c.aliases == {"ChCl": "Choline Chloride", "EG": "Ethylene Glycol"}


# ---------- the DES gate ----------
def test_des_gate():
    assert des_problem(parse_mixture("-CH3"))                          # Alotaibi T6
    assert des_problem(parse_mixture("DG (Decyl glucoside)"))          # Verma T9
    assert des_problem(parse_mixture("[C2C1im]Cl"))                    # Barbará T5
    assert des_problem(parse_mixture("1554"))                          # Solcan T9
    assert not des_problem(parse_mixture("ChCl:Urea (1:2)"))
    # salt:water at a molar ratio is a DES (Sadeghi lists 4)
    assert not des_problem(compose(["Methylbutylmorpholinium bromide", "Water"], "1:4"))
    # an IL used AS the HBA with a donor is a DES (Huang's DES1-5)
    assert not des_problem(parse_mixture("1-methyl-3-octyl imidazolium chloride:1-undecanol"))


# ---------- the paper's own dictionary ----------
def test_numbered_variant_from_a_ratio_range():          # Alotaibi T1 -> T2
    d = PaperDictionary("x")
    d.add_mixture("CEM", compose(["Choline Chloride", "Ethylene Glycol", "Malonic Acid"],
                                 "1:4:(1–11)", "Molar ratio"), "definitions_table")
    c = d.lookup("CEM5")
    assert c is not None and c.ratio_raw == "1:4:5" and c.ratio_source == "inferred"
    assert d.lookup("CEM12") is None                     # outside the stated range


def test_generic_des_resolves_to_the_family():           # Dinh "DES 1:4 + 10 %wt EC"
    from des_pipeline.composition import Composition

    d = PaperDictionary("x")
    d.family = Composition(components=["LiTFSI", "TFA"])
    c = parse_mixture("DES 1:4", dictionary=d)
    assert c.components == ["LiTFSI", "TFA"] and c.ratio_raw == "1:4"
    c = parse_mixture("DES 1:4 + 10 %wt EC", dictionary=d)
    assert c.additives == {"EC": (0.1, "wt")} and "EC" in c.components


def test_glossary_reader_handles_array_encoding():       # Marco-Velasco back matter
    from lxml import etree

    from des_pipeline.dialects import glossary_pairs

    root = etree.fromstring(
        "<article><back><glossary><title>List of Symbols and Abbreviations</title>"
        "<array><tbody><tr><td>1-butyl-3-methyl-imidazolium bromide</td>"
        "<td>[BMIM][Br]</td></tr><tr><td>Choline chloride</td><td>ChCl</td></tr>"
        "</tbody></array></glossary></back></article>")
    assert glossary_pairs(root) == [("[BMIM][Br]", "1-butyl-3-methyl-imidazolium bromide"),
                                    ("ChCl", "Choline chloride")]


def test_dictionary_matching_and_generic_des():           # Dinh, Solcan, Huang
    from des_pipeline.paper_dictionary import _set_family

    d = PaperDictionary("x")
    d.add_mixture("DES1:4", compose(["LiTFSI", "trifluoroacetamide"], "1:4"), "model")
    d.add_mixture("DES 1:4 + 10 %wt EC", compose(["DES", "EC"], "1:4"), "definitions_table")
    _set_family(d)
    entry = d.lookup("DES 1:4 + 10 %wt EC")
    assert entry.components[:2] == ["LiTFSI", "trifluoroacetamide"]
    assert d.matches(parse_mixture("LiTFSI:trifluoroacetamide (1:3)"))
    assert not d.matches(parse_mixture("Fe3O4:MIL-101"))           # Huang Table 5
    # a code alone is not enough: it must name two constituents
    from des_pipeline.composition import Composition
    assert des_problem(Composition(code="X", components=["glucose"]))


def test_ionic_liquid_and_des_caption_is_skipped():          # Barbará Table 7
    from des_pipeline.paper_dictionary import caption_mixes_il_and_des, mentions_des

    t = _table(["substrate", "IL/DES"], [["lignin", "[C4C1im]Cl"]])
    t.caption = "ILs/DESs Catalyzed Lignin Depolymerization"
    assert caption_mixes_il_and_des(t)
    v = _table(["Anode material", "Corrosion inhibitor(s)"], [["Zn", "Diphenylglyoxim"]])
    v.caption = "A summary of some recent reports on corrosion inhibition of Zn anode"
    assert not mentions_des(v)                                     # Verma Table 4


def test_domain_is_the_earliest_keyword():                  # Marco-Velasco T4
    from des_pipeline.extract_applications import domain_of

    assert domain_of("works using DESs for the preparation of flat-sheet polymeric "
                     "membranes for the removal of heavy metals")[0] == "membrane_separation"


# ---------- numbers in table columns (fix A) ----------
def _table(header, rows):
    return Table(id="t1", label="Table 1", header=[[Cell(text=h) for h in header]],
                 rows=[[Cell(text=c) for c in r] for r in rows], n_columns=len(header))


def _profile(columns, **kw):
    return TableProfile(relevant=True, content_types=["des_properties"],
                        layout="wide_per_mixture", reason="", header_row_count=0,
                        implied_components=[], implied_from="", columns=columns,
                        footnote_markers=[], missing_value_tokens=[],
                        default_temperature_C=None, **kw)


def _col(index, header, role, prop=None, unit=None, field=None):
    return ColumnSpec(index=index, header=header, role=role, property=prop,
                      unit_as_written=unit, component_role=None, context_field=field,
                      multi_valued=False)


def test_uncertainty_values_count_as_numbers():          # Alotaibi T2, Anđelović T2
    from des_pipeline.profile_table import _numeric_fraction, repair

    table = _table(["DES", "Density"], [["CE", "1.0101 ± 0.02"], ["CEM1", "1.0112 ± 0.04"],
                                        ["CEM5", "1.0179 ± 0.03"], ["CG", "1.01 ± 0.01"],
                                        ["CGT4", "1.1579 ± 0.01"]])
    profile = _profile([_col(0, "DES", "context", field="des_name"),
                        _col(1, "Density", "property", "Density", "g/cm3")])
    assert _numeric_fraction(table, 1, profile) == (1.0, 5)
    profile, notes = repair(profile, table)
    assert profile.columns[1].role == "property"          # not demoted any more
    assert profile.columns[0].role == "mixture"           # the DES column gets identity


def test_unit_that_measures_something_else_is_demoted():  # Alotaibi T6, Jablonský T7
    from des_pipeline.profile_table import repair

    table = _table(["DES", "Pc (Bar)", "Conversion (%)"],
                   [["CE", "45.1", "90"], ["CEM1", "44.0", "91"], ["CEM5", "43.2", "88"],
                    ["CG", "40.0", "70"], ["CGT4", "39.9", "75"]])
    profile = _profile([_col(0, "DES", "mixture"),
                        _col(1, "Pc (Bar)", "property", "Density", "Bar"),
                        _col(2, "Conversion (%)", "property", "pH", "%")])
    profile, notes = repair(profile, table)
    assert [c.role for c in profile.columns] == ["mixture", "other_quantity",
                                                 "other_quantity"]
    assert len(notes) == 2


def test_numeric_name_column_becomes_a_condition():      # Fan T3, qwen3 profile
    from des_pipeline.profile_table import repair

    table = _table(["K", "W·m−1·K−1"], [["294.59", "0.2385"], ["303.15", "0.2401"],
                                       ["313.15", "0.2420"], ["323.15", "0.2433"]])
    profile = _profile([_col(0, "K", "mixture", unit="K"),
                        _col(1, "W·m−1·K−1", "property", "Thermal_conductivity",
                             "W·m−1·K−1")])
    profile, notes = repair(profile, table)
    assert profile.columns[0].role == "condition"


def test_unit_conflicts():
    assert units.unit_conflict("Bar", "Density")
    assert units.unit_conflict("%", "pH")
    assert units.unit_conflict("kcal mol−1", "Solubility")              # Azad prose
    assert not units.unit_conflict("K", "Melting_point")
    assert not units.unit_conflict("cp", "Viscosity")
    assert not units.unit_conflict("nD", "Refractive_index")
    assert not units.unit_conflict("cSt", "Viscosity")    # kept, flagged unhandled later


def test_melting_point_gets_no_measuring_temperature():  # Marco-Velasco T3
    from des_pipeline.extract_table import ParsedCell, measurement_fields

    table = _table(["Tm (K)"], [["325.3"]])
    column = _col(0, "Tm (K)", "property", "Melting_point", "K")
    fields = measurement_fields(ParsedCell(value=325.3), column, table, "Melting_point",
                                25.0, "caption_default")
    assert fields["Temperature_C"] is None and fields["Temperature_source"] == "not_applicable"
    assert math.isclose(fields["Value"], 52.15) and fields["Value_as_written"] == 325.3


# ---------- the prose route (fix F) ----------
def test_quote_matching():
    text = "The density of ChCl:Phenol and ChCl:O-cresol (1:2) is 1.097 and 1.078 g/cm3."
    assert quote_found("ChCl:Phenol and ChCl:O-cresol (1:2) is 1.097", text)
    assert quote_found("ChCl:Phenol and ChCl:O–cresol (1:2) is 1.097", text)   # dash
    assert not quote_found("ChCl:Urea has a density of 1.25", text)
    assert number_in(1.078, text) and not number_in(1.08, text)


def _draft_row(**kw):
    base = dict(property="Density", value=1.0, quote_found=True, verified=True,
                components_resolved="A;B", components_written="A;B",
                component_status="resolved", unit="g*cm^-3")
    base.update(kw)
    return LLMMeasurement(**base)


def test_prose_status_order():
    from des_pipeline.extract_text_llm import _status_for

    assert _status_for(_draft_row(value=None)) == "qualitative"
    assert _status_for(_draft_row(quote_found=False)) == "quote_not_found"
    assert _status_for(_draft_row(verified=False)) == "value_not_in_quote"
    assert _status_for(_draft_row(components_resolved="", components_written="")) == \
        "no_des_in_quote"
    assert _status_for(_draft_row(unit="!kcal/mol")) == "unhandled_unit"
    assert _status_for(_draft_row()) == "ok"


def test_respectively_assigns_values_in_order():         # Marco-Velasco prose
    from des_pipeline.extract_text_llm import grounded_composition

    quote = ("ChCl:Phenol and ChCl:O-cresol at a molar ratio of 1:2 have densities of "
             "1.097 and 1.078 g/cm3, respectively [7]")
    assert grounded_composition([], quote, 1.097, None, None, None).components_written \
        == ["ChCl", "Phenol"]
    assert grounded_composition([], quote, 1.078, None, None, None).components_written \
        == ["ChCl", "O-cresol"]


def test_general_sentence_names_no_des():                # Alotaibi prose
    from des_pipeline.extract_text_llm import grounded_composition

    quote = "the refractive index of most binary DES systems lies within 1.48–1.6 nD [20]"
    comp = grounded_composition(["Choline chloride", "Urea"], quote, 1.48, None, None,
                                None)
    assert comp.components == []                          # the model's ChCl:urea refused


# ---------- slugs ----------
def test_slugs_are_transliterated():
    assert ascii_slug("Jablonský_2019") == "Jablonsky_2019"
    assert ascii_slug("Anđelović_2022") == "Andelovic_2022"
    assert ascii_slug("Barbará_2025") == "Barbara_2025"


# ---------- validation (fix K) ----------
def test_fidelity_compares_the_value_as_written():       # Jablonský T6: 445.6 K
    from des_pipeline import validate

    table = _table(["Composition, Molar Ratio", "Boiling Point (K)"],
                   [["Choline chloride, Urea, 1:2", "445.6"]])
    column = _col(1, "Boiling Point (K)", "property", "Boiling_point", "K")
    profile = _profile([_col(0, "Composition", "mixture"), column])
    row = {"Paper_key": "p", "Measurement_key": "m", "Table_id": "t1", "Source_row": 0,
           "Source_col": 1, "Property": "Boiling_point", "Value": 172.45,
           "Value_as_written": 445.6, "status": "ok", "Temperature_C": None}
    saved = (validate.store.read_all, validate._reread_context)
    try:
        validate.store.read_all = lambda kind: [row]
        validate._reread_context = lambda paper_row: ({"t1": table}, {"t1": profile},
                                                      {"t1": {0: profile.columns[0],
                                                              1: column}})
        checked, mismatches = validate.check_fidelity({"Paper_key": "p"})
    finally:
        validate.store.read_all, validate._reread_context = saved
    assert checked == 1 and mismatches == []


# ---------- the Anthropic backend, mocked: no network, no cost ----------
def _fake_anthropic(captured, text='{"x": 1}'):
    usage = types.SimpleNamespace(input_tokens=1000, output_tokens=200,
                                  cache_read_input_tokens=0,
                                  cache_creation_input_tokens=0)
    message = types.SimpleNamespace(
        usage=usage, stop_reason="end_turn",
        content=[types.SimpleNamespace(type="text", text=text)])

    class Messages:
        def create(self, **kwargs):
            captured.append(kwargs)
            return message

    class Client:
        def __init__(self, **kwargs):
            self.messages = Messages()
            self.beta = types.SimpleNamespace(messages=Messages())

    return types.SimpleNamespace(Anthropic=Client)


def test_anthropic_request_shape_and_usage_log(tmp=None):
    import tempfile

    from des_pipeline import extract_text_llm as t

    captured = []
    saved = (sys.modules.get("anthropic"), config.USAGE_CSV, config.ANTHROPIC_MODEL,
             config.BUDGET_USD, t._SPENT)
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            sys.modules["anthropic"] = _fake_anthropic(captured)
            config.USAGE_CSV = Path(tmpdir) / "usage.csv"
            t._SPENT = None
            schema = {"type": "object", "properties": {"x": {"type": "integer",
                                                             "default": 0}}}

            config.ANTHROPIC_MODEL = "claude-sonnet-5-5"
            assert t.call_llm("hi", schema, backend="anthropic", section_id="s1") == \
                '{"x": 1}'
            sent = captured[-1]
            fmt = sent["output_config"]["format"]
            assert fmt["type"] == "json_schema"
            assert fmt["schema"]["additionalProperties"] is False
            assert "default" not in fmt["schema"]["properties"]["x"]
            assert sent["output_config"]["effort"] == config.ANTHROPIC_EFFORT
            assert sent["fallbacks"] == "default" and "tool_choice" not in sent

            config.ANTHROPIC_MODEL = "claude-haiku-4-5"
            t.call_llm("hi", schema, backend="anthropic", section_id="s2")
            assert "effort" not in captured[-1]["output_config"]
            assert "fallbacks" not in captured[-1]

            rows = config.USAGE_CSV.read_text().splitlines()
            assert len(rows) == 3                          # header + two calls
            # Sonnet 5.5: 1000 in x $2/M + 200 out x $10/M = $0.004
            assert "0.004" in rows[1]

            config.BUDGET_USD = 0.001                      # already spent more than this
            try:
                t.call_llm("hi", schema, backend="anthropic", section_id="s3")
                raise AssertionError("budget guard did not stop the call")
            except t.BudgetExceeded:
                pass
        finally:
            if saved[0] is None:
                sys.modules.pop("anthropic", None)
            else:
                sys.modules["anthropic"] = saved[0]
            (config.USAGE_CSV, config.ANTHROPIC_MODEL, config.BUDGET_USD,
             t._SPENT) = saved[1:]


# ---------- runner ----------
if __name__ == "__main__":
    failures = 0
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as exc:                          # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
