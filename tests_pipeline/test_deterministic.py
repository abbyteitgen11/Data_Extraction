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


def test_legend_abbreviations():                           # Jablonský Table 2 legend
    from des_pipeline.paper_dictionary import legend_definitions

    got = dict(legend_definitions(
        "CF3CONH2–2,2,2-Trifluoroacetamide, 1,3-Me2U–1,3-dimethylurea, CitA–Citric acid, "
        "ChCl–Choline chloride, Gly–Glycerol, U–Urea"))
    assert got["U"] == "Urea" and got["Gly"] == "Glycerol"
    assert got["CF3CONH2"] == "2,2,2-Trifluoroacetamide"
    assert got["1,3-Me2U"] == "1,3-dimethylurea"
    assert legend_definitions("aStandard uncertainty of temperature is u(T) = 0.1 K.") == []


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


def test_kelvin_default_temperature_is_converted():     # Jablonský T2 "at 298 K"
    from des_pipeline.profile_table import repair

    table = _table(["DES", "Density"], [["A:B", "1.1"], ["C:D", "1.2"], ["E:F", "1.3"]])
    profile = _profile([_col(0, "DES", "mixture"),
                        _col(1, "Density", "property", "Density", "g/cm3")])
    profile.default_temperature_C = 298.0
    profile, notes = repair(profile, table)
    assert math.isclose(profile.default_temperature_C, 24.85)


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


def _fresh_budget(tmpdir):
    """Point the budget ledger at an empty temp folder; return a restore function."""
    from des_pipeline import budget

    saved = (budget._usage_files, dict(budget._STATE), config.USAGE_CSV,
             config.BUDGET_USD)
    config.USAGE_CSV = Path(tmpdir) / "usage.csv"
    budget._usage_files = lambda: [config.USAGE_CSV] if config.USAGE_CSV.exists() else []
    budget._STATE.update({"month": None, "month_spent": None, "run_spent": None,
                          "reserved": 0.0})

    def restore():
        budget._usage_files = saved[0]
        budget._STATE.clear()
        budget._STATE.update(saved[1])
        config.USAGE_CSV, config.BUDGET_USD = saved[2], saved[3]
    return restore


def test_anthropic_request_shape_and_usage_log():
    import tempfile

    from des_pipeline import extract_text_llm as t

    captured = []
    saved = (sys.modules.get("anthropic"), config.ANTHROPIC_MODEL)
    with tempfile.TemporaryDirectory() as tmpdir:
        restore = _fresh_budget(tmpdir)
        try:
            sys.modules["anthropic"] = _fake_anthropic(captured)
            config.BUDGET_USD = 0
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
        finally:
            if saved[0] is None:
                sys.modules.pop("anthropic", None)
            else:
                sys.modules["anthropic"] = saved[0]
            config.ANTHROPIC_MODEL = saved[1]
            restore()


def test_budget_refuses_before_sending():
    """Both caps refuse a call whose WORST CASE would pass them -- before it is sent."""
    import tempfile

    from des_pipeline import budget
    from des_pipeline import extract_text_llm as t

    captured = []
    saved = (sys.modules.get("anthropic"), config.ANTHROPIC_MODEL)
    with tempfile.TemporaryDirectory() as tmpdir:
        restore = _fresh_budget(tmpdir)
        try:
            sys.modules["anthropic"] = _fake_anthropic(captured)
            config.ANTHROPIC_MODEL = "claude-opus-5-5"
            # A month that has already spent $69.90 of a $70 cap: the worst case of one
            # Opus call (16k output tokens x $20/M = $0.32) no longer fits.
            config.USAGE_CSV.write_text(
                "time,run_tag,backend,model,call,input_tokens,output_tokens,"
                "cache_read_tokens,cache_write_tokens,seconds,cost_usd,stop_reason\n"
                f"{__import__('time').strftime('%Y-%m-%d')} 10:00:00,x,anthropic,"
                "claude-opus-5-5,c,0,0,0,0,1,69.90,end_turn\n")
            budget._STATE.update({"month_spent": None, "run_spent": None})
            try:
                t.call_llm("hi", {"type": "object", "properties": {}},
                           backend="anthropic", section_id="s")
                raise AssertionError("monthly cap did not stop the call")
            except budget.BudgetExceeded as exc:
                assert "monthly cap" in str(exc)
            assert captured == []                          # nothing was sent

            # Per-run cap: plenty of monthly room, but DES_BUDGET_USD = $0.10.
            config.USAGE_CSV.unlink()
            budget._STATE.update({"month_spent": None, "run_spent": None,
                                  "reserved": 0.0})
            config.BUDGET_USD = 0.10
            try:
                t.call_llm("hi", {"type": "object", "properties": {}},
                           backend="anthropic", section_id="s")
                raise AssertionError("run cap did not stop the call")
            except budget.BudgetExceeded as exc:
                assert "run cap" in str(exc)
            assert captured == [] and budget._STATE["reserved"] == 0.0
        finally:
            if saved[0] is None:
                sys.modules.pop("anthropic", None)
            else:
                sys.modules["anthropic"] = saved[0]
            config.ANTHROPIC_MODEL = saved[1]
            restore()


def test_reservations_are_released():
    import tempfile

    from des_pipeline import budget

    with tempfile.TemporaryDirectory() as tmpdir:
        restore = _fresh_budget(tmpdir)
        try:
            held = budget.reserve("claude-sonnet-5-5", 3000, 16000)
            assert math.isclose(held, 1000 * 2e-6 + 16000 * 10e-6)
            assert budget._STATE["reserved"] > 0
            budget.settle(held, 0.01)
            assert budget._STATE["reserved"] == 0.0
            assert math.isclose(budget.month_spent(), 0.01)
        finally:
            restore()


# ---------- fixes from the four-model comparison ----------
def test_cell_lines_split_on_break():                       # Anđelović Table 1 "1:2<break/>1:3"
    from lxml import etree

    from des_pipeline.dialects import _cell

    cell = _cell(etree.fromstring("<td>1:2<break/>1:3</td>"))
    assert cell.lines == ("1:2", "1:3") and cell.text == "1:2 1:3"
    d = PaperDictionary("x")
    d.add_mixture("ChCl:Gly", compose(["Choline chloride", "Glycerol"], "1:2"),
                  "definitions_table")
    assert parse_mixture("ChCl:Gly (1:3)", d).ratio_raw == "1:3"   # the row's own ratio


def test_speed_of_sound_is_not_viscosity():                # Azad Table 2 "u (m s−1)"
    assert units.unit_conflict("m s−1", "Viscosity")
    assert units.unit_conflict("cm2 s−1", "Density")           # Dinh Table 6
    assert not units.unit_conflict("mS cm−1", "Conductivity")


def test_undefined_superscript_is_stripped():               # Jablonský Table 3 "−54ᵍᵗ"
    from des_pipeline.extract_table import strip_undefined_superscript

    assert strip_undefined_superscript("−54gt", ("gt",)) == ("−54", "gt")
    assert strip_undefined_superscript("ChCla", ("a",)) == ("ChCla", "")


def test_caption_implied_component_with_typo():             # Canela-Xandri captions
    from des_pipeline.profile_table import caption_implied

    tables = []
    for i, cap in enumerate(["PSTA based DES used as reaction medium",
                             "PTSA based DES in esterification",
                             "PTSA based DES in lignocellulose fractionation"]):
        t = _table(["HBA"], [["ChCl"]])
        t.id, t.caption = f"t{i}", cap
        tables.append(t)
    assert set(caption_implied(tables).values()) == {"PTSA"}


def test_reagent_columns_are_context():                     # Canela-Xandri Table 2
    from des_pipeline.profile_table import repair

    table = _table(["Entry", "HBA", "Alcohol", "Acid", "Ref."],
                   [["2.1", "BTMAC", "MeOH", "Acetic (AA)", "87"]])
    table.caption = "PSTA based DES used in esterification reactions"
    profile = _profile([_col(0, "Entry", "context", field="entry"),
                        _col(1, "HBA", "component"), _col(2, "Alcohol", "component"),
                        _col(3, "Acid", "component"), _col(4, "Ref.", "reference")])
    profile.content_types = ["des_application"]
    profile, notes = repair(profile, table, implied_hint="PTSA")
    assert [c.role for c in profile.columns][1:4] == ["component", "context", "context"]
    assert profile.implied_components == ["PTSA"]


def test_placeholders_are_not_des():                        # Sadeghi Table 1
    assert des_problem(parse_mixture("Cat:X− zMClx"))
    assert not des_problem(parse_mixture("Xylitol:Choline chloride"))


def test_single_letter_abbreviations():                     # Azad text
    from des_pipeline.paper_dictionary import single_letter_aliases

    got = dict(single_letter_aliases(
        ["oxalic acid (OX), ethylene glycol (EG), glycerol (G) and urea (U) as donors",
         "shown in Figure (A) and the samples (B)"]))
    assert got == {"G": "glycerol", "U": "urea"}


def test_water_is_written_when_no_molar_ratio():            # Azad Table 2 "0.06%"
    c = compose(["ChCl", "U"], "1:2", "HBA-HBD (molar ratio)", water_text="0.06%",
                water_header="Water content")
    assert c.water_fraction == 0.0006 and "0.06 wt% water" in c.label()


def test_rows_take_the_dictionary_names_and_code():         # Azad ChCl/U
    d = PaperDictionary("x")
    d.aliases.update({"chcl": "Choline chloride", "u": "Urea"})
    d.add_mixture("ChCl/U", compose(["Choline chloride", "Urea"], "1:2"), "definitions_table")
    row = compose(["choline chloride", "U"], "1:2")
    d.canonicalize(row)
    assert row.components == ["Choline chloride", "Urea"] and row.code == "ChCl/U"
    # the same DES under another code: its own row (the paper uses that name), one entry
    again = d.add_mixture("choline chloride:urea",
                          compose(["choline chloride", "urea"], "1:2"), "model")
    assert again.code == d.lookup("choline chloride:urea").code == "ChCl/U"
    assert [r.code for r in d.rows if r.kind == "mixture"] == ["ChCl/U",
                                                               "choline chloride:urea"]


def test_family_by_majority():                              # Dinh
    from des_pipeline.paper_dictionary import _set_family

    d = PaperDictionary("x")
    for ratio in ("1:2", "1:3", "1:4"):
        d.add_mixture(f"DES{ratio}", compose(["LiTFSI", "trifluoroacetamide"], ratio), "model")
    d.add_mixture("EC-DMC + LiTFSI", compose(["EC", "DMC"], "1:1"), "model")
    _set_family(d)
    assert d.family is not None and d.family.components == ["LiTFSI", "trifluoroacetamide"]


def test_reference_numbers_stay_text():                     # "59.0" written by pandas
    import tempfile

    import pandas as pd

    from des_pipeline import store

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "x.csv"
        pd.DataFrame([{"Source_ref_numbers": "59", "Value": 1.5}]).to_csv(path, index=False)
        df = store._read_csv(path)
        assert df.Source_ref_numbers[0] == "59" and float(df.Value[0]) == 1.5


def test_implausible_measuring_temperature_is_flagged():
    from des_pipeline.extract_table import _temperature_plausibility

    assert _temperature_plausibility("Density", {"Temperature_C": 298.0, "Value": 1.1})
    assert not _temperature_plausibility("Density", {"Temperature_C": 25.0, "Value": 1.1})
    assert _temperature_plausibility("Melting_point",
                                     {"Value": 285.15, "Unit_as_written": "C"})


# ---------- runner ----------

def test_unitless_water_numbers_are_not_percent():          # Canela-Xandri Table 7
    two = compose(["ChCl", "PTSA"], water_text="2", water_header="H2O")
    assert two.water_fraction is None and "Water" in two.components
    listed = compose(["ChCl", "PTSA"], water_text="1,2,3", water_header="H2O")
    assert listed.water_fraction is None and any("several" in n for n in listed.notes)
    assert compose(["ChCl", "U"], water_text="0.06%", water_header="Water").water_fraction \
        == 0.0006
    assert compose(["ChCl", "U"], water_text="5", water_header="Water (wt)").water_fraction \
        == 0.05

def test_caption_abbreviations_and_stereo_prefix():         # Jablonský Tables 2-3
    from des_pipeline.paper_dictionary import legend_definitions
    pairs = dict(legend_definitions(
        "Ternary DESs at 298 K (AA = acetamide; MA = malic acid; [BMPI]Br = "
        "1-butyl-1-methylpyrrolidinium bromide); gt = glass transition temperature."))
    assert pairs["AA"] == "acetamide" and pairs["MA"] == "malic acid"
    assert pairs["[BMPI]Br"] == "1-butyl-1-methylpyrrolidinium bromide"
    legend = dict(legend_definitions("OxA–Oxalic acid, [P(Me)(Ph)3]Br–Methyltriphenyl"
                                     "phosphonium bromide, P(Bz)(Ph)3]Cl–Benzoyl chloride"))
    assert legend["OxA"] == "Oxalic acid" and "[P(Me)(Ph)3]Br" in legend
    d = PaperDictionary("x")
    d.aliases.update({"glu": "Glucose", "p(bz)(ph)3]cl": "Benzoyl chloride"})
    assert d.resolve("D-Glu") == "D-Glucose"
    assert d.resolve("[P(Bz)(Ph)3]Cl") == "Benzoyl chloride"     # bracket lost in print

def test_superscript_defined_in_caption_names_the_property():  # Jablonský "−54ᵍᵗ"
    from types import SimpleNamespace
    from des_pipeline.extract_table import marker_property
    table = SimpleNamespace(caption="Ternary DESs at 298 K (U = urea); gt = glass "
                                    "transition temperature.", footnotes="")
    assert marker_property(table, "gt", "Melting_point") == "Glass_transition"
    assert marker_property(table, "a", "Melting_point") == "Melting_point"



def test_offline_crossref_lookup_is_not_cached_as_done():    # laptop asleep mid-run
    import requests
    from des_pipeline import extract_references as er

    class Offline:
        def get(self, *a, **k):
            raise requests.ConnectionError("no network")

    er.time.sleep, sleep = (lambda s: None), er.time.sleep
    try:
        meta = {"num": "1", "title": "A deep eutectic solvent study", "authors": "Smith",
                "year": "2020", "journal": "J Mol Liq"}
        assert er.crossref_search(meta, Offline())[2] == er.NETWORK_FAILED
        assert er.crossref_metadata("10.1/x", Offline()) == er.NETWORK_FAILED
    finally:
        er.time.sleep = sleep


def test_graph_keys_and_prose_links():
    from des_pipeline import build_graph as bg

    rows = bg._doi_keys([{"key": "10.1039/C1RA00630D", "Source_paper_keys":
                          "10.1039/C0GC00395F|X_2020#ref3"}])
    assert rows[0]["key"] == "10.1039/c1ra00630d"
    assert rows[0]["Source_paper_keys"] == "10.1039/c0gc00395f|X_2020#ref3"

    def prose(mixture, ratio, cited=None):
        return {"Measurement_key": f"m{ratio}", "Mixture": mixture, "property": "Density",
                "value": 1.1, "components_resolved": "Choline chloride;Urea",
                "molar_ratio": ratio, "Paper_DOI": "10.1/p", "Paper_key": "10.1/p",
                "Source_paper_keys": cited}

    rows = [prose("ChCl:U", "1:2"), prose("ChCl:U", "1:3", "10.1/cited")]
    roles = {"10.1/p": "primary"}
    mixtures = bg._prose_mixture_rows(rows, {})
    measured = bg._prose_measurement_rows(rows, roles)
    assert {m["key"] for m in mixtures} == {m["mixtures"][0]["key"] for m in measured}
    assert measured[0]["source_keys"] == ["10.1/p"]          # own data: the paper itself
    assert measured[1]["source_keys"] == ["10.1/cited"]
    assert bg._prose_measurement_rows(rows, {"10.1/p": "review"})[0]["review_papers"] \
        == ["10.1/p"]


def test_mixture_cells_from_the_corpus():
    from des_pipeline.composition import compose, des_problem, parse_mixture as p

    c = p("ChCl/Gly (1:2 mol mol−1) + ZnCl2 (20 mol%")            # PMC10974547
    assert c.components == ["ChCl", "Gly", "ZnCl2"] and c.ratio_raw == "1:2"
    c = p("DES#02:Bet:OxA (2:1 30 wt % H2O")                      # PMC10373463
    assert c.components == ["Bet", "OxA", "Water"] and c.water_fraction == 0.3
    c = p("Choline chloride:urea (1:2) + 20 wt% H2O)")             # PMC13377926
    assert c.components[-1] == "Water" and not c.additives
    for several in ("choline chloride:ethylene glycol 1:2 ChCl:urea 1:2 ChCl:glycerol 1:2",
                    "menthol:camphor (1:1) and menthol:borneol (1:1)",
                    "Choline chloride:Urea ChCl:Glycerol ChCl:Formic acid ChCl:Acetic acid"):
        assert des_problem(p(several)).startswith("several DES"), several
    for junk in ("Cat+X−:zRZ (1:1)", "solid–liquid extraction:Ethanol",
                 "Citric acid:Citric acid"):
        assert des_problem(p(junk)), junk
    assert not des_problem(p("ChCl:Gly (1:2, 1:3)"))               # ratios of ONE DES
    assert des_problem(p("ChCl/EG (1:2) ChCl/Gly (1:2) ChCl/U (1:2)")).startswith("several")
    assert p("ChCl:EG (1:2)-10% water").water_fraction == 0.1
    assert compose(["thymol", "menthol"], implied=["Monoterpenoid"]).components == \
        ["thymol", "menthol"]                                       # a class, not a compound
    implied = compose(["choline chloride", "urea"], implied=["ChCl"],
                      resolve=lambda n: "choline chloride" if n == "ChCl" else n)
    assert implied.components == ["choline chloride", "urea"]     # PMC8271379


def test_case_variants_merge_but_short_names_do_not():
    from des_pipeline.build_graph import _case_merges

    merges = _case_merges(["L-Proline", "l-proline", "l-proline", "CO", "Co"], {})
    assert merges == {"L-Proline": "l-proline"}
    assert _case_merges(["L-Proline", "l-proline"], {"Proline": "L-Proline"}) == \
        {"l-proline": "L-Proline"}


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

