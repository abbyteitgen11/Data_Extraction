"""
The applications route: what a DES was used FOR, rather than what it measures.

A review like Canela-Xandri tabulates no physical properties at all. Its seven tables
say which DES were used for esterification, for biodiesel, for lignin fractionation,
for cleaning crude oil -- each with the substrates, yields and products that matter for
that domain, and none of which are the same columns twice:

    Table 2   Entry | HBA | Alcohol | Acid                    | Ref.
    Table 4   Entry | HBA : HBD | Biomass source | Lignin yield | Ref.
    Table 5   Entry | HBA : HBD | Material | Product          | Ref.

That heterogeneity is why nothing here is table-specific. The domain comes from the
caption; every other column the profiler labelled `context` becomes a named detail,
using the `context_field` it already assigns. A new application table in a new paper
needs no code.

    (:Mixture) -[:USED_IN {yield_pct, solvent, ...}]-> (:Application {domain})
"""
import itertools
import re

from . import config
from .enrich_components import component_index, resolve_component
from .extract_references import sources
from .extract_table import (parse_components, parse_ratio, ratio_from_coefficients,
                            read_references, with_implied)
from .schema import ApplicationRow

# Domains worth collapsing onto one node. The caption is free text, so two papers will
# word the same use differently ("lignocellulose fractionation" / "lignin extraction");
# matching on keywords keeps the vocabulary small enough to query without inventing an
# ontology. Anything unmatched keeps a snake_case version of its own caption.
DOMAIN_KEYWORDS = (
    ("biodiesel", ("biodiesel", "fame", "transesterification")),
    ("esterification", ("esterification", "ester synthesis")),
    ("biomass_fractionation", ("lignocellulose", "lignin", "fractionation",
                               "delignification", "pretreatment")),
    ("metal_processing", ("metal", "leaching", "solubilization", "solubilisation",
                          "dissolution", "recovery")),
    ("fuel_cleaning", ("crude oil", "desulfurization", "desulphurization",
                       "denitrogenation", "fuel")),
    ("materials_synthesis", ("material", "platform chemical", "nanocrystal",
                             "synthesis of", "cellulose")),
    ("reaction_medium", ("reaction medium", "catalys", "promoter", "organic synthesis")),
    ("extraction", ("extraction", "separation", "absorb")),
)


# A comma FOLLOWED BY A SPACE. Checked against every component name in the corpus: 59
# contain a comma and 55 of those are structural -- "1,2-Propanediol", "N,N-dimethyl
# urea", "...pyrimido[1,2-a]azepine" -- and not one of them puts a space after it. The
# four that do are genuine lists.
_ALTERNATIVES = re.compile(r",\s+")

# A ceiling on the expansion, since alternatives in two columns multiply. Nothing in the
# corpus comes close (the largest is 7), so hitting it means a cell was misread.
MAX_ALTERNATIVES = 24


def alternatives(text):
    """'MeOH, BuOH, HexOH, 2-EH' -> the four names. -> [text] when it names one thing.

    These cells list what a reaction was tried WITH, not what a solvent is made OF:
    "esterification of MeOH, BuOH, HexOH or 2-EH with lauric acid" is four experiments,
    not one four-alcohol mixture. So they expand into separate application rows rather
    than being packed into a mixture's component slots -- which would assert a solvent
    nobody ever made, and assert it plausibly enough to be believed.
    """
    parts = [p.strip() for p in _ALTERNATIVES.split(str(text or "")) if p.strip()]
    if len(parts) < 2:
        return [str(text or "").strip()]
    # Inverted CAS nomenclature ("Ethanol, 2-chloro-") is one name written with a comma
    # and a space; its fragments are not standalone chemicals.
    if any(p.endswith("-") or len(re.findall(r"[A-Za-z]", p)) < 2 for p in parts):
        return [str(text or "").strip()]
    return parts


def domain_of(caption, label=""):
    """A short, queryable name for what this table is about. -> str.

    Deterministic: the caption is already the paper's own statement of the domain, so
    there is nothing here for a model to add.
    """
    text = f"{caption} {label}".lower()
    for domain, keywords in DOMAIN_KEYWORDS:
        if any(k in text for k in keywords):
            return domain
    words = [w for w in re.findall(r"[A-Za-z]+", caption.lower())
             if w not in ("the", "of", "for", "in", "and", "a", "an", "to", "used",
                          "based", "des", "with", "using", "best", "results")]
    return "_".join(words[:3]) or "unspecified"


# A count of water equivalents, not a name: Table 7's "H2O" column holds 1, 2, 3.
# A DES with two waters is a different solvent from the dry one, so the count belongs in
# the composition -- but "0/1" and "-" state no single number and must add nothing.
_WATER_COUNT = re.compile(r"^\d{1,2}$")


def water_equivalents(detail):
    """The water count from a `water_equivalents` context field. -> [counts].

    Returns several when the cell lists alternatives ("1,2,3"), which expand into
    separate rows the same way the esterification alcohols do. Returns none for "0/1"
    or a dash: a cell that will not commit to a number should not have one invented.
    """
    raw = str(detail.get("water_equivalents") or "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if parts and all(_WATER_COUNT.match(p) for p in parts):
        return [int(p) for p in parts]
    return []


def resolve_names(names, vocabulary):
    """Map written component names onto the corpus vocabulary. -> (resolved, written).

    Application tables are written in the paper's own shorthand -- "ChCl", "TEAB" --
    while property tables spell the same chemicals out. Without this the graph holds
    `ChCl` and `Choline chloride` as two unrelated components, and a solvent used for
    esterification never meets its own measured properties. `resolve_component` is the
    same alias-then-PubChem resolver the prose route uses.

    An unresolved name keeps its written form rather than being dropped or guessed at;
    it shows up in the review queue as an abbreviation to define.
    """
    if vocabulary is None:
        return list(names), list(names)
    resolved = []
    for name in names:
        canonical, _how = resolve_component(name, vocabulary)
        resolved.append(canonical or name)
    return resolved, list(names)


def extract_application_table(table, profile, paper, reference_map,
                              vocabulary=None):
    """One des_application table -> list[ApplicationRow]."""
    by_role = {}
    for column in profile.columns:
        by_role.setdefault(column.role, []).append(column)
    component_cols = by_role.get("component", [])
    ratio_col = next(iter(by_role.get("ratio", [])), None)
    ref_col = next(iter(by_role.get("reference", [])), None)
    context_cols = by_role.get("context", [])

    domain = domain_of(table.caption, table.label)
    rows = []
    ragged = {index for index, _ in table.ragged}

    for index, row in enumerate(table.rows):
        if index in ragged:
            continue
        names = [row[c.index].text for c in component_cols if c.index < len(row)]
        names = [n for n in names if n and n.strip() and n.strip() not in config.DASH]
        if not names:
            continue                              # a spacer or a continuation row

        if ratio_col is not None and ratio_col.index < len(row):
            cell = row[ratio_col.index]
            ratio_raw, _parts, _flag = parse_ratio(cell.text, cell.markers)
        else:
            ratio_raw = ""

        ref_numbers = []
        if ref_col is not None and ref_col.index < len(row):
            ref_numbers = read_references(row[ref_col.index], reference_map)
        cited = sources(ref_numbers, reference_map, paper.key)

        detail = {}
        for column in context_cols:
            if column.index >= len(row):
                continue
            text = row[column.index].text.strip()
            if text and text not in config.DASH:
                detail[column.context_field or f"col{column.index}"] = text

        # One row per combination of the alternatives each cell lists. Usually one.
        # Water equivalents join the product, so "1,2,3" waters becomes three rows in
        # exactly the way "MeOH, BuOH, HexOH" becomes three.
        waters = water_equivalents(detail) or [None]
        choices = [alternatives(n) for n in names] + [waters]
        combinations = [(c[:-1], c[-1]) for c in itertools.product(*choices)]
        if len(combinations) > MAX_ALTERNATIVES:
            combinations = combinations[:MAX_ALTERNATIVES]
            detail["alternatives_truncated"] = "yes"

        for n, (chosen, water) in enumerate(combinations):
            written = with_implied(list(chosen), profile)
            if water is not None:
                written = written + ["Water"]
            resolved, _written = resolve_names(written, vocabulary)
            (c1, c2, c3), component_flag, coefficients = parse_components(resolved)
            # "ChCl : 2PTSA" states its stoichiometry on the name because the table has
            # no ratio column; without this the only thing distinguishing it from
            # ChCl : PTSA is lost.
            row_ratio = ratio_raw
            if not row_ratio:
                row_ratio = ratio_from_coefficients(
                    coefficients, sum(1 for x in (c1, c2, c3) if x))
            if water is not None:
                # The other components are one part each unless the table said otherwise.
                parts = row_ratio.split(":") if row_ratio else \
                    ["1"] * max(0, sum(1 for x in (c1, c2, c3) if x) - 1)
                row_ratio = ":".join(parts + [str(water)])
            mixture_names = ":".join(x for x in (c1, c2, c3) if x)
            suffix = f":{n + 1:02d}" if len(combinations) > 1 else ""
            row_detail = dict(detail)
            if water is not None:
                row_detail["water_equivalents"] = str(water)
            if len(combinations) > 1:
                row_detail["one_of"] = f"{n + 1} of {len(combinations)} alternatives"
            rows.append(ApplicationRow(
                Application_key=f"{paper.slug}:{table.id}:{index:04d}{suffix}",
                Paper_key=paper.key, Paper_DOI=paper.doi,
                Table_id=table.id, Source_row=index,
                Domain=domain,
                Table_caption=table.caption[:300],
                Component_1=c1, Component_2=c2, Component_3=c3,
                Ratio_raw=row_ratio,
                Mixture=f"{mixture_names} ({row_ratio})" if row_ratio else mixture_names,
                Component_flag=component_flag,
                Components_written=";".join(written),
                Implied_components=";".join(profile.implied_components or []),
                Detail=" | ".join(f"{k}={v}" for k, v in row_detail.items())[:500],
                Source_ref_numbers=",".join(str(n) for n in ref_numbers),
                Source_DOIs=cited["doi"], Source_paper_keys=cited["key"],
                Source_titles=cited["title"], Source_years=cited["year"],
            ))
    return rows


def extract_applications(tables, profiles, paper, reference_map):
    """Every des_application table in one paper. -> list[ApplicationRow]."""
    relevant = [t for t in tables
                if (profiles.get(t.id) is not None and profiles[t.id].relevant
                    and profiles[t.id].record_type == "des_application")]
    if not relevant:
        return []

    # Built once for the paper, not once per table. Absent on a first-ever run, before
    # components.csv exists; names then stay as written.
    try:
        vocabulary = component_index()
    except Exception as exc:
        print(f"    no component vocabulary yet ({type(exc).__name__}); "
              f"application components stay as written")
        vocabulary = None

    out = []
    for table in relevant:
        profile = profiles[table.id]
        rows = extract_application_table(table, profile, paper, reference_map,
                                         vocabulary)
        out += rows
        print(f"    {table.label or table.id}: {len(rows)} application row(s) "
              f"[{domain_of(table.caption, table.label)}]")
    return out
