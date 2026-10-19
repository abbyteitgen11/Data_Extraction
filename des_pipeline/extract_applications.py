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
from .composition import des_problem, is_water, parse_mixture
from .enrich_components import component_index, resolve_component
from .extract_references import sources
from .extract_table import (_is_group_row, _refusal, read_references, row_composition,
                            table_gate, water_columns)
from .schema import ApplicationRow

# Domains live in config.APPLICATION_DOMAINS: a controlled vocabulary, so two papers'
# wordings of the same use land on one node, and an unmatched use is "other" rather than
# a name invented from caption words.


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


def domain_of(*texts):
    """The controlled application domain for some free text. -> (domain, detail).

    Deterministic keyword matching over config.APPLICATION_DOMAINS, first match wins.
    Anything that matches nothing is "other" -- never the first words of a caption,
    which is what produced "summary_some_recent" and "examples_different_liquids".
    The text it was decided from is returned as the detail, so nothing is lost.
    """
    joined = " ".join(str(t or "") for t in texts if t)
    text = f" {joined.lower()} "
    # The keyword that appears FIRST wins, not the first domain in the list: a caption
    # about "polymer membranes for ... heavy metal removal" is about membranes. Ties
    # (same position) go to the more specific domain, which is listed earlier.
    best = None
    for rank, (domain, keywords) in enumerate(config.APPLICATION_DOMAINS):
        hits = [text.find(k) for k in keywords if k in text]
        if hits and (best is None or (min(hits), rank) < best[0]):
            best = ((min(hits), rank), domain)
    return (best[1] if best else "other"), joined.strip()[:200]


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


def extract_application_table(table, profile, paper, reference_map, dictionary=None,
                              resolve=None, strict=""):
    """One des_application table -> (list[ApplicationRow], stats).

    Every row goes through the same composition reader and the same DES test as a
    property table: a row naming one compound (a corrosion inhibitor, an ionic liquid on
    its own, an HMF-separation method) is not a DES and is refused, whatever the table's
    caption says.
    """
    by_role = {}
    for column in profile.columns:
        by_role.setdefault(column.role, []).append(column)
    name_cols = by_role.get("mixture", []) + by_role.get("component", [])
    ref_col = next(iter(by_role.get("reference", [])), None)
    water_ids = {c.index for c in water_columns(profile, table)}
    detail_cols = [c for c in by_role.get("context", []) + by_role.get("other_quantity", [])
                   + by_role.get("property", []) if c.index not in water_ids]

    domain, domain_detail = domain_of(table.caption)
    rows, refused = [], []
    ragged = {index for index, _ in table.ragged}

    for index, row in enumerate(table.rows):
        if index in ragged or _is_group_row(row, table):
            continue
        names = [row[c.index].text for c in name_cols if c.index < len(row)]
        names = [n for n in names if n and n.strip() and n.strip() not in config.DASH]
        if not names:
            continue                              # a spacer or a continuation row

        ref_numbers = []
        if ref_col is not None and ref_col.index < len(row):
            ref_numbers = read_references(row[ref_col.index], reference_map)
        cited = sources(ref_numbers, reference_map, paper.key)

        detail = {}
        for column in detail_cols:
            if column.index >= len(row):
                continue
            text = row[column.index].text.strip()
            if text and text not in config.DASH:
                key = column.context_field or column.property or f"col{column.index}"
                detail[key] = text

        # One row per combination of the alternatives each NAME cell lists ("MeOH,
        # BuOH, HexOH" is three experiments). Water equivalents join the product.
        waters = water_equivalents(detail) or [None]
        choices = [alternatives(n) for n in names] + [waters]
        combinations = [(c[:-1], c[-1]) for c in itertools.product(*choices)]
        if len(combinations) > MAX_ALTERNATIVES:
            combinations = combinations[:MAX_ALTERNATIVES]
            detail["alternatives_truncated"] = "yes"

        for n, (chosen, water) in enumerate(combinations):
            fake_row = list(row)
            for column, value in zip([c for c in name_cols if c.index < len(row)
                                      and row[c.index].text.strip()
                                      and row[c.index].text.strip() not in config.DASH],
                                     chosen):
                fake_row[column.index] = type(row[column.index])(
                    text=value, markers=row[column.index].markers,
                    ref_ids=row[column.index].ref_ids)
            comp, _flag = row_composition(fake_row, table, profile, dictionary, resolve)
            if water is not None and not any(is_water(c) for c in comp.components):
                comp.components.append("Water")
                comp.components_written.append("H2O")
                if comp.parts:
                    comp.parts = list(comp.parts) + [float(water)]
                    comp.ratio_raw = ":".join(f"{p:g}" for p in comp.parts if p is not None)
                    comp.ratio_molar = comp.ratio_raw
            problem = _refusal(comp, dictionary, strict)
            if problem:
                refused.append({"Table_id": table.id, "Source_row": index,
                                "reason": f"not extracted: {problem}",
                                "raw": " | ".join(chosen)[:200]})
                continue
            (c1, c2, c3), _parts, component_flag = comp.slots()
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
                Domain=domain, Domain_detail=domain_detail,
                Table_caption=table.caption[:300],
                Component_1=c1, Component_2=c2, Component_3=c3,
                Ratio_raw=comp.ratio_raw,
                Mixture=comp.label(),
                Component_flag=component_flag,
                Components_written=";".join(comp.components_written),
                Implied_components=";".join(profile.implied_components or []),
                Detail=" | ".join(f"{k}={v}" for k, v in row_detail.items())[:500],
                DES_code=comp.code, Ratio_molar=comp.ratio_molar,
                Water_fraction=comp.water_fraction,
                Source_ref_numbers=",".join(str(x) for x in ref_numbers),
                Source_DOIs=cited["doi"], Source_paper_keys=cited["key"],
                Source_titles=cited["title"], Source_years=cited["year"],
            ))
    stats = {"n_applications": len(rows), "n_refused": len(refused),
             "why": refused[0]["reason"] if refused and not rows else ""}
    return rows, refused, stats


def extract_applications(tables, profiles, paper, reference_map, dictionary=None,
                         resolve=None, paragraphs=None):
    """Every des_application table in one paper. -> (rows, refused, {table id: stats})."""
    relevant = [t for t in tables
                if (profiles.get(t.id) is not None and profiles[t.id].relevant
                    and "des_application" in profiles[t.id].content_types)]
    out, refused_all, stats = [], [], {}
    for table in relevant:
        profile = profiles[table.id]
        strict, skip = table_gate(table, profile, paper, dictionary, paragraphs)
        if skip:
            stats[table.id] = {"n_applications": 0, "n_refused": 1, "why": skip}
            print(f"    {table.label or table.id}: skipped -- {skip}")
            continue
        rows, refused, stat = extract_application_table(
            table, profile, paper, reference_map, dictionary, resolve, strict)
        out += rows
        refused_all += [{**r, "Paper_key": paper.key} for r in refused]
        stats[table.id] = stat
        print(f"    {table.label or table.id}: {len(rows)} application row(s) "
              f"[{domain_of(table.caption)[0]}]"
              f"{f', {len(refused)} refused (not a DES)' if refused else ''}")
    return out, refused_all, stats


def paper_applications(overview, dictionary, paper, source_text, resolve=None):
    """The paper-level uses from the overview call. -> list[ApplicationRow].

    "What is this paper's DES used for" is usually answered by the title and abstract,
    not by a table: every DES in Azad is used to dissolve fluconazole, and no table says
    so. Each claim is kept only if its evidence is in the text the model read and each
    DES it names resolves to one the paper defines (its dictionary) -- a DES the model
    names that the paper does not is the hallucination this route must not admit.
    """
    from .textmatch import quote_found

    out = []
    if overview is None:
        return out
    for n, draft in enumerate(overview.applications, 1):
        found = quote_found(draft.evidence, source_text)
        domain, detail = domain_of(draft.use, draft.target, draft.evidence)
        for m, name in enumerate(draft.des or ["?"], 1):
            comp = dictionary.lookup(name) if dictionary is not None else None
            if comp is None:
                comp = parse_mixture(name, dictionary, resolve)
                known = dictionary is not None and dictionary.matches(comp)
                status = "ok" if known else "not_in_dictionary"
            else:
                status = "ok"
            if not found:
                status = "unverified"
            elif status == "ok" and des_problem(comp, dictionary):
                status = "not_des"
            (c1, c2, c3), _parts, flag = comp.slots()
            out.append(ApplicationRow(
                Application_key=f"{paper.slug}:paper:{n:03d}:{m:02d}",
                Paper_key=paper.key, Paper_DOI=paper.doi,
                Domain=domain, Domain_detail=detail,
                Component_1=c1, Component_2=c2, Component_3=c3,
                Ratio_raw=comp.ratio_raw, Mixture=comp.label(), Component_flag=flag,
                Components_written=";".join(comp.components_written) or name,
                Detail=" | ".join(f"{k}={v}" for k, v in (("use", draft.use),
                                                          ("role", draft.role),
                                                          ("target", draft.target)) if v)[:500],
                Extractor="paper", Role=(draft.role or "")[:80],
                Target=(draft.target or "")[:200],
                source_text=draft.evidence[:300], quote_found=found, status=status,
                DES_code=comp.code, Ratio_molar=comp.ratio_molar,
                Water_fraction=comp.water_fraction,
                Source_paper_keys=paper.key, Source_DOIs=paper.doi,
            ))
    return out


def _same_des(a, b):
    """Same set of non-water components (ratio ignored). -> bool."""
    def key(comp):
        return frozenset(re.sub(r"\W+", "", c).lower() for c in comp.components
                         if c and not is_water(c))
    return bool(key(a)) and key(a) == key(b)
