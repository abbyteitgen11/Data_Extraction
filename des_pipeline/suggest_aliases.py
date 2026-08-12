"""
Work out what the paper's abbreviations mean, from the paper's own data.

The prose writes "TBAB:LA (1:4, density 1.1031)". Table 2 contains exactly one row
with density 1.1031 at ratio 1:4, and that row names its components in full. So the
table decodes the abbreviation -- no chemistry knowledge, no guessing.

This is a CANDIDATE GENERATOR, not an oracle. It proposes; a human confirms by
editing component_aliases.json. Some probes match several table rows, and a wrong
alias writes wrong chemistry into the graph that nothing downstream can detect.

Why bother: asking an LLM instead gets roughly half of them wrong, and wrong in a
way that looks right -- it reads TBAB as tetraethylammonium bromide (the paper means
tetrabutyl), Gly as glycine (the paper means glycolic acid), AA as acetic acid (the
table says acetamide).

    python run_pipeline.py --steps aliases
"""
import math

import pandas as pd

from . import config
from .enrich_components import component_index, resolve_component


def _table_rows(path=None):
    """Table 2 as (components, ratio, {property: value}) tuples."""
    df = pd.read_csv(path or config.TABLE_CSV)
    rows = []
    for r in df.to_dict("records"):
        comps = [r.get(f"Component_{i}") for i in (1, 2, 3)]
        comps = [str(c).strip() for c in comps if isinstance(c, str) and str(c).strip()]
        values = {p: r.get(p) for p in config.PROPERTY_NAMES
                  if r.get(p) is not None and not pd.isna(r.get(p))}
        rows.append((r["Row_id"], comps, str(r.get("Ratio_raw") or ""), values))
    return rows


def candidates_for(unresolved_name, prose_rows, table_rows, rel_tol=1e-3):
    """Table components co-occurring with this name's measurements.

    For every prose row mentioning `unresolved_name`, find table rows with the same
    property and (nearly) the same value. Whatever component names those rows
    contribute, ranked by how often they show up, are the candidates.
    """
    tally = {}
    for prose in prose_rows:
        value, prop = prose.get("value"), prose.get("property")
        if value is None or pd.isna(value) or not prop:
            continue
        # NaN is truthy, so an unstated ratio must be tested with pd.isna, not `or ""`
        raw_ratio = prose.get("molar_ratio")
        ratio = "" if raw_ratio is None or pd.isna(raw_ratio) else str(raw_ratio).strip()
        for row_id, comps, table_ratio, values in table_rows:
            table_value = values.get(prop)
            if table_value is None:
                continue
            if not math.isclose(float(table_value), float(value),
                                rel_tol=rel_tol, abs_tol=1e-9):
                continue
            if ratio and table_ratio and ratio != table_ratio:
                continue
            for c in comps:
                tally.setdefault(c, []).append(row_id)
    return sorted(tally.items(), key=lambda kv: -len(kv[1]))


def suggest(path=None, index=None):
    """-> DataFrame of (unresolved name, candidate, evidence) for human review."""
    path = path or config.SECTIONS_LLM_CSV
    if not path.exists():
        print("  no sections_llm.csv yet -- run --steps text first")
        return pd.DataFrame()

    index = index or component_index()
    prose = pd.read_csv(path).to_dict("records")
    table_rows = _table_rows()

    # group prose rows by each unresolved component name they mention
    by_name = {}
    for row in prose:
        names = str(row.get("components") or "").split(";")
        for name in (n.strip() for n in names if n.strip()):
            canonical, _ = resolve_component(name, index)
            if canonical is None:
                by_name.setdefault(name, []).append(row)

    out = []
    for name, rows in sorted(by_name.items(), key=lambda kv: -len(kv[1])):
        cands = candidates_for(name, rows, table_rows)
        # names that already resolve are useless as candidates for THIS name
        cands = [(c, ids) for c, ids in cands
                 if resolve_component(c, index)[0] != name]
        if not cands:
            out.append({"abbreviation": name, "uses": len(rows), "rank": 0,
                        "candidate": "", "evidence_rows": "", "confidence": "none"})
            continue
        top = cands[0][1]
        for rank, (candidate, row_ids) in enumerate(cands[:4], 1):
            out.append({
                "abbreviation": name,
                "uses": len(rows),
                "rank": rank,
                "candidate": candidate,
                "evidence_rows": ",".join(row_ids[:4]),
                # one candidate from one table row is a clean decode; several is a guess
                "confidence": "unique" if len(cands) == 1 and len(top) == 1
                              else ("strong" if rank == 1 and len(top) > len(cands[1][1])
                                    else "ambiguous"),
            })
    return pd.DataFrame(out)


def report(path=None):
    """Print the suggestions and write them to data/alias_suggestions.csv."""
    df = suggest(path)
    if df.empty:
        print("  every prose component name already resolves")
        return df

    unresolved = df.abbreviation.nunique()
    print(f"  {unresolved} unresolved component name(s) in the prose\n")
    print(f"  {'abbreviation':<24}{'uses':>5}  {'candidate from Table 2':<44}{'confidence'}")
    print("  " + "-" * 88)
    for name, group in df.groupby("abbreviation", sort=False):
        for i, row in enumerate(group.itertuples()):
            label = name if i == 0 else ""
            uses = str(row.uses) if i == 0 else ""
            print(f"  {label:<24}{uses:>5}  {row.candidate[:43]:<44}{row.confidence}")
    config.ALIAS_SUGGESTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.ALIAS_SUGGESTIONS_CSV, index=False)
    print(f"\n  wrote {config.ALIAS_SUGGESTIONS_CSV.name}")
    print(f"  confirm the ones you believe in {config.COMPONENT_ALIASES.name}; "
          f"anything left unresolved stays out of the graph")
    return df
