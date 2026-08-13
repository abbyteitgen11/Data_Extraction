"""
Propose abbreviation definitions for a human to confirm into component_aliases.json.

Two independent sources, and the interesting part is where they disagree:

  harvest_definitions   what the paper says about itself -- every "Full name (ABBR)"
                        it writes, matched with the Schwartz-Hearst algorithm
  suggest               what the paper's own data says -- for an abbreviation we
                        cannot resolve, the Table 2 components whose measurements
                        match the prose value

Neither decides anything. The alias file is edited by hand, because a wrong entry
writes wrong chemistry into the graph and nothing downstream can detect it.

The cross-check is what makes this worth running: for `LA` the paper's text says
"Lactic acid" while its own Table 2 says Levulinic acid, and only seeing both tells
you the abbreviation is overloaded and belongs in neither camp.

    python run_pipeline.py --steps aliases
"""
import json
import math
import re

import pandas as pd

from . import config
from .enrich_components import (_CLASS_WORDS, component_index, load_alias_records,
                                resolve_component, resolve_phrase)

# A bracketed candidate abbreviation: "(ChAc)", "(TBPB)", "([BMIM]Br)".
_PAREN = re.compile(r"\(([A-Za-z0-9\[\]'-]{2,14})\)")

# Abbreviations that are not chemicals. Reported separately rather than dropped, so
# the CSV shows what was skipped and why.
NOT_COMPOUNDS = {
    "des", "dess", "hba", "hbas", "hbd", "hbds", "nades", "lttm", "lttms",
    "swot", "swots", "dsc", "tga", "nmr", "ftir", "atr", "uv", "il", "ils",
    "i", "ii", "iii", "iv", "v", "vi", "taax", "taaxs",
}

# A long form containing one of these is a sentence fragment, not a chemical name.
_STOPWORDS = re.compile(
    r"\b(of|in|a|an|the|and|or|is|are|was|were|to|by|with|as|for|that|which|"
    r"results?|such|other|its|their|from|obtained|prepared|known|following)\b", re.I)


def long_form(window, abbr):
    """Schwartz-Hearst: match the abbreviation's letters right-to-left through the
    words before the bracket. -> the long form, or None.

    The subtlety that matters: when a letter is found but not at a word boundary,
    keep scanning further left for another occurrence rather than giving up. Without
    that, PABr, TBPB and TEAB all fail.
    """
    letters = [c.lower() for c in abbr if c.isalnum()]
    if not letters:
        return None
    li, ai = len(window) - 1, len(letters) - 1
    while ai >= 0:
        target = letters[ai]
        while li >= 0:
            same = window[li].lower() == target
            # the abbreviation's first letter must begin a word
            if same and (ai > 0 or li == 0 or window[li - 1] in " -('"):
                break
            li -= 1
        if li < 0:
            return None
        if ai == 0:
            return window[li:].strip()
        li -= 1
        ai -= 1
    return None


def harvest_definitions(sections, review_doi=""):
    """Every "Full name (ABBR)" the paper defines itself. -> list[dict]."""
    out, seen = [], set()
    for section_id, _title, text in sections:
        for match in _PAREN.finditer(text):
            abbr = match.group(1).strip()
            if abbr.lower() in seen:
                continue
            # The window is the words just before the bracket, bounded by the last
            # punctuation so we never reach back into the previous sentence.
            before = text[:match.start()]
            before = re.split(r"[.;:,()]", before)[-1]
            words = before.split()
            window = " ".join(words[-min(len(abbr) + 5, len(abbr) * 2):])
            name = long_form(window, abbr)

            status = "candidate"
            if abbr.lower() in NOT_COMPOUNDS or not re.search(r"[A-Z]", abbr):
                status = "not_a_compound"
            elif not name:
                continue
            elif _STOPWORDS.search(name) or len(name.split()) > 5:
                status = "rejected"
            elif _CLASS_WORDS.search(name):
                status = "not_a_compound"

            seen.add(abbr.lower())
            out.append({
                "abbreviation": abbr,
                "name": (name or "").strip(" ,-"),
                "status": status,
                "section_id": section_id,
                "evidence": re.sub(r"\s+", " ", text[max(0, match.start() - 70):
                                                     match.end() + 10]).strip(),
                "review_doi": review_doi,
            })
    return out


def cross_check(candidates, index=None, prose_path=None):
    """Compare each harvested definition with the alias file and with Table 2.

    `verdict` is the column to read:
      new                  not in the alias file yet -- the ones to consider adding
      agrees               already present, same meaning
      conflicts_with_file  already present with a DIFFERENT meaning
      conflicts_with_table the paper's text and its own data disagree (see LA)
    """
    index = index or component_index()
    records = load_alias_records()
    lowered = {k.lower(): v for k, v in records.items()}
    table_hints = _table_candidates(prose_path, index)

    for row in candidates:
        canonical, _how = resolve_component(row["name"], index)
        row["in_table2"] = bool(canonical)
        row["name_in_table2"] = canonical or ""

        existing = lowered.get(row["abbreviation"].lower())
        hint = table_hints.get(row["abbreviation"].lower(), "")
        row["table_candidate"] = hint

        if row["status"] != "candidate":
            row["verdict"] = row["status"]
        elif hint and canonical and hint.lower() != canonical.lower():
            row["verdict"] = "conflicts_with_table"
        elif existing is None:
            row["verdict"] = "new"
        elif not existing.get("name"):
            row["verdict"] = "deliberately_unresolved"
        elif existing["name"].lower() == row["name"].lower():
            row["verdict"] = "agrees"
        else:
            row["verdict"] = "conflicts_with_file"
    return candidates


# ---------- what the paper's own data says (the value-matching cross-check) ----------
def _table_rows(path=None):
    """Table 2 as (row id, components, ratio, {property: value}) tuples."""
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
    """Table components co-occurring with this name's measurements, most frequent first."""
    tally = {}
    for prose in prose_rows:
        value, prop = prose.get("value"), prose.get("property")
        if value is None or pd.isna(value) or not prop:
            continue
        # NaN is truthy, so an unstated ratio must be tested with pd.isna
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


def _table_candidates(prose_path=None, index=None):
    """{abbreviation: best Table 2 candidate} for names the prose could not resolve."""
    path = prose_path or config.SECTIONS_LLM_CSV
    if not path.exists():
        return {}
    index = index or component_index()
    df = pd.read_csv(path)
    column = "components_written" if "components_written" in df.columns else "components"
    table_rows = _table_rows()

    by_name = {}
    for row in df.to_dict("records"):
        for name in str(row.get(column) or "").split(";"):
            name = name.strip()
            if name and name.lower() != "nan" and resolve_phrase(name, index) is None:
                by_name.setdefault(name, []).append(row)

    out = {}
    for name, rows in by_name.items():
        hits = candidates_for(name, rows, table_rows)
        hits = [(c, ids) for c, ids in hits if resolve_phrase(c, index) != name]
        if hits:
            out[name.lower()] = hits[0][0]
    return out


def suggest(path=None, index=None):
    """Table 2 candidates for every prose component name that still does not resolve."""
    path = path or config.SECTIONS_LLM_CSV
    if not path.exists():
        return pd.DataFrame()
    index = index or component_index()
    df = pd.read_csv(path)
    column = "components_written" if "components_written" in df.columns else "components"
    table_rows = _table_rows()

    by_name = {}
    for row in df.to_dict("records"):
        for name in str(row.get(column) or "").split(";"):
            name = name.strip()
            if name and name.lower() != "nan" and resolve_phrase(name, index) is None:
                by_name.setdefault(name, []).append(row)

    out = []
    for name, rows in sorted(by_name.items(), key=lambda kv: -len(kv[1])):
        hits = candidates_for(name, rows, table_rows)
        hits = [(c, ids) for c, ids in hits if resolve_phrase(c, index) != name]
        if not hits:
            out.append({"abbreviation": name, "uses": len(rows), "rank": 0,
                        "candidate": "", "evidence_rows": ""})
            continue
        for rank, (candidate, row_ids) in enumerate(hits[:3], 1):
            out.append({"abbreviation": name, "uses": len(rows), "rank": rank,
                        "candidate": candidate, "evidence_rows": ",".join(row_ids[:4])})
    return pd.DataFrame(out)


# ---------- reporting ----------
def report(sections=None, review_doi="", path=None):
    """Print both views and write data/alias_candidates.csv."""
    index = component_index()

    if sections:
        found = cross_check(harvest_definitions(sections, review_doi), index, path)
        df = pd.DataFrame(found)
        good = df[df.verdict.isin(["new", "conflicts_with_file", "conflicts_with_table"])]
        print(f"  definitions the paper states itself: {len(df)} bracket hits, "
              f"{(df.status == 'candidate').sum()} look like compounds\n")
        print(f"  {'abbrev':<10}{'name from the text':<42}{'in T2':<7}{'verdict'}")
        print("  " + "-" * 84)
        for r in df[df.status == "candidate"].itertuples():
            print(f"  {r.abbreviation:<10}{r.name[:41]:<42}"
                  f"{'yes' if r.in_table2 else 'no':<7}{r.verdict}")
        skipped = df[df.status != "candidate"]
        if not skipped.empty:
            print(f"\n  skipped as not-a-compound: "
                  f"{', '.join(sorted(skipped.abbreviation))}")
        for r in df[df.verdict == "conflicts_with_table"].itertuples():
            print(f"\n  CONFLICT  {r.abbreviation}: the text says {r.name!r}, but this "
                  f"paper's own Table 2 data says {r.table_candidate!r}.")
            print(f"            Leave it out unless you can tell which usage is which.")
        config.ALIAS_CANDIDATES_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.ALIAS_CANDIDATES_CSV, index=False)
        print(f"\n  wrote {config.ALIAS_CANDIDATES_CSV.name} "
              f"({len(good)} worth reviewing)")

    still = suggest(path, index)
    if not still.empty:
        names = still.abbreviation.nunique()
        print(f"\n  still unresolved after the alias file: {names} name(s)")
        print(f"  {'name':<28}{'uses':>5}  Table 2 candidates (from matching values)")
        print("  " + "-" * 84)
        for name, group in still.groupby("abbreviation", sort=False):
            cands = [g.candidate for g in group.itertuples() if g.candidate]
            uses = group.uses.iloc[0]
            print(f"  {name[:27]:<28}{uses:>5}  {', '.join(cands[:3]) or '-'}")
    print(f"\n  confirm what you believe in {config.COMPONENT_ALIASES.name}; "
          f"anything unresolved stays out of the graph")
    return None
