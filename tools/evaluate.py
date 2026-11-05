#!/usr/bin/env python3
"""
Score runs against the hand-reviewed gold set, write side-by-side comparison files, and
report tokens, time, cost (USD and EUR) and a full-corpus cost estimate.

    python tools/evaluate.py --tag qwen3                      # one run
    python tools/evaluate.py --tag qwen3 --tag sonnet55       # side by side
    python tools/evaluate.py --tag qwen3 --self-test          # gold vs gold must be 100%

Outputs:
    eval/results/<tag>.json             every number, per paper and pooled
    eval/gold_comparison/<paper>/<task>.csv
                                        each gold row, its verdict, and what every model
                                        predicted for it; then every prediction that matched
                                        no gold row ("extra")
    eval/gold_comparison/summary.csv    one row per (model, task)
    eval/gold_comparison/REPORT.md      the comparison, readable

Scoring (gold rules are in tools/gold_rules.py):
  * gold = rows marked correct / fix / added. Rows marked wrong are known errors: a model
    predicting one is charged an error. Blank rows are unreviewed and not scored.
  * a prediction that matches no gold row is an EXTRA. Extras from tables (table values,
    mixtures, table-route applications) count as errors, because tables were reviewed in
    full. Extras from prose, from the paper-level overview, and in the DES dictionary are
    NOT scored either way -- the papers' text was not read exhaustively -- and are listed
    for review instead.
  * a value matches when the DES (component set, order-free; ratio compared when both
    sides state one), property, value (0.5%) and temperature (0.5 C; for table values a
    missing temperature is a mismatch) agree. "Right DES, property and temperature, wrong
    value" is counted separately.
  * 95% intervals resample papers (bootstrap), because 12 papers is a small sample.
"""
import argparse
import csv
import json
import math
import os
import random
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import gold_rules as rules  # noqa: E402

GOLD_DIR = Path(os.environ.get("DES_GOLD_DIR") or ROOT / "eval" / "gold")
COMPARE_DIR = ROOT / "eval" / "gold_comparison"
RESULTS_DIR = ROOT / "eval" / "results"
TASKS = ("tables", "columns", "dictionary", "mixtures", "measurements", "applications")
GOLD_FILE = {"dictionary": "des_dictionary"}


# ---------- reading ----------
def _read(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _float(x):
    try:
        if x in (None, "", "nan", "None"):
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


_ALIASES = None


def _global_aliases():
    global _ALIASES
    if _ALIASES is None:
        try:
            from des_pipeline.enrich_components import load_aliases

            _ALIASES = {k.lower(): v.lower() for k, v in load_aliases().items()}
        except Exception:                                    # noqa: BLE001
            _ALIASES = {}
    return _ALIASES


_CID = None


def _cid_index():
    """name / synonym (lower case) -> PubChem CID, from the shared lookup cache.

    Only names of 5+ characters, or chemical formulas, are indexed: bare abbreviations
    resolve to the wrong compound in PubChem ("U" is uranium, "Gly" glycine), so those
    must come from the paper's own definitions or the alias file instead.
    """
    global _CID
    if _CID is None:
        _CID = {}
        try:
            cache = json.loads((ROOT / "data" / "components_cache.json").read_text())
        except (OSError, ValueError):
            cache = {}
        for name, row in cache.items():
            if not row or row.get("lookup_status") != "ok" or not row.get("cid"):
                continue
            cid = int(row["cid"])
            for text in [name, row.get("matched_name") or ""] + \
                    str(row.get("synonyms") or "").split(";"):
                text = text.strip().lower()
                if _indexable(text):
                    _CID.setdefault(text, cid)
    return _CID


def _indexable(text):
    return len(text) >= 5 or bool(re.search(r"[a-z]\d|\d[a-z]", text))


def norm_name(name, aliases):
    """One component name -> a comparable identity: 'water', 'cid:<n>', or letters."""
    key = str(name or "").strip().lower()
    if key in ("water", "h2o", "h₂o"):
        return "water"
    key = aliases.get(key, key)
    key = _global_aliases().get(key, key)
    if _indexable(key) and key in _cid_index():
        return f"cid:{_cid_index()[key]}"
    squeezed = re.sub(r"[^a-z0-9]", "", key)
    # "1-butyl-3-methyl-imidazolium chloride" when PubChem knows only the spelling
    # without the second hyphen: the same name once punctuation is ignored.
    if len(squeezed) >= 5 and squeezed in _squeezed_cid_index():
        return f"cid:{_squeezed_cid_index()[squeezed]}"
    return squeezed


_SQUEEZED_CID = None


def _squeezed_cid_index():
    global _SQUEEZED_CID
    if _SQUEEZED_CID is None:
        _SQUEEZED_CID = {}
        for text, cid in _cid_index().items():
            _SQUEEZED_CID.setdefault(re.sub(r"[^a-z0-9]", "", text), cid)
    return _SQUEEZED_CID


def identity(names, ratio, aliases):
    """-> (frozenset of names, {name: part scaled so min = 1} or None)."""
    names = [norm_name(n, aliases) for n in names if str(n).strip()]
    names = [n for n in names if n]
    parts = []
    for p in re.split(r"\s*:\s*", str(ratio or "").strip()):
        value = _float(p)
        if value is None:
            parts = []
            break
        parts.append(value)
    ratio_map = None
    if parts and len(parts) == len(names) and min(parts) > 0:
        low = min(parts)
        ratio_map = frozenset((n, round(p / low, 2)) for n, p in zip(names, parts))
    return frozenset(names), ratio_map


def same_des(a, b):
    """Same component set; the ratio must agree when BOTH sides state one."""
    if not a[0] or a[0] != b[0]:
        return False
    if a[1] is None or b[1] is None:
        return True
    return a[1] == b[1]


def same_value(a, b, rel=0.005):
    a, b = _float(a), _float(b)
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=rel, abs_tol=1e-9)


def same_temperature(a, b, strict, tol=0.5):
    a, b = _float(a), _float(b)
    if a is None or b is None:
        return (a is None and b is None) or not strict
    return abs(a - b) <= tol


def code_key(text):
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def content_types(text):
    out = set()
    for t in re.split(r"[+;,\s]+", str(text or "").lower()):
        t = {"des_applications": "des_application", "des_property": "des_properties",
             "des_definition": "des_definitions"}.get(t, t)
        if t and t != "other":
            out.add(t)
    return out


# ---------- gold ----------
def load_gold(slug):
    """-> ({task: {"real": [...], "wrong": [...], "unreviewed": n}}, meta, aliases)."""
    folder = GOLD_DIR / slug
    meta = json.loads((folder / "meta.json").read_text()) if (folder / "meta.json").exists() \
        else {}
    aliases = {}
    for r in _read(folder / "des_dictionary.csv"):
        if r.get("kind") == "alias" and rules.verdict(r) in rules.REAL:
            aliases[str(r.get("code")).strip().lower()] = \
                str(r.get("gold_components")).strip().lower()

    # Your reviewed dictionary also says what each abbreviation in a code means: a code
    # "ChCl:Gly (1:2)" reviewed as "Choline chloride;Glycerol" defines Gly = glycerol.
    for r in _read(folder / "des_dictionary.csv"):
        if r.get("kind") != "mixture" or rules.verdict(r) not in rules.REAL:
            continue
        code = re.sub(r"\([^)]*\)", "", str(r.get("code") or "")).strip()
        written = [t.strip() for t in re.split(r"\s*[:/]\s*", code) if t.strip()]
        full, _ = rules.components(r.get("gold_components"))
        full = [c for c in full if c.lower() != "water"]
        if len(written) == len(full) >= 2:
            for abbr, name in zip(written, full):
                if abbr.lower() != name.lower():
                    aliases.setdefault(abbr.lower(), name.lower())

    out = {}
    for task in TASKS:
        if (slug, task) in rules.IGNORED_FILES:
            continue
        rows = _read(folder / f"{GOLD_FILE.get(task, task)}.csv")
        bucket = {"real": [], "wrong": [], "unreviewed": 0, "rows": rows}
        for i, r in enumerate(rows):
            v = rules.verdict(r, task)
            if not v:
                bucket["unreviewed"] += 1
                continue
            item = gold_item(task, r, aliases)
            if item is None:
                continue
            item["_row"] = i
            item["_verdict"] = v
            (bucket["wrong"] if v == "wrong" else bucket["real"]).append(item)
        if task == "columns":
            # Only columns with a known label are scored ("wrong" = table not relevant).
            bucket["wrong"] = []
        out[task] = bucket
    return out, meta, aliases


def gold_item(task, r, aliases):
    if task == "tables":
        return {"table_id": r["table_id"], "relevant": str(r["gold_relevant"]).strip().lower(),
                "types": content_types(r["gold_content_types"])}
    if task == "columns":
        return {"table_id": r["table_id"], "column": str(r["column"]),
                "role": str(r["gold_role"]).strip(),
                "property": rules.property_name(r["gold_property"])}
    names, parsed_ratio = rules.components(r.get("gold_components"))
    ratio = r.get("gold_ratio") or parsed_ratio
    des = identity(names, ratio, aliases)
    if task == "dictionary":
        if r.get("kind") == "alias":
            return None
        return {"code": code_key(r.get("code")), "des": des}
    if task == "mixtures":
        return {"des": des}
    if task == "applications":
        return {"des": des, "domain": str(r.get("gold_domain") or "").strip(),
                "origin": r.get("extractor") or "table"}
    if task == "measurements":
        return {"des": des, "property": rules.property_name(r.get("gold_property")),
                "value": r.get("gold_value"), "temperature": r.get("gold_temperature_C"),
                "origin": (r.get("origin") or "table").strip() or "table"}
    return None


# ---------- predictions ----------
def _comps(r):
    return [str(r.get(f"Component_{i}")) for i in (1, 2, 3)
            if r.get(f"Component_{i}") not in (None, "", "nan")]


def load_pred(tag, slug, meta, aliases):
    src = ROOT / "runs" / tag / "data" / "papers" / slug
    out = {t: [] for t in TASKS}
    for r in _read(src / "tables.csv"):
        # "Relevant" in the gold means "should be extracted", so the prediction is
        # whether anything WAS extracted from the table -- a table the model called DES
        # data but that yielded nothing (no_rows, failed) was not extracted.
        relevant = "yes" if r["status"] == "extracted" else "no"
        out["tables"].append({"table_id": r["table_id"], "relevant": relevant,
                              "types": content_types(r["content_types"]) if relevant == "yes"
                              else set(), "show": f"{r['label']}: {r['status']} "
                                                  f"({r['content_types']})"})
    profiles_path = src / "table_profiles.json"
    profiles = json.loads(profiles_path.read_text()) if profiles_path.exists() else {}
    for table_id, record in profiles.items():
        for c in record.get("profile", {}).get("columns", []):
            out["columns"].append({"table_id": table_id, "column": str(c.get("index")),
                                   "role": c.get("role") or "",
                                   "property": c.get("property") or "",
                                   "show": f"{c.get('role')} {c.get('property') or ''}"})
    for r in _read(src / "des_dictionary.csv"):
        if r.get("kind") != "mixture" or str(r.get("verified")).lower() in ("false", "0"):
            continue
        names = [c for c in str(r.get("components") or "").split(";") if c]
        out["dictionary"].append({
            "code": code_key(r.get("code")),
            "des": identity(names, r.get("ratio_molar") or r.get("ratio_raw"), aliases),
            "origin": "text", "show": f"{r.get('code')} = {r.get('components')} "
                                      f"({r.get('ratio_molar') or r.get('ratio_raw')})"})
    scope = set(meta.get("rows") or []) if meta.get("measurement_scope") == "sampled_rows" \
        else None

    def in_scope(r):
        if scope is None:
            return True
        return f"{r.get('Table_id')}:{int(float(r.get('Source_row') or 0))}" in scope

    mixtures = _read(src / "mixtures.csv")
    by_row = {r["Row_id"]: r for r in mixtures}
    seen = set()
    for r in mixtures:
        if not in_scope(r):
            continue
        ratio = r.get("Ratio_molar") or r.get("Ratio_raw") or ""
        key = (tuple(sorted(_comps(r))), ratio)
        if key in seen:
            continue
        seen.add(key)
        out["mixtures"].append({"des": identity(_comps(r), ratio, aliases), "origin": "table",
                                "show": r.get("Mixture", "")})
    for r in _read(src / "measurements.csv"):
        if (r.get("status") or "ok") != "ok" or not in_scope(r):
            continue
        mix = by_row.get(r["Row_id"], {})
        out["measurements"].append({
            "des": identity(_comps(mix), mix.get("Ratio_molar") or mix.get("Ratio_raw"),
                            aliases),
            "property": r["Property"], "value": r["Value"],
            "temperature": r.get("Temperature_C"), "origin": "table",
            "literature": str(r.get("Literature_value")).lower() in ("true", "1"),
            "show": f"{mix.get('Mixture', '')} {r['Property']}={r['Value']} {r.get('Unit') or ''}"
                    f" @{r.get('Temperature_C') or '-'}C [{r['Table_id']} row "
                    f"{r['Source_row']}]",
            "fields": {"gold_components": ";".join(_comps(mix)),
                       "gold_ratio": mix.get("Ratio_molar") or mix.get("Ratio_raw") or "",
                       "gold_property": r["Property"], "gold_value": r["Value"],
                       "gold_unit": r.get("Unit") or "",
                       "gold_temperature_C": r.get("Temperature_C") or "",
                       "evidence": f"{r['Table_id']} row {r['Source_row']}"}})
    for r in _read(src / "sections_llm.csv"):
        if r.get("status") != "ok":
            continue
        names = [c for c in str(r.get("components_resolved") or "").split(";") if c]
        out["measurements"].append({
            "des": identity(names, r.get("molar_ratio"), aliases),
            "property": r.get("property"), "value": r.get("value"),
            "temperature": r.get("temperature_C"), "origin": "prose",
            "show": f"{r.get('Mixture') or r.get('components_resolved')} {r.get('property')}"
                    f"={r.get('value')} {r.get('unit') or ''} | \"{r.get('source_text', '')[:90]}\"",
            "fields": {"gold_components": r.get("components_resolved", ""),
                       "gold_ratio": r.get("molar_ratio") or "",
                       "gold_property": r.get("property"), "gold_value": r.get("value"),
                       "gold_unit": r.get("unit") or "",
                       "gold_temperature_C": r.get("temperature_C") or "",
                       "evidence": r.get("source_text", "")[:300], "origin": "prose"}})
    for r in _read(src / "applications.csv"):
        if (r.get("status") or "ok") != "ok":
            continue
        origin = r.get("Extractor") or "table"
        out["applications"].append({
            "des": identity(_comps(r), r.get("Ratio_molar") or r.get("Ratio_raw"), aliases),
            "domain": r.get("Domain", ""), "origin": origin,
            "show": f"[{origin}] {r.get('Mixture')} -> {r.get('Domain')} "
                    f"| {(r.get('source_text') or r.get('Detail') or '')[:90]}",
            "fields": {"gold_components": ";".join(_comps(r)),
                       "gold_ratio": r.get("Ratio_molar") or r.get("Ratio_raw") or "",
                       "gold_domain": r.get("Domain", ""),
                       "domain_detail": r.get("Domain_detail", ""),
                       "evidence": (r.get("source_text") or r.get("Detail") or "")[:300],
                       "extractor": origin}})
    for task in ("dictionary", "mixtures"):
        for p in out[task]:
            p.setdefault("fields", {"gold_components": "", "evidence": p["show"]})
    return out


# ---------- matching ----------
def _equal(task):
    if task == "dictionary":
        # Same DES, and the codes agree when both have one -- codes are written in
        # different styles ("CEM5" vs "CEM 5"), so a differing code still matches when
        # both sides state a ratio and it agrees.
        return lambda p, g: same_des(p["des"], g["des"]) and (
            not p["code"] or not g["code"] or p["code"] == g["code"]
            or (p["des"][1] is not None and g["des"][1] is not None))
    if task == "mixtures":
        return lambda p, g: same_des(p["des"], g["des"])
    if task == "applications":
        return lambda p, g: p["domain"] == g["domain"] and same_des(p["des"], g["des"])
    if task == "measurements":
        return lambda p, g: (p["property"] == g["property"]
                             and same_value(p["value"], g["value"])
                             and same_temperature(p["temperature"], g["temperature"],
                                                  strict=g.get("origin") == "table")
                             and same_des(p["des"], g["des"]))
    raise ValueError(task)


def _greedy(preds, golds, equal, used_gold=None):
    """One-to-one greedy matching. -> ({pred index: gold index}, used gold set)."""
    used = set(used_gold or ())
    pairs = {}
    for i, p in enumerate(preds):
        for j, g in enumerate(golds):
            if j not in used and equal(p, g):
                pairs[i] = j
                used.add(j)
                break
    return pairs, used


def scored_extra(task, pred):
    """Does an unmatched prediction count as an error? (see the module docstring)"""
    if task == "dictionary":
        return False
    if task in ("measurements", "applications"):
        # A literature value reprinted in a research paper's table is kept in the data
        # but was left out of the gold, so it is listed for review, not counted.
        return pred.get("origin") == "table" and not pred.get("literature")
    return True                                   # mixtures: from tables


def score_task(task, pred, gold):
    """-> (counts, per-gold-row results, extras)."""
    real, wrong = gold["real"], gold["wrong"]
    results = {}
    if task == "tables":
        by_id = {p["table_id"]: p for p in pred}
        correct = 0
        tp = fp = fn = 0
        for g in real:
            p = by_id.get(g["table_id"], {"relevant": "no", "types": set(), "show": "(none)"})
            ok = p["relevant"] == g["relevant"]
            correct += ok
            tp += len(g["types"] & p["types"])
            fp += len(p["types"] - g["types"])
            fn += len(g["types"] - p["types"])
            results[g["_row"]] = ("correct" if ok else "wrong", p["show"])
        return ({"gold": len(real), "correct": correct, "errors": len(real) - correct,
                 "missed": 0, "extras_unscored": 0, "types_tp": tp, "types_fp": fp,
                 "types_fn": fn}, results, [])
    if task == "columns":
        by_key = {(p["table_id"], p["column"]): p for p in pred}
        correct = 0
        for g in real:
            p = by_key.get((g["table_id"], g["column"]))
            ok = bool(p) and p["role"] == g["role"] and (p["property"] or "") == \
                (g["property"] or "")
            correct += ok
            results[g["_row"]] = ("correct" if ok else "wrong",
                                  p["show"] if p else "(table not profiled)")
        return ({"gold": len(real), "correct": correct, "errors": len(real) - correct,
                 "missed": 0, "extras_unscored": 0}, results, [])

    equal = _equal(task)
    pairs, used = _greedy(pred, real, equal)
    for i, j in pairs.items():
        results[real[j]["_row"]] = ("correct", pred[i]["show"])
    rest = [i for i in range(len(pred)) if i not in pairs]
    known_wrong, _ = _greedy([pred[i] for i in rest], wrong, equal)
    known_wrong = {rest[k]: v for k, v in known_wrong.items()}
    for i, j in known_wrong.items():
        results[wrong[j]["_row"]] = ("predicted_known_wrong", pred[i]["show"])
    for j, g in enumerate(wrong):
        results.setdefault(g["_row"], ("not_predicted", ""))
    rest = [i for i in rest if i not in known_wrong]

    # Right DES, property and temperature, wrong value: a value-reading error.
    wrong_value = {}
    if task == "measurements":
        missed = [j for j in range(len(real)) if j not in used]
        for i in rest:
            p = pred[i]
            for j in missed:
                g = real[j]
                if j in wrong_value.values():
                    continue
                if p["property"] == g["property"] and same_des(p["des"], g["des"]) and \
                        same_temperature(p["temperature"], g["temperature"], True):
                    wrong_value[i] = j
                    results[g["_row"]] = ("wrong_value", p["show"])
                    break
    rest = [i for i in rest if i not in wrong_value]
    for j, g in enumerate(real):
        results.setdefault(g["_row"], ("missed", ""))

    extras = [pred[i] for i in rest]
    scored = [p for p in extras if scored_extra(task, p)]
    counts = {"gold": len(real), "correct": len(pairs),
              "errors": len(known_wrong) + len(wrong_value) + len(scored),
              "known_wrong_predicted": len(known_wrong), "wrong_value": len(wrong_value),
              "extras_scored": len(scored),
              "missed": len(real) - len(pairs),
              "extras_unscored": len(extras) - len(scored)}
    return counts, results, extras


# ---------- pooling ----------
def metrics(c):
    correct, errors, gold = c.get("correct", 0), c.get("errors", 0), c.get("gold", 0)
    p = correct / (correct + errors) if correct + errors else None
    r = correct / gold if gold else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if (p == 0 or r == 0) else None)
    return p, r, f


def pooled(per_paper, task, slugs=None):
    total = Counter()
    for slug in (slugs if slugs is not None else per_paper):
        total.update({k: v for k, v in per_paper[slug].get(task, {}).items()
                      if isinstance(v, (int, float))})
    return dict(total)


def bootstrap(per_paper, task, n=2000, seed=0):
    slugs = [s for s in per_paper if per_paper[s].get(task, {}).get("gold")]
    if len(slugs) < 2:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(n):
        sample = [rng.choice(slugs) for _ in slugs]
        f = metrics(pooled(per_paper, task, sample))[2]
        if f is not None:
            values.append(f)
    if not values:
        return None
    values.sort()
    return values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]


# ---------- cost, time, currency ----------
def eur_per_usd():
    """ECB daily reference rate, fetched once and cached; DES_EUR_PER_USD overrides."""
    if os.environ.get("DES_EUR_PER_USD"):
        return float(os.environ["DES_EUR_PER_USD"]), "DES_EUR_PER_USD"
    cache = ROOT / "eval" / "fx_rate.json"
    import time as _time

    today = _time.strftime("%Y-%m-%d")
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved.get("fetched") == today:
            return saved["eur_per_usd"], saved["source"]
    try:
        with urllib.request.urlopen(
                "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml",
                timeout=10) as response:
            xml = response.read().decode()
        usd = float(re.search(r"currency='USD' rate='([\d.]+)'", xml).group(1))
        date = re.search(r"time='([\d-]+)'", xml).group(1)
        rate, source = 1 / usd, f"ECB reference rate {date} (1 EUR = {usd} USD)"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"fetched": today, "eur_per_usd": rate,
                                     "source": source}))
        return rate, source
    except Exception as exc:                                 # noqa: BLE001
        if cache.exists():
            saved = json.loads(cache.read_text())
            return saved["eur_per_usd"], saved["source"] + " (cached; fetch failed)"
        return 0.92, f"fallback 0.92 (ECB fetch failed: {type(exc).__name__})"


def call_kind(call):
    call = str(call)
    if call.startswith("tbl-"):
        return "table"
    if call.startswith("paper-"):
        return "overview"
    if call.startswith("comp-"):
        return "component"
    return "section"


def usage(tag):
    rows = _read(ROOT / "runs" / tag / "usage.csv")
    by_kind = defaultdict(lambda: Counter())
    for r in rows:
        k = by_kind[call_kind(r.get("call"))]
        k["calls"] += 1
        k["input_tokens"] += int(float(r.get("input_tokens") or 0))
        k["output_tokens"] += int(float(r.get("output_tokens") or 0))
        k["seconds"] += float(r.get("seconds") or 0)
        k["cost_usd"] += float(r.get("cost_usd") or 0)
    timing = _read(ROOT / "runs" / tag / "timing.csv")
    return rows, dict(by_kind), timing


def cost_time(tag, reference_counts=None):
    """Tokens, time and cost of a run. For qwen3, time is per-call-type seconds x the
    clean run's call counts, because its logs include interrupted and repeated calls."""
    rows, by_kind, timing = usage(tag)
    extraction = {k: v for k, v in by_kind.items() if k != "component"}
    total = Counter()
    for v in extraction.values():
        total.update(v)
    out = {"calls": total["calls"], "input_tokens": total["input_tokens"],
           "output_tokens": total["output_tokens"], "cost_usd": round(total["cost_usd"], 4),
           "model_seconds": round(total["seconds"], 1),
           "wall_seconds": round(sum(_float(r.get("seconds")) or 0 for r in timing), 1),
           "by_kind": {k: dict(v) for k, v in extraction.items()},
           "models": sorted({r.get("model") for r in rows if r.get("model")}),
           "truncated_calls": sum(1 for r in rows if r.get("stop_reason") in
                                  ("length", "max_tokens"))}
    if tag.startswith("qwen") and reference_counts:
        est = 0.0
        for kind, n in reference_counts.items():
            v = by_kind.get(kind)
            if v and v["calls"]:
                est += n * v["seconds"] / v["calls"]
        out["model_seconds_logged"] = out["model_seconds"]
        out["model_seconds"] = round(est, 1)
        out["wall_seconds"] = round(est, 1)
        out["time_note"] = ("estimated: seconds per call type x the calls a clean run "
                            "makes (the logs include interrupted and repeated calls)")
    return out


def corpus_estimate(tag_tokens, eur):
    """Tokens and EUR for the ~2,114 title/abstract papers, per Anthropic model."""
    from des_pipeline import config

    corpus = _read(ROOT / "data" / "review" / "pmc_corpus.csv")
    tiab = [r for r in corpus if r.get("phrase_in") == "title_abstract"]
    n_papers = len(tiab) or 2114
    n_tables = int(sum(_float(r.get("n_tables")) or 0 for r in tiab)) or 6862
    sections_per_paper = tag_tokens["_sections_per_paper"]
    profiled_share = tag_tokens["_profiled_share"]
    counts = {"table": n_tables * profiled_share, "overview": n_papers,
              "section": n_papers * sections_per_paper}
    out = {"papers": n_papers, "tables": n_tables, "calls": {k: round(v) for k, v in
                                                             counts.items()}}
    per_model = {}
    for model, (p_in, p_out) in config.ANTHROPIC_PRICES.items():
        if model == "claude-fable-5-1":
            continue
        source = tag_tokens.get(model) or tag_tokens.get("claude-sonnet-5-5") \
            or next((v for k, v in tag_tokens.items() if not k.startswith("_")), None)
        if not source:
            continue
        tin = sum(counts[k] * source[k]["in"] for k in counts if k in source)
        tout = sum(counts[k] * source[k]["out"] for k in counts if k in source)
        usd = (tin * p_in + tout * p_out) / 1e6
        per_model[model] = {
            "input_tokens_M": round(tin / 1e6, 1), "output_tokens_M": round(tout / 1e6, 1),
            "usd": round(usd, 0), "eur": round(usd * eur, 0),
            "eur_batch_api": round(usd * eur * 0.5, 0),
            "tokens_from": "measured" if model in tag_tokens else
            "measured on another model (same prompts; tokenizer and thinking differ)"}
    out["per_model"] = per_model
    return out


# ---------- driver ----------
def evaluate(tags, write=True, self_test=False):
    slugs = sorted(p.name for p in GOLD_DIR.iterdir() if p.is_dir()) \
        if GOLD_DIR.exists() else []
    scored = [s for s in slugs if s not in rules.EXCLUDED_PAPERS]
    results = {tag: {"per_paper": {}, "coverage": {}} for tag in tags}
    comparison = {}                    # (slug, task) -> rows to write
    for slug in slugs:
        gold, meta, aliases = load_gold(slug)
        # Abbreviations the paper itself defines (glossary, "Full name (ABBR)", its
        # definitions tables), as harvested in every run being compared -- the same
        # text for each model, so the same mapping for each.
        for tag in tags:
            for r in _read(ROOT / "runs" / tag / "data" / "papers" / slug /
                           "des_dictionary.csv"):
                if r.get("kind") == "alias" and r.get("source") in (
                        "glossary", "text", "definitions_table", "legend"):
                    aliases.setdefault(str(r.get("code")).strip().lower(),
                                       str(r.get("components")).strip().lower())
        preds = {}
        for tag in tags:
            if self_test:
                preds[tag] = {t: [dict(g, show="gold", origin=g.get("origin", "table"),
                                       fields={}) for g in gold.get(t, {}).get("real", [])]
                              for t in TASKS}
            else:
                preds[tag] = load_pred(tag, slug, meta, aliases)
        if slug not in scored:
            if write and not self_test:
                write_predictions_only(slug, preds)
            continue
        for task in TASKS:
            if task not in gold:
                continue
            rows = [dict(r) for r in gold[task]["rows"]]
            for tag in tags:
                counts, per_row, extras = score_task(task, preds[tag].get(task, []),
                                                     gold[task])
                results[tag]["per_paper"].setdefault(slug, {})[task] = counts
                for i, row in enumerate(rows):
                    res, shown = per_row.get(i, ("unreviewed", ""))
                    row[f"{tag}_result"] = res
                    row[f"{tag}_prediction"] = shown
                for p in extras:
                    extra = {k: "" for k in rows[0]} if rows else {}
                    extra.update(p.get("fields", {}))
                    extra["verdict"] = "extra"
                    extra["note"] = f"found by {tag}; " + (
                        "scored as an error" if scored_extra(task, p) else
                        "NOT scored -- review: add as 'added' or 'wrong'")
                    extra[f"{tag}_result"] = "extra" if scored_extra(task, p) \
                        else "extra_unscored"
                    extra[f"{tag}_prediction"] = p.get("show", "")
                    rows.append(extra)
            for tag in tags:
                results[tag]["coverage"].setdefault(slug, {})[task] = (
                    len(gold[task]["real"]) + len(gold[task]["wrong"]),
                    len(gold[task]["rows"]))
            comparison[(slug, task)] = rows

    for tag in tags:
        per_paper = results[tag]["per_paper"]
        summary = {}
        for task in TASKS:
            c = pooled(per_paper, task)
            if not c.get("gold"):
                continue
            p, r, f = metrics(c)
            summary[task] = {**c, "precision": p, "recall": r, "f1": f,
                             "f1_ci95": bootstrap(per_paper, task)}
        results[tag]["summary"] = summary
    if write and not self_test:
        write_comparison(comparison, tags)
    return results


def write_comparison(comparison, tags):
    for (slug, task), rows in comparison.items():
        if not rows:
            continue
        path = COMPARE_DIR / slug / f"{GOLD_FILE.get(task, task)}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = []
        for row in rows:
            for k in row:
                if k not in columns:
                    columns.append(k)
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def write_predictions_only(slug, preds):
    """An excluded paper: each model's predictions, for later review."""
    for tag, by_task in preds.items():
        for task, items in by_task.items():
            if not items:
                continue
            path = COMPARE_DIR / slug / f"predictions_{tag}_{GOLD_FILE.get(task, task)}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["prediction", "origin"])
                for p in items:
                    writer.writerow([p.get("show", ""), p.get("origin", "")])


# ---------- report ----------
def _pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def report(results, tags, eur, eur_source):
    reference = None
    for tag in tags:
        if not tag.startswith("qwen"):
            _rows, by_kind, _t = usage(tag)
            reference = {k: v["calls"] for k, v in by_kind.items() if k != "component"}
            break
    if reference is None:
        reference = {"table": 109, "overview": 13, "section": 65}
    costs = {tag: cost_time(tag, reference) for tag in tags}

    lines = ["# Gold-set comparison", ""]
    lines += ["Gold rules applied:", *[f"- {n}" for n in rules.NOTES], ""]
    lines += [f"Currency: {eur_source}.", ""]
    head = "| task | gold items | " + " | ".join(
        f"{t} correct | {t} errors | {t} missed | {t} P / R / F1" for t in tags) + " |"
    lines += [head, "|" + "---|" * (2 + 4 * len(tags))]
    for task in TASKS:
        cells = []
        gold_n = None
        for tag in tags:
            s = results[tag]["summary"].get(task)
            if not s:
                cells += ["–"] * 4
                continue
            gold_n = s["gold"]
            ci = s.get("f1_ci95")
            ci_text = f" [{_pct(ci[0])}–{_pct(ci[1])}]" if ci else ""
            cells += [str(s["correct"]), str(s["errors"]), str(s.get("missed", 0)),
                      f"{_pct(s['precision'])} / {_pct(s['recall'])} / "
                      f"{_pct(s['f1'])}{ci_text}"]
        lines.append(f"| {task} | {gold_n if gold_n is not None else '–'} | "
                     + " | ".join(cells) + " |")
    lines += ["", "Unscored extras (found by a model, not in gold; listed in "
                  "gold_comparison for review):", ""]
    for tag in tags:
        extra = {t: results[tag]["summary"].get(t, {}).get("extras_unscored", 0)
                 for t in TASKS}
        lines.append(f"- {tag}: " + ", ".join(f"{t} {n}" for t, n in extra.items() if n)
                     or f"- {tag}: none")
    lines += ["", "| run | calls | input tokens | output tokens | model time | wall time | "
                  "cost USD | cost EUR |", "|---|---|---|---|---|---|---|---|"]
    for tag in tags:
        c = costs[tag]
        lines.append(f"| {tag} | {c['calls']} | {c['input_tokens']:,} | "
                     f"{c['output_tokens']:,} | {c['model_seconds'] / 60:.0f} min | "
                     f"{c['wall_seconds'] / 60:.0f} min | ${c['cost_usd']:.2f} | "
                     f"€{c['cost_usd'] * eur:.2f} |")
        if c.get("time_note"):
            lines.append(f"|  | ({c['time_note']}) | | | | | | |")

    # corpus estimate from the measured per-call tokens of each Claude run
    tag_tokens = {"_sections_per_paper": reference.get("section", 65) / 13,
                  "_profiled_share": 1.0}
    for tag in tags:
        if tag.startswith("qwen"):
            continue
        model = (costs[tag]["models"] or [""])[0]
        # The API names the model with a date suffix ("claude-haiku-4-5-20251001");
        # the price list does not.
        from des_pipeline import config as _config
        model = next((k for k in _config.ANTHROPIC_PRICES if model.startswith(k)), model)
        per = {}
        for kind, v in costs[tag]["by_kind"].items():
            if v.get("calls"):
                per[kind] = {"in": v["input_tokens"] / v["calls"],
                             "out": v["output_tokens"] / v["calls"]}
        tag_tokens[model] = per
    profiled = sum(1 for p in (ROOT / "runs" / tags[-1] / "data" / "papers").glob("*/tables.csv")
                   for r in _read(p) if r["status"] != "not_profiled")
    total_tables = sum(len(_read(p)) for p in
                       (ROOT / "runs" / tags[-1] / "data" / "papers").glob("*/tables.csv"))
    if total_tables:
        tag_tokens["_profiled_share"] = profiled / total_tables
    estimate = None
    if any(not k.startswith("_") for k in tag_tokens):
        estimate = corpus_estimate(tag_tokens, eur)
        lines += ["", f"## Estimated cost for the full corpus ({estimate['papers']} papers, "
                      f"{estimate['tables']} tables)", "",
                  f"Calls: {estimate['calls']}", "",
                  "| model | input tokens | output tokens | cost EUR | with Batch API (-50%) | "
                  "tokens |", "|---|---|---|---|---|---|"]
        for model, e in estimate["per_model"].items():
            lines.append(f"| {model} | {e['input_tokens_M']}M | {e['output_tokens_M']}M | "
                         f"€{e['eur']:,.0f} | €{e['eur_batch_api']:,.0f} | {e['tokens_from']} |")
        lines += ["", "Excludes the optional component-property pass (one call per unique "
                      "compound, ~0.85k input / ~0.75k output tokens each on qwen3; 3k–10k "
                      "compounds for the corpus)."]
    return "\n".join(lines), costs, estimate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", action="append", required=True)
    parser.add_argument("--self-test", action="store_true",
                        help="score the gold against itself; every task must be 100%%")
    args = parser.parse_args(argv)

    if args.self_test:
        results = evaluate(args.tag[:1], write=False, self_test=True)
        bad = {t: s["f1"] for t, s in results[args.tag[0]]["summary"].items()
               if t not in ("tables", "columns") and s["f1"] not in (None, 1.0)}
        acc = {t: s["correct"] / s["gold"] for t, s in results[args.tag[0]]["summary"].items()
               if t in ("tables", "columns") and s["gold"]}
        print("self-test F1:", {t: s["f1"] for t, s in results[args.tag[0]]["summary"].items()})
        ok = not bad and all(abs(a - 1) < 1e-9 for a in acc.values())
        print(f"self-test {'PASSES' if ok else 'FAILS'}: {bad or ''}")
        sys.exit(0 if ok else 1)

    eur, eur_source = eur_per_usd()
    results = evaluate(args.tag)
    text, costs, estimate = report(results, args.tag, eur, eur_source)
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    (COMPARE_DIR / "REPORT.md").write_text(text)
    rows = []
    for tag in args.tag:
        for task, s in results[tag]["summary"].items():
            rows.append({"model": tag, "task": task, "gold_items": s["gold"],
                         "correct": s["correct"], "errors": s["errors"],
                         "missed": s.get("missed", 0),
                         "extras_unscored": s.get("extras_unscored", 0),
                         "precision": s["precision"], "recall": s["recall"], "f1": s["f1"]})
    with open(COMPARE_DIR / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["model"])
        writer.writeheader()
        writer.writerows(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for tag in args.tag:
        (RESULTS_DIR / f"{tag}.json").write_text(json.dumps(
            {**results[tag], "cost_time": costs[tag], "eur_per_usd": eur,
             "corpus_estimate": estimate}, indent=2, default=list))
    print(text)


if __name__ == "__main__":
    main()
