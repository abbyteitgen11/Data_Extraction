#!/usr/bin/env python3
"""
Score one run against the hand-checked gold set, and compare runs.

    python tools/evaluate.py --tag qwen3                  # one run -> eval/results/qwen3.json
    python tools/evaluate.py --tag qwen3 --tag sonnet55   # several, side by side
    python tools/evaluate.py --self-test --tag qwen3      # gold vs itself must score 1.0

What is scored, per paper and pooled (micro-averaged):

    tables        is the table DES data (accuracy), and which content types (F1)
    columns       role + property of each column of the relevant tables (accuracy)
    dictionary    each DES code the paper defines -> components + ratio (P/R/F1)
    mixtures      each distinct DES the paper reports values for (P/R/F1)
    measurements  each value: DES + property + value (0.5%) + temperature (0.5 C) (P/R/F1)
                  plus "right DES and property, wrong value" -- the costly error
    applications  each (DES, domain) link (P/R/F1)

Only gold rows with a verdict count (correct / fix / added); "wrong" rows are not gold,
and blank rows are unreviewed and skipped -- the report says how much of each paper is
reviewed, because a recall over half-reviewed gold means little. For a sampled paper
(meta.json "sampled_rows"), table predictions outside the sampled rows are ignored.

The headline numbers carry 95% bootstrap intervals, resampling PAPERS: thirteen papers
is a small sample, and an interval is what says whether two models really differ.
Cost and time come from runs/<tag>/usage.csv and timing.csv.
"""
import argparse
import csv
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GOLD_DIR = Path(os.environ.get("DES_GOLD_DIR") or ROOT / "eval" / "gold")
RESULTS_DIR = ROOT / "eval" / "results"
REVIEWED = {"correct", "fix", "added"}
TASKS = ("tables", "columns", "dictionary", "mixtures", "measurements", "applications")


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


def _aliases():
    global _ALIASES
    if _ALIASES is None:
        try:
            from des_pipeline.enrich_components import load_aliases

            _ALIASES = load_aliases()
        except Exception:                                  # noqa: BLE001
            _ALIASES = {}
    return _ALIASES


def norm_name(name, paper_aliases=None):
    """'ChCl' / 'Choline Chloride' / 'choline-chloride' -> 'cholinechloride'."""
    key = str(name or "").strip().lower()
    if key in ("water", "h2o", "h₂o"):
        return "water"
    key = (paper_aliases or {}).get(key, key)
    key = _aliases().get(key, key).lower() if key in _aliases() else key
    return re.sub(r"[^a-z0-9]", "", key)


def des_identity(components, ratio, paper_aliases=None):
    """-> (frozenset of names, ratio fingerprint or None). Order-independent.

    The ratio is attached to each component (they are positionally aligned) and scaled
    so its smallest part is 1, so "1:2" and "2:4" are the same DES.
    """
    names = [norm_name(c, paper_aliases) for c in str(components or "").split(";")
             if str(c).strip()]
    names = [n for n in names if n]
    parts = []
    for p in re.split(r"\s*:\s*", str(ratio or "").strip()):
        value = _float(p)
        if value is None:
            parts = []
            break
        parts.append(value)
    fingerprint = None
    if parts and len(parts) == len(names) and min(parts) > 0:
        low = min(parts)
        fingerprint = frozenset((n, round(p / low, 2)) for n, p in zip(names, parts))
    return frozenset(names), fingerprint


def same_des(a, b, require_ratio=True):
    names_a, ratio_a = a
    names_b, ratio_b = b
    if names_a != names_b or not names_a:
        return False
    if not require_ratio:
        return True
    if ratio_a is None and ratio_b is None:
        return True
    return ratio_a == ratio_b


def same_value(a, b, rel=0.005):
    a, b = _float(a), _float(b)
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=rel, abs_tol=1e-9)


def same_temperature(a, b, tol=0.5):
    a, b = _float(a), _float(b)
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol


# ---------- matching ----------
def match(pred, gold, equal):
    """Greedy one-to-one matching. -> (tp, fp, fn, unmatched_pred, unmatched_gold)."""
    used = set()
    tp = 0
    unmatched_pred = []
    for p in pred:
        hit = next((i for i, g in enumerate(gold) if i not in used and equal(p, g)), None)
        if hit is None:
            unmatched_pred.append(p)
        else:
            used.add(hit)
            tp += 1
    unmatched_gold = [g for i, g in enumerate(gold) if i not in used]
    return tp, len(unmatched_pred), len(unmatched_gold), unmatched_pred, unmatched_gold


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if (p == 0 or r == 0) else None)
    return p, r, f


# ---------- per paper ----------
def gold_items(slug):
    """-> {task: [items]}, coverage, meta, paper aliases. Only reviewed rows."""
    folder = GOLD_DIR / slug
    meta = json.loads((folder / "meta.json").read_text()) if (folder / "meta.json").exists() \
        else {}
    coverage, items = {}, {}
    aliases = {}
    dictionary = _read(folder / "des_dictionary.csv")
    for r in dictionary:
        if r.get("kind") == "alias" and r.get("verdict") in REVIEWED:
            aliases[str(r["code"]).strip().lower()] = str(r["gold_components"]).strip().lower()
    for task in TASKS:
        rows = _read(folder / f"{task if task != 'dictionary' else 'des_dictionary'}.csv")
        reviewed = [r for r in rows if (r.get("verdict") or "").strip() in REVIEWED
                    or (r.get("verdict") or "").strip() == "wrong"]
        coverage[task] = (len(reviewed), len(rows))
        items[task] = [r for r in rows if (r.get("verdict") or "").strip() in REVIEWED]
    return items, coverage, meta, aliases


def pred_items(run_data, slug, meta):
    """The run's predictions for one paper, in the same shapes as the gold files."""
    src = run_data / "papers" / slug
    out = {}
    ledger = _read(src / "tables.csv")
    out["tables"] = [{
        "table_id": r["table_id"],
        "relevant": "no" if r["status"] in ("not_relevant", "not_profiled", "not_des")
        else "yes",
        "content_types": r["content_types"] if r["status"] not in (
            "not_relevant", "not_profiled", "not_des") else ""} for r in ledger]
    profiles_path = src / "table_profiles.json"
    profiles = json.loads(profiles_path.read_text()) if profiles_path.exists() else {}
    out["columns"] = [{"table_id": t, "column": str(c.get("index")), "role": c.get("role"),
                       "property": c.get("property") or ""}
                      for t, rec in profiles.items()
                      for c in rec.get("profile", {}).get("columns", [])]
    out["dictionary"] = [{"code": r["code"], "components": r["components"],
                          "ratio": r.get("ratio_molar") or r.get("ratio_raw") or ""}
                         for r in _read(src / "des_dictionary.csv")
                         if r.get("kind") == "mixture"
                         and str(r.get("verified")).lower() not in ("false", "0")]
    mixtures = _read(src / "mixtures.csv")
    by_row = {r["Row_id"]: r for r in mixtures}

    def comps(r):
        return ";".join(str(r.get(f"Component_{i}")) for i in (1, 2, 3)
                        if r.get(f"Component_{i}") not in (None, "", "nan"))

    seen, out["mixtures"] = set(), []
    scope = set(meta.get("rows") or []) if meta.get("measurement_scope") == "sampled_rows" \
        else None
    for r in mixtures:
        if scope is not None and \
                f"{r['Table_id']}:{int(float(r['Source_row']))}" not in scope:
            continue
        key = (comps(r), r.get("Ratio_molar") or r.get("Ratio_raw") or "")
        if key not in seen:
            seen.add(key)
            out["mixtures"].append({"components": key[0], "ratio": key[1]})
    sampled = set(meta.get("rows") or []) if meta.get("measurement_scope") == "sampled_rows" \
        else None
    out["measurements"] = []
    for r in _read(src / "measurements.csv"):
        if (r.get("status") or "ok") != "ok":
            continue
        if sampled is not None and f"{r['Table_id']}:{int(float(r['Source_row']))}" \
                not in sampled:
            continue
        mix = by_row.get(r["Row_id"], {})
        out["measurements"].append({
            "origin": "table", "components": comps(mix),
            "ratio": mix.get("Ratio_molar") or mix.get("Ratio_raw") or "",
            "property": r["Property"], "value": r["Value"],
            "temperature": r.get("Temperature_C")})
    for r in _read(src / "sections_llm.csv"):
        if r.get("status") == "ok":
            out["measurements"].append({
                "origin": "prose", "components": r.get("components_resolved", ""),
                "ratio": r.get("molar_ratio") or "", "property": r.get("property"),
                "value": r.get("value"), "temperature": r.get("temperature_C")})
    out["applications"] = [{"components": comps(r),
                            "ratio": r.get("Ratio_molar") or r.get("Ratio_raw") or "",
                            "domain": r.get("Domain", "")}
                           for r in _read(src / "applications.csv")
                           if (r.get("status") or "ok") == "ok"]
    return out


def gold_as_pred(items):
    """The gold rows reshaped as predictions -- for --self-test."""
    g = items
    return {
        "tables": [{"table_id": r["table_id"], "relevant": r["gold_relevant"],
                    "content_types": r["gold_content_types"]} for r in g["tables"]],
        "columns": [{"table_id": r["table_id"], "column": str(r["column"]),
                     "role": r["gold_role"], "property": r["gold_property"]}
                    for r in g["columns"]],
        "dictionary": [{"code": r["code"], "components": r["gold_components"],
                        "ratio": r["gold_ratio"]} for r in g["dictionary"]
                       if r.get("kind") == "mixture"],
        "mixtures": [{"components": r["gold_components"], "ratio": r["gold_ratio"]}
                     for r in g["mixtures"]],
        "measurements": [{"origin": r["origin"], "components": r["gold_components"],
                          "ratio": r["gold_ratio"], "property": r["gold_property"],
                          "value": r["gold_value"], "temperature": r["gold_temperature_C"]}
                         for r in g["measurements"]],
        "applications": [{"components": r["gold_components"], "ratio": r["gold_ratio"],
                          "domain": r["gold_domain"]} for r in g["applications"]],
    }


def score_paper(pred, gold, aliases):
    """-> {task: counts}. Counts are what pool across papers."""
    out = {}

    # tables: relevance accuracy + content-type F1
    by_id = {r["table_id"]: r for r in pred["tables"]}
    correct = total = 0
    tp = fp = fn = 0
    for g in gold["tables"]:
        p = by_id.get(g["table_id"], {"relevant": "no", "content_types": ""})
        total += 1
        correct += (p["relevant"] == g["gold_relevant"])
        gt = {t for t in str(g["gold_content_types"]).split("+") if t and t != "other"}
        pt = {t for t in str(p["content_types"]).split("+") if t and t != "other"}
        tp += len(gt & pt)
        fp += len(pt - gt)
        fn += len(gt - pt)
    out["tables"] = {"correct": correct, "total": total, "tp": tp, "fp": fp, "fn": fn}

    # columns: accuracy over reviewed columns
    by_col = {(r["table_id"], r["column"]): r for r in pred["columns"]}
    correct = total = 0
    for g in gold["columns"]:
        p = by_col.get((g["table_id"], str(g["column"])))
        total += 1
        correct += bool(p and p["role"] == g["gold_role"]
                        and (p["property"] or "") == (g["gold_property"] or ""))
    out["columns"] = {"correct": correct, "total": total}

    # dictionary: by code -> components + ratio
    def dict_eq(p, g):
        return re.sub(r"[^a-z0-9]", "", p["code"].lower()) == \
            re.sub(r"[^a-z0-9]", "", g["code"].lower()) and same_des(
                des_identity(p["components"], p["ratio"], aliases),
                des_identity(g["gold_components"], g["gold_ratio"], aliases))
    gold_dict = [g for g in gold["dictionary"] if g.get("kind") == "mixture"]
    tp, fp, fn, _, _ = match(pred["dictionary"], gold_dict, dict_eq)
    out["dictionary"] = {"tp": tp, "fp": fp, "fn": fn}

    # mixtures
    tp, fp, fn, _, _ = match(
        pred["mixtures"], gold["mixtures"],
        lambda p, g: same_des(des_identity(p["components"], p["ratio"], aliases),
                              des_identity(g["gold_components"], g["gold_ratio"], aliases)))
    out["mixtures"] = {"tp": tp, "fp": fp, "fn": fn}

    # measurements, and the wrong-value count among the misses
    def m_eq(p, g):
        return (p["property"] == g["gold_property"]
                and same_value(p["value"], g["gold_value"])
                and same_temperature(p["temperature"], g["gold_temperature_C"])
                and same_des(des_identity(p["components"], p["ratio"], aliases),
                             des_identity(g["gold_components"], g["gold_ratio"], aliases)))
    tp, fp, fn, extra, missed = match(pred["measurements"], gold["measurements"], m_eq)
    wrong_value = 0
    for p in extra:
        if any(p["property"] == g["gold_property"]
               and same_temperature(p["temperature"], g["gold_temperature_C"])
               and same_des(des_identity(p["components"], p["ratio"], aliases),
                            des_identity(g["gold_components"], g["gold_ratio"], aliases))
               for g in missed):
            wrong_value += 1
    by_origin = {}
    for origin in ("table", "prose"):
        o_tp, o_fp, o_fn, _, _ = match(
            [p for p in pred["measurements"] if p["origin"] == origin],
            [g for g in gold["measurements"] if g["origin"] == origin], m_eq)
        by_origin[origin] = {"tp": o_tp, "fp": o_fp, "fn": o_fn}
    out["measurements"] = {"tp": tp, "fp": fp, "fn": fn, "wrong_value": wrong_value,
                           "by_origin": by_origin}

    # applications: DES (ratio not required -- many uses name a DES without one) + domain
    tp, fp, fn, _, _ = match(
        pred["applications"], gold["applications"],
        lambda p, g: p["domain"] == g["gold_domain"] and same_des(
            des_identity(p["components"], p["ratio"], aliases),
            des_identity(g["gold_components"], g["gold_ratio"], aliases),
            require_ratio=False))
    out["applications"] = {"tp": tp, "fp": fp, "fn": fn}
    return out


# ---------- pooling and intervals ----------
def pooled(per_paper, slugs=None):
    slugs = slugs if slugs is not None else list(per_paper)
    summary = {}
    for task in TASKS:
        counts = defaultdict(float)
        for slug in slugs:
            for k, v in per_paper[slug].get(task, {}).items():
                if isinstance(v, (int, float)):
                    counts[k] += v
        if "tp" in counts:
            p, r, f = prf(counts["tp"], counts["fp"], counts["fn"])
            summary[task] = {"precision": p, "recall": r, "f1": f, **counts}
        if "total" in counts:
            summary.setdefault(task, {})["accuracy"] = \
                counts["correct"] / counts["total"] if counts["total"] else None
            summary[task].update({"correct": counts["correct"], "total": counts["total"]})
    return summary


def bootstrap(per_paper, task, metric="f1", n=1000, seed=0):
    slugs = [s for s in per_paper if per_paper[s].get(task)]
    if len(slugs) < 2:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(n):
        sample = [rng.choice(slugs) for _ in slugs]
        value = pooled(per_paper, sample).get(task, {}).get(metric)
        if value is not None:
            values.append(value)
    if not values:
        return None
    values.sort()
    return values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]


# ---------- cost and time ----------
def cost_and_time(tag):
    run_dir = ROOT / "runs" / tag
    usage = _read(run_dir / "usage.csv")
    timing = _read(run_dir / "timing.csv")
    out = {"calls": len(usage),
           "cost_usd": round(sum(_float(r.get("cost_usd")) or 0 for r in usage), 4),
           "model_seconds": round(sum(_float(r.get("seconds")) or 0 for r in usage), 1),
           "input_tokens": int(sum(_float(r.get("input_tokens")) or 0 for r in usage)),
           "output_tokens": int(sum(_float(r.get("output_tokens")) or 0 for r in usage)),
           "truncated_calls": sum(1 for r in usage
                                  if r.get("stop_reason") in ("length", "max_tokens")),
           "wall_seconds": round(sum(_float(r.get("seconds")) or 0 for r in timing), 1),
           "models": sorted({r.get("model") for r in usage if r.get("model")})}
    return out


# ---------- driver ----------
def evaluate(tag, self_test=False, corrupt=False):
    os.environ["DES_RUN_TAG"] = tag
    run_data = ROOT / "runs" / tag / "data"
    per_paper, coverage = {}, {}
    for folder in sorted(GOLD_DIR.iterdir()) if GOLD_DIR.exists() else []:
        if not folder.is_dir():
            continue
        slug = folder.name
        gold, cover, meta, aliases = gold_items(slug)
        coverage[slug] = cover
        if not any(gold.values()):
            continue
        if self_test:
            pred = gold_as_pred(gold)
            if corrupt:
                for m in pred["measurements"][::3]:
                    v = _float(m["value"])
                    m["value"] = None if v is None else v * 1.1
                pred["applications"] = pred["applications"][::2]
        else:
            pred = pred_items(run_data, slug, meta)
        per_paper[slug] = score_paper(pred, gold, aliases)
    summary = pooled(per_paper)
    for task in TASKS:
        if task in summary and "f1" in summary[task]:
            summary[task]["f1_ci95"] = bootstrap(per_paper, task, "f1")
        if task in summary and "accuracy" in summary[task]:
            summary[task]["accuracy_ci95"] = bootstrap(per_paper, task, "accuracy")
    return {"tag": tag, "self_test": self_test, "summary": summary,
            "per_paper": per_paper, "coverage": coverage,
            "cost_time": cost_and_time(tag) if not self_test else {}}


def _fmt(x, pct=True):
    if x is None:
        return "  -  "
    return f"{x * 100:5.1f}" if pct else f"{x}"


def print_report(results):
    print(f"\n{'task':<14}" + "".join(f"{r['tag']:>26}" for r in results))
    for task in TASKS:
        cells = []
        for r in results:
            s = r["summary"].get(task, {})
            if "f1" in s:
                ci = s.get("f1_ci95")
                ci_text = f" [{_fmt(ci[0])}-{_fmt(ci[1])}]" if ci else ""
                cells.append(f"F1 {_fmt(s['f1'])}{ci_text}")
            elif "accuracy" in s:
                ci = s.get("accuracy_ci95")
                ci_text = f" [{_fmt(ci[0])}-{_fmt(ci[1])}]" if ci else ""
                cells.append(f"acc {_fmt(s['accuracy'])}{ci_text}")
            else:
                cells.append("-")
        print(f"{task:<14}" + "".join(f"{c:>26}" for c in cells))
    for key in ("precision", "recall"):
        print(f"{'meas. ' + key:<14}" + "".join(
            f"{_fmt(r['summary'].get('measurements', {}).get(key)):>26}" for r in results))
    print(f"{'wrong values':<14}" + "".join(
        f"{int(r['summary'].get('measurements', {}).get('wrong_value', 0)):>26}"
        for r in results))
    for key, label in (("cost_usd", "cost $"), ("model_seconds", "model s"),
                       ("wall_seconds", "wall s"), ("calls", "calls"),
                       ("truncated_calls", "truncated")):
        print(f"{label:<14}" + "".join(
            f"{str(r.get('cost_time', {}).get(key, '-')):>26}" for r in results))
    print("\nreview coverage (reviewed/total rows) -- unreviewed rows are not scored:")
    first = results[0]
    for slug, cover in first["coverage"].items():
        parts = [f"{t[:5]} {a}/{b}" for t, (a, b) in cover.items() if b]
        print(f"  {slug:<22} " + "  ".join(parts))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", action="append", required=True,
                        help="run tag(s) to score; repeat to compare")
    parser.add_argument("--self-test", action="store_true",
                        help="score the gold against itself (must be 100%%), and a "
                             "corrupted copy (must be lower)")
    args = parser.parse_args(argv)

    if args.self_test:
        clean = evaluate(args.tag[0], self_test=True)
        broken = evaluate(args.tag[0], self_test=True, corrupt=True)
        clean["tag"], broken["tag"] = "gold vs gold", "gold vs corrupted"
        print_report([clean, broken])
        perfect = all(abs((v.get("f1") if v.get("f1") is not None else v.get("accuracy", 1))
                          - 1) < 1e-9 for v in clean["summary"].values()
                      if v.get("f1") is not None or v.get("accuracy") is not None)
        print(f"\nself-test: gold vs gold {'PASSES' if perfect else 'FAILS'} at 100%")
        sys.exit(0 if perfect else 1)

    results = [evaluate(tag) for tag in args.tag]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for r in results:
        (RESULTS_DIR / f"{r['tag']}.json").write_text(json.dumps(r, indent=2, default=list))
    print_report(results)
    rows = []
    for r in results:
        row = {"tag": r["tag"]}
        for task, s in r["summary"].items():
            for k in ("precision", "recall", "f1", "accuracy"):
                if s.get(k) is not None:
                    row[f"{task}_{k}"] = round(s[k], 4)
        row.update({k: v for k, v in r["cost_time"].items() if k != "models"})
        rows.append(row)
    columns = sorted({k for row in rows for k in row}, key=lambda k: (k != "tag", k))
    with open(ROOT / "eval" / "comparison.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n-> eval/comparison.csv, eval/results/<tag>.json")


if __name__ == "__main__":
    main()
