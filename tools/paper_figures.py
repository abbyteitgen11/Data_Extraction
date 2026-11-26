#!/usr/bin/env python3
"""
Publication figures for the DES knowledge graph, regenerated from whatever is loaded.

    /opt/anaconda3/bin/python tools/paper_figures.py --tag corpus_full --refresh
    /opt/anaconda3/bin/python tools/paper_figures.py --tag corpus_full --only fig4,fig5

--refresh first runs tools/figure_data.py in the GraphDataBase env (it needs the neo4j
driver and RDKit, which this env lacks; this env has matplotlib, which that one lacks),
writing runs/<tag>/figures/figure_data.json. Without it the last JSON is redrawn, so
restyling never touches the database.

Writes to runs/<tag>/figures/ (each as .png, .svg and .pdf unless --formats says otherwise):
  fig1_workflow                 the extraction pipeline, with this run's counts
  fig2a_overview_values         Sankey: DES property values, publisher -> graph -> property
  fig2b_overview_applications   Sankey: application rows, publisher -> graph -> domain
  fig2c_overview_components     Sankey: components, HBA/HBD -> PubChem -> category -> values
  fig2d_overview_numbers        the headline numbers as tiles
  fig2_overview                 2a + 2d on one page
  fig3a..fig3n                  distributions, one figure each (see FIGURES below)
  fig4_applications_radial      domains -> most frequent targets (after Pruyn et al. 2025)
  fig5_network                  every DES linked to its components, properties, domains
  README.md                     what each figure counts, for the captions
"""
import argparse
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402
from matplotlib import patheffects                    # noqa: E402
from matplotlib.collections import LineCollection     # noqa: E402
from matplotlib.colors import LogNorm                 # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch, Wedge  # noqa: E402
from matplotlib.path import Path as MPath             # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GRAPH_PYTHON = os.environ.get("DES_GRAPH_PYTHON", "/opt/anaconda3/envs/GraphDataBase/bin/python")

# Edit these when the method changes; they are printed in the workflow figure.
EXTRACTION_MODEL = "Claude Haiku 4.5 (Message Batches)"
GOLD_SET = "13-paper gold set, scored per field;\n4 LLMs compared (Qwen3, Haiku 4.5,\nSonnet 5.5, Opus 5.5)"

plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "savefig.dpi": 300, "svg.fonttype": "none",
                     "pdf.fonttype": 42})

PALETTE = ["#8f6b8e", "#d58b7d", "#e6cf9f", "#9cc7a2", "#5f7fbf", "#7f98ad", "#b4aed6",
           "#6c9a8b", "#c9a46b", "#b5838d", "#a3a380", "#7a7aa6", "#d4a5a5", "#88a9c3",
           "#c2b280", "#9e9e9e"]
GREY = "#c8c8c8"
KIND_COLOR = {"component": "#e07b8c", "property": "#4a8fc2", "domain": "#5cae5c"}

PROP_LABEL = {"Melting_point": "Melting point", "Boiling_point": "Boiling point",
              "Density": "Density", "Viscosity": "Viscosity", "Conductivity": "Ionic conductivity",
              "Thermal_conductivity": "Thermal conductivity", "Surface_tension": "Surface tension",
              "Refractive_index": "Refractive index", "pH": "pH",
              "Decomposition_temperature": "Decomposition temp.", "Glass_transition": "Glass transition",
              "Zeta_potential": "Zeta potential", "Heat_capacity": "Heat capacity",
              "Polarity_ET30": "Polarity $E_T(30)$", "Solubility": "Solubility"}
DOMAIN_LABEL = {"extraction_separation": "Extraction & separation",
                "natural_product_extraction": "Natural product extraction",
                "organic_synthesis_medium": "Organic synthesis",
                "biomass_fractionation": "Biomass fractionation",
                "metal_processing": "Metal processing", "analytical_sample_prep": "Analytical sample prep.",
                "pharmaceutical_solubilization": "Pharmaceutical solubilisation",
                "polymer_materials": "Polymers & materials", "gas_capture": "Gas capture",
                "catalysis": "Catalysis", "electrochemistry_batteries": "Electrochemistry & batteries",
                "membrane_separation": "Membrane separation", "drug_delivery": "Drug delivery",
                "esterification": "Esterification", "fuel_cleaning": "Fuel desulfurisation",
                "biodiesel": "Biodiesel", "other": "Unclassified ('other')"}
STATUS_LABEL = {"no_des_in_quote": "No DES in quote", "unhandled_unit": "Unit not handled",
                "no_solute": "Solubility without solute", "range_only": "Range only",
                "ambiguous_basis": "Ambiguous basis", "unverified": "Quote not verified",
                "not_in_dictionary": "DES not defined in paper",
                "unresolved_components": "Unresolved components", "quote_not_found": "Quote not found",
                "value_not_in_quote": "Value not in quote", "duplicate": "Duplicate",
                "not_des": "Not a DES"}


def prop(p):
    return PROP_LABEL.get(p, p.replace("_", " "))


def domain(d):
    return DOMAIN_LABEL.get(d, d.replace("_", " ").capitalize())


def unit(u):
    if not u or u == "None":
        return ""
    if u == "C":
        return "°C"
    out = []
    for part in u.split("*"):
        base, _, exp = part.partition("^")
        out.append(f"{base}$^{{{exp}}}$" if exp else base)
    return " ".join(out)


def nice(name):
    """Component names as printed: first letter up, unless it is a formula/abbreviation."""
    return name[:1].upper() + name[1:] if name and name[:1].islower() else name


def fmt(n):
    return f"{n:,}"


def save(fig, out, name, formats):
    for f in formats:
        fig.savefig(out / f"{name}.{f}")
    plt.close(fig)
    print(f"  {name}")


# ------------------------------------------------------------------------------ Sankey
def sankey(ax, columns, flows, color, headers=None, node_w=0.014, pad=0.018, label_w=0.16,
           band_alpha=0.45, label=None, flow_color=None):
    """Alluvial diagram. columns: node ids per column, top to bottom; flows: (src, dst, n).
    A node's height is the larger of what flows in and out, so a node whose data stops
    (e.g. held for review) simply has no outgoing band."""
    flows = [(s, t, v) for s, t, v in flows if v > 0]
    inflow, outflow = Counter(), Counter()
    for s, t, v in flows:
        outflow[s] += v
        inflow[t] += v
    columns = [[n for n in col if max(inflow[n], outflow[n]) > 0] for col in columns]
    val = {n: max(inflow[n], outflow[n]) for col in columns for n in col}
    ncol = len(columns)
    xs = [i * (1 - label_w) / (ncol - 1) for i in range(ncol)]
    gaps = max(len(c) - 1 for c in columns)
    scale = (1 - pad * gaps) / max(sum(val[n] for n in c) for c in columns)
    pos, order = {}, {}
    for i, col in enumerate(columns):
        height = sum(val[n] for n in col) * scale + pad * (len(col) - 1)
        y = 1 - (1 - height) / 2
        for j, n in enumerate(col):
            h = val[n] * scale
            pos[n] = (xs[i], y, h)
            order[n] = (i, j)
            y -= h + pad
    out_y = {n: pos[n][1] for n in pos}
    in_y = {n: pos[n][1] for n in pos}
    spans = {}
    for s, t, v in sorted(flows, key=lambda f: (order[f[0]], order[f[1]])):
        spans[(s, t)] = [out_y[s], None]
        out_y[s] -= v * scale
    for s, t, v in sorted(flows, key=lambda f: (order[f[1]], order[f[0]])):
        spans[(s, t)][1] = in_y[t]
        in_y[t] -= v * scale
    for s, t, v in flows:
        y0, y1 = spans[(s, t)]
        h = v * scale
        x0, x1 = pos[s][0] + node_w, pos[t][0]
        xm = (x0 + x1) / 2
        verts = [(x0, y0), (xm, y0), (xm, y1), (x1, y1), (x1, y1 - h), (xm, y1 - h),
                 (xm, y0 - h), (x0, y0 - h), (x0, y0)]
        codes = [MPath.MOVETO] + [MPath.CURVE4] * 3 + [MPath.LINETO] + [MPath.CURVE4] * 3 + \
                [MPath.CLOSEPOLY]
        c = (flow_color or {}).get((s, t)) or color.get(s, GREY)
        ax.add_patch(PathPatch(MPath(verts, codes), fc=c, ec="none", alpha=band_alpha, lw=0))
    for n, (x, y, h) in pos.items():
        ax.add_patch(plt.Rectangle((x, y - h), node_w, h, fc=color.get(n, "#777"), ec="#333",
                                   lw=0.4))
        text = label(n, val[n]) if label else f"{n}  {fmt(int(val[n]))}"
        ax.text(x + node_w + 0.006, y - h / 2, text, va="center", ha="left", fontsize=7.5)
    for i, head in enumerate(headers or []):
        if head:
            ax.text(xs[i] + node_w / 2, 1.04, head, ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold", color="#444")
    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(-0.02, 1.1)
    ax.axis("off")


def _top(pairs, k, other="Others"):
    pairs = sorted(pairs, key=lambda p: -p[1])
    head = [(n, v) for n, v in pairs[:k]]
    rest = sum(v for _, v in pairs[k:])
    return head + ([(other, rest)] if rest else [])


def _reasons(statuses, held):
    """Top 3 reasons a row was held back, exactly; whatever else kept a row out of the
    graph (not a status, e.g. a row the loader could not attach) joins "Other reasons"."""
    named = [(STATUS_LABEL.get(k, k), n) for k, n in statuses if k != "ok"]
    head = sorted(named, key=lambda kv: -kv[1])[:3]
    rest = held - sum(n for _, n in head)
    return [(f"⟂ {k}", n) for k, n in head] + ([("⟂ Other reasons", rest)] if rest > 0 else [])


def draw_values_sankey(ax, d):
    v = d["values"]
    pubs = _top(v["rows_by_publisher"], 6)
    statuses = Counter(dict(v["status_table"])) + Counter(dict(v["status_prose"]))
    loaded, nodes, all_rows = v["rows_in_graph"], v["nodes"], v["rows_extracted"]
    held = all_rows - loaded
    reasons = _reasons(statuses.items(), held)
    props = _top([(prop(p), n) for p, n in v["by_property"]], 5,
                 f"{len(v['by_property']) - 5} other properties")
    pmap = {prop(p): prop(p) for p, _ in v["by_property"]}
    top5 = {k for k, _ in props[:5]}
    flows = [(f"P:{p}", "All extracted values", n) for p, n in pubs]
    flows += [("All extracted values", "Loaded into graph", loaded),
              ("All extracted values", "Held for review (CSV)", held),
              ("Loaded into graph", "Unique values", nodes),
              ("Loaded into graph", "Merged duplicates", loaded - nodes)]
    flows += [("Held for review (CSV)", k, n) for k, n in reasons]
    pt, tu = Counter(), Counter()
    for (p, has_t, has_u), n in v["flows"]:
        name = prop(p) if prop(p) in top5 else props[-1][0]
        pt[(name, has_t)] += n
        tu[(has_t, has_u)] += n
    flows += [("Unique values", k, n) for k, n in props]
    flows += [(p, f"T:{t}", n) for (p, t), n in pt.items()]
    flows += [(f"T:{t}", f"U:{u}", n) for (t, u), n in tu.items()]
    columns = [[f"P:{p}" for p, _ in pubs], ["All extracted values"],
               ["Loaded into graph", "Held for review (CSV)"],
               ["Unique values", "Merged duplicates"] + [k for k, _ in reasons],
               [k for k, _ in props], ["T:True", "T:False"], ["U:True", "U:False"]]
    color = {f"P:{p}": PALETTE[i] for i, (p, _) in enumerate(pubs)}
    color.update({"All extracted values": "#8a8a8a", "Loaded into graph": "#5c5c66",
                  "Held for review (CSV)": "#d9d9d9", "Unique values": "#4b4b55",
                  "Merged duplicates": "#e5e5e5", "T:True": "#7e9fc6", "T:False": "#4f6383",
                  "U:True": "#9fc2b6", "U:False": "#5d746c"})
    color.update({k: "#eeeeee" for k, _ in reasons})
    color.update({k: c for (k, _), c in zip(props, ["#c7bfc0", "#a39494", "#b9a7a7",
                                                     "#8d7f7f", "#cfc6c6", "#6f6565"])})

    def label(n, value):
        name = {"T:True": "Temperature given", "T:False": "No temperature",
                "U:True": "Uncertainty given", "U:False": "No uncertainty"}.get(n, n)
        return f"{name.split(':', 1)[-1] if n.startswith('P:') else name.lstrip('⟂ ')}  {fmt(int(value))}"

    sankey(ax, columns, flows, color, label=label,
           headers=["Publisher", "", "", "", "DES property", "Measuring\ntemperature",
                    "Uncertainty"])


def draw_applications_sankey(ax, d):
    a = d["applications"]
    pubs = _top(a["rows_by_publisher"], 6)
    loaded, all_rows = a["records"], a["rows_extracted"]
    held = all_rows - loaded
    reasons = _reasons(a["status"], held)
    doms = _top([(domain(k), n) for k, n in a["by_domain"] if k != "other"], 8,
                f"{a['domains'] - 9} other domains")
    unclassified = dict(a["by_domain"]).get("other", 0)
    flows = [(f"P:{p}", "All application rows", n) for p, n in pubs]
    flows += [("All application rows", "Application records", loaded),
              ("All application rows", "Held for review (CSV)", held)]
    flows += [("Held for review (CSV)", k, n) for k, n in reasons]
    flows += [("Application records", k, n) for k, n in doms]
    flows += [("Application records", domain("other"), unclassified)]
    columns = [[f"P:{p}" for p, _ in pubs], ["All application rows"],
               ["Application records", "Held for review (CSV)"],
               [k for k, _ in doms] + [domain("other")] + [k for k, _ in reasons]]
    color = {f"P:{p}": PALETTE[i] for i, (p, _) in enumerate(pubs)}
    color.update({"All application rows": "#8a8a8a", "Application records": "#5c5c66",
                  "Held for review (CSV)": "#d9d9d9", domain("other"): "#bdbdbd"})
    color.update({k: "#eeeeee" for k, _ in reasons})
    color.update({k: PALETTE[(i + 4) % len(PALETTE)] for i, (k, _) in enumerate(doms)})
    sankey(ax, columns, flows, color,
           label=lambda n, v: f"{n[2:] if n.startswith('P:') else n.lstrip('⟂ ')}  {fmt(int(v))}",
           headers=["Publisher", "", "", "Application domain"], label_w=0.25)


def draw_components_sankey(ax, d):
    flows_in = d["components"]["flows"]
    f = Counter()
    for (role, ident, cat, vals), n in flows_in:
        idn = "In PubChem" if ident else "Not identified in PubChem"
        f[(role, idn)] += n
        if ident:
            f[(idn, cat)] += n
            f[(cat, "Has database values" if vals else "No database values")] += n
    cats = [c for c, _ in Counter({k[1]: 0 for k in f}).items()]
    cat_order = [c for c, _ in sorted(Counter({c: n for (s, c), n in f.items() if s == "In PubChem"}).items(),
                                      key=lambda kv: -kv[1])]
    columns = [["HBA only", "both", "HBD only"], ["In PubChem", "Not identified in PubChem"],
               cat_order, ["Has database values", "No database values"]]
    del cats
    color = {"HBA only": "#d58b7d", "HBD only": "#5f7fbf", "both": "#8f6b8e",
             "In PubChem": "#6c9a8b", "Not identified in PubChem": "#d9d9d9",
             "Has database values": "#c9a46b", "No database values": "#e8dcc3"}
    color.update({c: PALETTE[i + 6] for i, c in enumerate(cat_order)})
    names = {"HBA only": "Used as HBA only", "HBD only": "Used as HBD only",
             "both": "Used as HBA and HBD"}
    sankey(ax, columns, [(s, t, n) for (s, t), n in f.items()], color,
           label=lambda n, v: f"{names.get(n, n[:1].upper() + n[1:])}  {fmt(int(v))}",
           headers=["Role in DES", "Identity", "Category", "Pure-component values"], label_w=0.2)


def draw_numbers(ax, d):
    p, c, m, v = d["papers"], d["components"], d["mixtures"], d["values"]
    a, cv = d["applications"], d["component_values"]
    top5 = " · ".join(f"{prop(k)} {fmt(n)}" for k, n in v["by_property"][:5])
    groups = [
        ("Literature", "#ed8274", [
            ("Papers extracted", fmt(p["extracted"]), f"{fmt(p['primary'])} primary · {fmt(p['review'])} review"),
            ("Publishers", fmt(len([x for x in p["publisher"] if x[0] != "Others"])),
             " · ".join(f"{k} {n}" for k, n in p["publisher"][:3])),
            ("Papers cited as original source", fmt(p["cited_only"]), "of a value or application"),
            ("Extracted data points", fmt(v["rows_extracted"] + a["rows_extracted"]),
             f"{fmt(v['rows_extracted'])} values · {fmt(a['rows_extracted'])} application rows"),
            ("Unique data points in graph", fmt(v["nodes"] + a["records"]),
             f"{fmt(v['rows_in_graph'] - v['nodes'])} merged duplicates")]),
        ("Deep eutectic solvents", "#9fd77f", [
            ("DES (mixtures)", fmt(m["total"]), "components + ratio + water/additive"),
            ("Distinct component sets", fmt(m["distinct_sets"]), f"{fmt(m['distinct_sets_dry'])} ignoring water"),
            ("DES with measured properties", fmt(m["with_values"]),
             f"{fmt(m['with_apps'])} with applications · {fmt(m['with_both'])} both"),
            ("Binary / ternary DES", " / ".join(fmt(n) for k, n in sorted(m["n_components"])[:2]),
             f"{fmt(sum(n for k, n in m['n_components'] if k > 3))} with 4+ components")]),
        ("Data", "#bab3f4", [
            ("DES property values", fmt(v["nodes"]), f"{len(v['by_property'])} property types"),
            ("With uncertainty", fmt(v["with_uncertainty"]), f"{fmt(v['with_temperature'])} with temperature"),
            ("Top 5 properties", "", top5),
            ("Application records", fmt(a["records"]), f"{a['domains']} application domains"),
            ("Component property values", fmt(cv["total"]), f"from {cv['sources']} sources (PubChem, NIST)")]),
        ("Components", "#def1d3", [
            ("Components", fmt(c["total"]), f"{fmt(c['pubchem'])} identified in PubChem"),
            ("With database values", fmt(c["with_db_values"]), f"{fmt(c['qm9'])} with QM9 descriptors"),
            ("Used as HBA / HBD", f"{fmt(c['roles'].get('HBA only', 0))} / {fmt(c['roles'].get('HBD only', 0))}",
             f"{fmt(c['roles'].get('both', 0))} used as both")]),
    ]
    ax.axis("off")
    ax.set_xlim(0, len(groups))
    rows = max(len(g[2]) for g in groups)
    ax.set_ylim(0, rows + 0.8)
    for gi, (title, col, tiles) in enumerate(groups):
        ax.text(gi + 0.5, rows + 0.45, title, ha="center", va="center", fontsize=11, fontweight="bold")
        for ti, (name, big, small) in enumerate(tiles):
            y = rows - ti - 0.95
            ax.add_patch(FancyBboxPatch((gi + 0.04, y), 0.92, 0.86, boxstyle="round,pad=0,rounding_size=0.06",
                                        fc=col, ec="none", alpha=0.35))
            ax.text(gi + 0.09, y + 0.68, name, fontsize=8, color="#333", va="center")
            if big:
                ax.text(gi + 0.09, y + 0.40, big, fontsize=15, fontweight="bold", va="center")
                ax.text(gi + 0.09, y + 0.13, small, fontsize=6.8, color="#555", va="center")
            else:
                ax.text(gi + 0.09, y + 0.33, small.replace(" · ", "\n", 2).replace(" · ", "\n"),
                        fontsize=6.8, color="#333", va="center", linespacing=1.25)


# ---------------------------------------------------------------------- fig 1 workflow
def fig_workflow(d, out, formats):
    pl, p = d["pipeline"], d["papers"]
    ts = pl["table_status"]
    fig, ax = plt.subplots(figsize=(16, 8.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")
    panels = [(1, 17, "1  Literature", "#fbe3df"), (20, 35, "2  Parse full-text XML", "#eef0f5"),
              (38, 63, f"3  Extract  ·  {EXTRACTION_MODEL}", "#efedfb"),
              (66, 85, "4  Enrich & validate", "#e9f4e4"), (88, 99.5, "5  Knowledge graph", "#e3eef8")]
    for x0, x1, title, col in panels:
        ax.add_patch(FancyBboxPatch((x0, 2), x1 - x0, 52.5, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc=col, ec="none"))
        ax.text((x0 + x1) / 2, 56.3, title, ha="center", va="center", fontsize=11, fontweight="bold")

    def box(x, y, w, h, title, body, ec="#555", fc="white"):
        ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.6",
                                    fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h / 2 - 1.1, title, ha="center", va="top", fontsize=9, fontweight="bold")
        ax.text(x + w / 2, y + h / 2 - 2.9, body, ha="center", va="top", fontsize=7.6, color="#333",
                linespacing=1.3)
        return (x, y, w, h)

    def arrow(a, b, side_a="r", side_b="l", **kw):
        xa = a[0] + a[2] if side_a == "r" else a[0] + a[2] / 2
        ya = a[1] if side_a == "r" else a[1] - a[3] / 2
        xb = b[0] if side_b == "l" else b[0] + b[2] / 2
        yb = b[1] if side_b == "l" else b[1] + b[3] / 2
        ax.add_patch(FancyArrowPatch((xa, ya), (xb, yb), arrowstyle="-|>", mutation_scale=10,
                                     lw=0.9, color="#555", shrinkA=1, shrinkB=1, **kw))

    rows = [47, 38.5, 30, 21.5, 13]
    a1 = box(2, 44, 14, 12, "PMC open-access search",
             f"\"deep eutectic solvent(s)\"\n{fmt(pl.get('downloaded', 0))} full-text\nJATS XML papers")
    a2 = box(2, 24, 14, 13, "Corpus",
             f"phrase in title/abstract:\n{fmt(pl.get('title_abstract', 0))} papers\n"
             f"{fmt(p['extracted'])} extracted\n({fmt(p['primary'])} primary · {fmt(p['review'])} review)")
    arrow(a1, a2, "b", "t")
    b_tab = box(21, rows[0], 13, 7, "Tables", f"{fmt(pl['tables'])} tables")
    b_pro = box(21, rows[2], 13, 7, "Prose sections", "abstract, results, methods")
    b_bib = box(21, rows[4], 13, 7, "Bibliography", f"{fmt(pl['references'])} references")
    box(21, 5.5, 13, 5.5, "Figures", f"{fmt(pl['figures'])} listed, not read", ec="#aaa")
    for b in (b_tab, b_pro, b_bib):
        arrow(a2, b)
    c1 = box(39, rows[0], 23, 7.4, "Table profiler (LLM) + rule-based reader",
             f"column roles, units, ratio, water, uncertainty\n{fmt(ts.get('extracted', 0))} tables "
             f"extracted · {fmt(ts.get('not_relevant', 0) + ts.get('not_des', 0))} not DES-relevant")
    c2 = box(39, rows[1], 23, 7, "Paper dictionary",
             f"abbreviations and DES codes\n{fmt(pl['dictionary'])} entries")
    c3 = box(39, rows[2], 23, 7, "Prose extraction (LLM) + quote check",
             f"value must appear in a verbatim quote\n{fmt(pl['prose_ok'])} of {fmt(pl['prose_rows'])} rows accepted")
    c4 = box(39, rows[3], 23, 7, "Applications",
             f"domain, role, target per DES\n{fmt(pl['application_rows'])} rows")
    c5 = box(39, rows[4], 23, 7, "Original-source attribution",
             f"Crossref for cited references only\n{fmt(p['cited_only'])} papers credited as source")
    for a, b in [(b_tab, c1), (b_tab, c2), (b_pro, c2), (b_pro, c3), (b_tab, c4), (b_pro, c4),
                 (b_bib, c5)]:
        arrow(a, b)
    d1 = box(67, rows[0], 17, 7.4, "Components → PubChem",
             f"{fmt(pl['component_names'])} names, {fmt(pl['component_names_resolved'] or 0)} resolved\n"
             "SMILES, descriptors,\nhuman-reviewed merges")
    d2 = box(67, rows[1], 17, 7, "Component properties",
             f"PubChem text (LLM) + NIST\n{fmt(d['component_values']['total'])} values, "
             f"{d['component_values']['sources']} sources")
    d3 = box(67, rows[2], 17, 7, "QM9 descriptors", f"{fmt(d['components']['qm9'])} components")
    d4 = box(67, rows[3], 17, 7.4, "Validation",
             f"re-read every value from its cell\nplausibility ranges · {fmt(pl['duplicate_groups'] or 0)} "
             f"duplicate groups\nreview queue: {fmt(pl['review_queue'] or 0)} items")
    box(67, rows[4], 17, 7.4, "Accuracy", GOLD_SET)
    for a in (c1, c2, c3, c4):
        arrow(a, d1 if a is not c4 else d4)
    arrow(d1, d2, "b", "t")
    arrow(d2, d3, "b", "t")
    e = box(89, 30, 10, 34, "Neo4j",
            f"{fmt(pl['graph_nodes'])} nodes\n{fmt(pl['graph_rels'])}\nrelationships\n\n"
            "Component\nMixture (DES)\nMeasurement\nComponentProperty\nApplicationRecord\n"
            "Application\nPaper\nSource")
    for a in (d1, d2, d3, d4):
        ax.add_patch(FancyArrowPatch((a[0] + a[2], a[1]), (e[0], 30 + (a[1] - 30) * 0.5),
                                     arrowstyle="-|>", mutation_scale=10, lw=0.9, color="#555",
                                     connectionstyle="arc3,rad=0"))
    arrow(c5, d4)
    save(fig, out, "fig1_workflow", formats)


# ----------------------------------------------------------------------- fig 2 overview
def fig_overview(d, out, formats):
    fig, ax = plt.subplots(figsize=(15, 6.5))
    draw_values_sankey(ax, d)
    save(fig, out, "fig2a_overview_values", formats)
    fig, ax = plt.subplots(figsize=(13, 6.5))
    draw_applications_sankey(ax, d)
    save(fig, out, "fig2b_overview_applications", formats)
    fig, ax = plt.subplots(figsize=(12, 6))
    draw_components_sankey(ax, d)
    save(fig, out, "fig2c_overview_components", formats)
    fig, ax = plt.subplots(figsize=(15, 6.2))
    draw_numbers(ax, d)
    save(fig, out, "fig2d_overview_numbers", formats)
    fig = plt.figure(figsize=(15, 13))
    a1 = fig.add_axes([0, 0.5, 1, 0.47])
    draw_values_sankey(a1, d)
    a2 = fig.add_axes([0, 0, 1, 0.45])
    draw_numbers(a2, d)
    fig.text(0.0, 0.985, "a  Overview of the database", fontsize=15, fontweight="bold")
    fig.text(0.0, 0.47, "b  Key numbers", fontsize=15, fontweight="bold")
    save(fig, out, "fig2_overview", formats)


# --------------------------------------------------------------------- fig 3 distributions
def donut(ax, pairs, title, center, k=10, other="Others", colors=None, min_label=0.02):
    rest = [(n, v) for n, v in pairs if n == other]
    pairs = _top([(n, v) for n, v in pairs if n != other], k, other)
    if rest:
        pairs = [(n, v) for n, v in pairs if n != other] + \
                [(other, rest[0][1] + sum(v for n, v in pairs if n == other))]
    total = sum(v for _, v in pairs)
    colors = colors or [PALETTE[i % len(PALETTE)] if n != other else GREY
                        for i, (n, _) in enumerate(pairs)]
    wedges, _ = ax.pie([v for _, v in pairs], colors=colors, startangle=90, counterclock=False,
                       wedgeprops=dict(width=0.38, edgecolor="white", linewidth=1))
    for w, (n, v) in zip(wedges, pairs):
        if v / total < min_label:
            continue
        ang = math.radians((w.theta2 + w.theta1) / 2)
        ax.text(1.17 * math.cos(ang), 1.17 * math.sin(ang), f"{fmt(v)}\n({v / total:.0%})",
                ha="center", va="center", fontsize=7)
    ax.text(0, 0, center, ha="center", va="center", fontsize=9, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
    ax.legend(wedges, [n for n, _ in pairs], loc="center left", bbox_to_anchor=(1.12, 0.5),
              frameon=False, fontsize=8)
    ax.set_aspect("equal")


def _single(out, formats, name, draw, size=(7.5, 4.6)):
    fig, ax = plt.subplots(figsize=size)
    draw(ax)
    save(fig, out, name, formats)


def fig_distributions(d, out, formats):
    m, c, v, cv, a = d["mixtures"], d["components"], d["values"], d["component_values"], d["applications"]

    # 3a most common DES, share of all records (values + application records)
    total = v["nodes"] + a["records"]
    top = m["top_sets"][:10]
    pairs = [(" + ".join(nice(x) for x in s["components"]) + f"  ({s['papers']} papers)", s["records"])
             for s in top]
    pairs.append(("All other DES", total - sum(s["records"] for s in top)))
    _single(out, formats, "fig3a_top_des", lambda ax: donut(
        ax, pairs, "Most studied DES (water ignored)", f"Records\n{fmt(total)}", k=11,
        other="All other DES"), size=(9, 4.8))

    # 3b most common components, share of all component-in-DES occurrences
    part_of = c.get("part_of_total") or sum(t["n_des"] for t in c["top"])
    pairs = [(nice(t["name"]), t["n_des"]) for t in c["top"][:10]]
    pairs.append(("All other components", part_of - sum(n for _, n in pairs)))
    _single(out, formats, "fig3b_top_components", lambda ax: donut(
        ax, pairs, "Most used components (number of DES)", f"Uses in DES\n{fmt(part_of)}", k=11,
        other="All other components"), size=(8, 4.6))

    def roles(ax):
        top = [t for t in c["top"] if t["name"].lower() != "water"][:15][::-1]
        y = np.arange(len(top))
        ax.barh(y, [t["HBA"] for t in top], color="#d58b7d", label="as HBA")
        ax.barh(y, [t["HBD"] for t in top], left=[t["HBA"] for t in top], color="#5f7fbf", label="as HBD")
        ax.set_yticks(y, [nice(t["name"]) for t in top])
        ax.set_xlabel("Number of DES")
        ax.legend(frameon=False)
        ax.set_title("Commonly used components and their role", fontsize=11, fontweight="bold", loc="left")
    _single(out, formats, "fig3b2_component_roles", roles, size=(6.5, 5))

    cats = [(k[:1].upper() + k[1:], n) for k, n in c["category"] if k != "not identified"]
    _single(out, formats, "fig3c_component_categories", lambda ax: donut(
        ax, cats, "Categories of components", f"Identified\n{fmt(c['pubchem'])}\nof {fmt(c['total'])}",
        k=8), size=(8, 4.6))

    types = [(k, n) for k, n in m["type"] if k != "unclassified"]
    order = ["Type I/II", "Type III", "Type IV", "Type V", "inorganic-salt eutectic", "other"]
    types = sorted(types, key=lambda kv: order.index(kv[0]) if kv[0] in order else 99)
    unclass = dict(m["type"]).get("unclassified", 0)
    _single(out, formats, "fig3d_des_types", lambda ax: donut(
        ax, [(k[:1].upper() + k[1:], n) for k, n in types], "Type of DES (rule-based)",
        f"Classified\n{fmt(m['total'] - unclass)}\nof {fmt(m['total'])}", k=8, other="—"),
        size=(8, 4.6))

    _single(out, formats, "fig3e_des_properties", lambda ax: donut(
        ax, [(prop(k), n) for k, n in v["by_property"]], "DES property values",
        f"Values\n{fmt(v['nodes'])}", k=9, other="Other properties"), size=(8, 4.6))

    def comp_props(ax):
        inner = [(prop(k), n) for k, n in cv["by_property"]]
        outer = [(k.split("|")[0], k.split("|")[1], n) for k, n in cv["by_property_db"]]
        outer.sort(key=lambda t: [p for p, _ in cv["by_property"]].index(t[0]))
        cols = PALETTE[:len(inner)]
        w1, _ = ax.pie([n for _, n in inner], radius=0.68, colors=cols, startangle=90,
                       counterclock=False, wedgeprops=dict(width=0.3, edgecolor="white"))
        shade = {"pubchem": 1.0, "nist": 0.55}
        ocols = []
        for p, db, _ in outer:
            base = matplotlib.colors.to_rgb(cols[[q for q, _ in cv["by_property"]].index(p)])
            f = shade.get(db, 0.8)
            ocols.append(tuple(1 - (1 - x) * f for x in base))
        w2, _ = ax.pie([n for *_, n in outer], radius=1.0, colors=ocols, startangle=90,
                       counterclock=False, wedgeprops=dict(width=0.3, edgecolor="white"))
        for w, (p, db, n) in zip(w2, outer):
            if n / cv["total"] > 0.03:
                ang = math.radians((w.theta1 + w.theta2) / 2)
                ax.text(1.2 * math.cos(ang), 1.2 * math.sin(ang), f"{'PubChem' if db == 'pubchem' else db.upper()}\n{fmt(n)}",
                        ha="center", va="center", fontsize=7)
        ax.text(0, 0, f"Values\n{fmt(cv['total'])}", ha="center", va="center", fontsize=9, fontweight="bold")
        ax.legend(w1, [f"{k}  {fmt(n)}" for k, n in inner], loc="center left",
                  bbox_to_anchor=(1.12, 0.5), frameon=False, fontsize=8)
        ax.set_title("Component property values (inner: property, outer: database)", fontsize=11,
                     fontweight="bold", loc="left")
        ax.set_aspect("equal")
    _single(out, formats, "fig3f_component_properties", comp_props, size=(8, 4.8))

    def credited(ax):
        rows = cv["credited"][:12][::-1]
        ax.barh(range(len(rows)), [n for _, n in rows], color="#86d0c6")
        ax.set_yticks(range(len(rows)), [k if len(k) < 48 else k[:46] + "…" for k, _ in rows], fontsize=7.5)
        ax.set_xlabel("Component property values credited")
        ax.set_title("Sources credited by PubChem/NIST", fontsize=11, fontweight="bold", loc="left")
    _single(out, formats, "fig3f2_component_property_sources", credited, size=(7, 4.6))

    _single(out, formats, "fig3g_application_roles", lambda ax: donut(
        ax, [(k[:1].upper() + k[1:], n) for k, n in a["roles"]], "Role of the DES in its application",
        f"Records\nwith a role\n{fmt(sum(n for _, n in a['roles']))}", k=9), size=(8, 4.6))

    doms = [(domain(k), n) for k, n in a["by_domain"]]
    _single(out, formats, "fig3h_application_domains", lambda ax: donut(
        ax, doms, "Application domains", f"Records\n{fmt(a['records'])}", k=12,
        colors=[GREY if n.startswith("Unclassified") else PALETTE[i % len(PALETTE)]
                for i, (n, _) in enumerate(_top(doms, 12))]), size=(8.5, 4.8))

    # 3i descriptors of identified components (Peng et al. Fig. b-e)
    desc = c["descriptors"]
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    specs = [("hba", "H-bond acceptor count", np.arange(0, 12) - 0.5, 10, "#b5838d"),
             ("hbd", "H-bond donor count", np.arange(0, 8) - 0.5, 6, "#5f7fbf"),
             ("tpsa", "TPSA (Å$^2$)", np.arange(0, 165, 10), 150, "#b4aed6"),
             ("xlogp", "XLogP", np.arange(-5, 12, 1), 10, "#9cc7a2")]
    for ax, (key, label, bins, cap, col) in zip(axs.flat, specs):
        x = np.array([r[key] for r in desc if r[key] is not None], float)
        ax.hist(np.clip(x, bins[0] + 1e-9, cap), bins=bins, color=col, edgecolor="#444", lw=0.4)
        ax.set_xlabel(label)
        ax.set_ylabel("Components")
        ax.text(0.98, 0.92, f"n = {fmt(len(x))}", transform=ax.transAxes, ha="right", fontsize=8)
    fig.suptitle("Descriptors of components identified in PubChem (last bin includes higher values)",
                 fontsize=10, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    save(fig, out, "fig3i_component_descriptors", formats)

    # 3j pure-component property values: one value per component (the median of its reports)
    per = defaultdict(list)
    for comp, p, val in cv["list"]:
        if isinstance(val, (int, float)):
            per[(p, comp)].append(val)
    props = [p for p, n in cv["by_property"] if n >= 50][:3]
    fig, axs = plt.subplots(1, len(props), figsize=(4 * len(props), 3.4))
    for ax, p, col in zip(np.atleast_1d(axs), props, PALETTE):
        x = np.array([np.median(vs) for (q, _), vs in per.items() if q == p])
        lo, hi = np.percentile(x, [1, 99])
        ax.hist(x[(x >= lo) & (x <= hi)], bins=40, color=col, edgecolor="#444", lw=0.3)
        ax.set_xlabel(f"{prop(p)} ({unit(cv['units'].get(p))})")
        ax.set_ylabel("Components")
        ax.text(0.98, 0.92, f"{fmt(len(x))} components", transform=ax.transAxes, ha="right", fontsize=8)
    fig.suptitle("Pure-component property values (median per component; central 98%)", fontsize=10,
                 fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    save(fig, out, "fig3j_component_property_values", formats)

    # 3k DES property values, plausible only
    by = defaultdict(list)
    for p, val, T, ok in v["list"]:
        rng = v["plausible_range"].get(p)
        if ok and isinstance(val, (int, float)) and (not rng or rng[0] <= val <= rng[1]):
            by[p].append(val)
    top = [p for p, _ in v["by_property"][:6]]
    fig, axs = plt.subplots(2, 3, figsize=(11, 6))
    for ax, p, col in zip(axs.flat, top, PALETTE):
        x = np.array(by[p])
        if p in ("Viscosity", "Conductivity"):
            ax.hist(x[x > 0], bins=np.logspace(np.log10(x[x > 0].min()), np.log10(x.max()), 40),
                    color=col, edgecolor="#444", lw=0.3)
            ax.set_xscale("log")
        else:
            ax.hist(x, bins=40, color=col, edgecolor="#444", lw=0.3)
        ax.set_xlabel(f"{prop(p)} ({unit(v['units'].get(p))})" if unit(v["units"].get(p)) else prop(p))
        ax.set_ylabel("Values")
        ax.text(0.98, 0.92, f"n = {fmt(len(x))}", transform=ax.transAxes, ha="right", fontsize=8)
    fig.suptitle("DES property values (within plausible ranges)", fontsize=10, fontweight="bold",
                 x=0.02, ha="left")
    fig.tight_layout()
    save(fig, out, "fig3k_des_property_values", formats)

    # 3l mean XLogP vs mean TPSA of each DES's components (Peng et al. Fig. j)
    xy = np.array(m["avg_xlogp_tpsa"])
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    h = ax.hist2d(xy[:, 0], xy[:, 1], bins=[np.arange(-5, 10.5, 0.5), np.arange(0, 185, 5)],
                  cmap="Blues", norm=LogNorm(), cmin=1)
    fig.colorbar(h[3], ax=ax, label="DES per bin")
    ax.set_xlabel("Mean XLogP of components")
    ax.set_ylabel("Mean TPSA of components (Å$^2$)")
    ax.set_title(f"DES polarity map (n = {fmt(len(xy))})", fontsize=11, fontweight="bold", loc="left")
    save(fig, out, "fig3l_des_xlogp_tpsa", formats)

    def years(ax):
        ys = [(int(y), n) for y, n in d["papers"]["year"] if str(y).isdigit()]
        ax.bar([y for y, _ in ys], [n for _, n in ys], color="#ed8274", edgecolor="#444", lw=0.4)
        ax.set_xlabel("Publication year")
        ax.set_ylabel("Papers extracted")
        ax.set_title("Extracted papers by year", fontsize=11, fontweight="bold", loc="left")
    _single(out, formats, "fig3m_papers_by_year", years, size=(6, 3.6))

    _single(out, formats, "fig3n_publishers", lambda ax: donut(
        ax, d["papers"]["publisher"], "Publishers of the extracted papers",
        f"Papers\n{fmt(d['papers']['extracted'])}", k=7), size=(8, 4.6))


# ----------------------------------------------------------------- fig 4 radial (Pruyn)
def fig_radial(d, out, formats, per_domain=4, include_other=False, min_count=3):
    a = d["applications"]
    doms = [k for k, _ in a["by_domain"] if include_other or k != "other"]
    leaves = []
    for k in doms:
        ts = [(t, n) for t, n in a["targets"].get(k, []) if n >= min_count][:per_domain]
        leaves += [(k, t, n) for t, n in ts]
    doms = [k for k in doms if any(l[0] == k for l in leaves)]
    gap = 1.2
    slots = len(leaves) + gap * len(doms)
    step = 2 * math.pi / slots
    theta, cursor, dom_theta = {}, 0.0, {}
    for k in doms:
        mine = [l for l in leaves if l[0] == k]
        angles = [cursor + (i + 0.5) * step for i in range(len(mine))]
        for l, t in zip(mine, angles):
            theta[l] = t
        dom_theta[k] = (angles[0] + angles[-1]) / 2
        cursor += (len(mine) + gap) * step
    cmap = list(plt.cm.tab10.colors) + list(plt.cm.Dark2.colors)
    colors = {k: cmap[i % len(cmap)] for i, k in enumerate(doms)}
    maxn = max(n for *_, n in leaves)
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set_aspect("equal")
    ax.axis("off")
    R, r_dom = 1.0, 0.22
    recs = dict(a["by_domain"])
    for k in doms:
        t = dom_theta[k]
        dx, dy = r_dom * math.cos(t), r_dom * math.sin(t)
        ax.plot([0, dx], [0, dy], color="#bbb", lw=0.8, zorder=1)
        ax.scatter([dx], [dy], s=25 + 120 * math.sqrt(recs[k] / max(recs.values())), color=colors[k],
                   zorder=3, edgecolor="white", lw=0.5)
        mine = [l for l in leaves if l[0] == k]
        for l in mine:
            tl = theta[l]
            lx, ly = R * math.cos(tl), R * math.sin(tl)
            cx, cy = 0.72 * math.cos(t), 0.72 * math.sin(t)
            path = MPath([(dx, dy), (cx, cy), (lx, ly)], [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3])
            ax.add_patch(PathPatch(path, fc="none", ec="#c4c4c4", lw=0.8, zorder=1))
        t0 = math.degrees(theta[mine[0]] - step / 2)
        t1 = math.degrees(theta[mine[-1]] + step / 2)
        ax.add_patch(Wedge((0, 0), R + 0.012, t0, t1, width=0.012, color=colors[k], lw=0))
        deg = math.degrees(t)
        flip = 90 < deg % 360 < 270
        lab = f"{domain(k)}\n({fmt(recs[k])})"
        tx = (r_dom + 0.03) * math.cos(t)
        ty = (r_dom + 0.03) * math.sin(t)
        ax.text(tx, ty, lab, rotation=deg + 180 if flip else deg, rotation_mode="anchor",
                ha="right" if flip else "left", va="center", fontsize=7, color=colors[k], linespacing=1.0,
                fontweight="bold", path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")],
                zorder=4)
    for l, tl in theta.items():
        k, target, n = l
        length = 0.04 + 0.22 * math.sqrt(n / maxn)
        t0, t1 = math.degrees(tl - step * 0.42), math.degrees(tl + step * 0.42)
        ax.add_patch(Wedge((0, 0), R + 0.012 + length, t0, t1, width=length, color=colors[k], lw=0))
        deg = math.degrees(tl)
        flip = 90 < deg % 360 < 270
        r = R + 0.03 + length
        name = target if len(target) <= 34 else target[:32] + "…"
        name = nice(name)
        ax.text(r * math.cos(tl), r * math.sin(tl), f"{name} ({n})", rotation=deg + 180 if flip else deg,
                rotation_mode="anchor", ha="right" if flip else "left", va="center", fontsize=7,
                color=colors[k])
    ax.scatter([0], [0], s=30, color="#888", zorder=3)
    lim = R + 0.75
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    save(fig, out, "fig4_applications_radial", formats)


# ----------------------------------------------------------------- fig 5 network (Pruyn)
def fig_network(d, out, formats, min_component_des=15, label_components=14, seed=7, iters=150):
    """Pruyn et al. style: one dot per DES, linked to its hubs (components, properties,
    domains). The layout is ForceAtlas2 on the whole DES-hub graph, so hubs sharing many
    DES are pulled together. DES with exactly the same hubs are laid out as one weighted
    node (3-4k instead of 7.7k, ~1 min) and then spread around that point."""
    import networkx as nx
    links = defaultdict(set)
    for mix, hub in d["network"]:
        links[mix].add(hub)
    deg = Counter(h for hs in links.values() for h in hs)
    hubs = {h for h, n in deg.items() if not h.startswith("C:") or n >= min_component_des}
    links = {m: frozenset(hs & hubs) for m, hs in links.items() if hs & hubs}
    groups = defaultdict(list)
    for m, hs in links.items():
        groups[hs].append(m)
    G = nx.Graph()
    G.add_nodes_from(hubs)
    for i, (hs, ms) in enumerate(groups.items()):
        for h in hs:
            G.add_edge(("set", i), h, weight=len(ms))
    pos = nx.forceatlas2_layout(G, max_iter=iters, seed=seed, weight="weight", dissuade_hubs=True,
                                scaling_ratio=2.0, gravity=1.0)
    span = np.ptp(np.array(list(pos.values())), axis=0).max()
    rng = np.random.default_rng(seed)
    mixpos, segs, segcol = {}, [], []
    for i, (hs, ms) in enumerate(groups.items()):
        centre = np.array(pos[("set", i)])
        spread = 0.004 * span * math.sqrt(len(ms))
        for m in ms:
            p = centre + rng.normal(0, spread, 2)
            mixpos[m] = p
            for h in hs:
                segs.append([p, pos[h]])
                segcol.append(KIND_COLOR[{"C": "component", "P": "property", "D": "domain"}[h[0]]])
    fig, ax = plt.subplots(figsize=(14, 11))
    ax.axis("off")
    ax.add_collection(LineCollection(segs, colors=segcol, linewidths=0.2, alpha=0.05, rasterized=True))
    mp = np.array(list(mixpos.values()))
    ax.scatter(mp[:, 0], mp[:, 1], s=0.8, color="#8a8a8a", alpha=0.7, lw=0, rasterized=True)
    smax = max(deg[h] for h in hubs)
    for h in sorted(hubs, key=lambda h: deg[h]):
        kind = {"C": "component", "P": "property", "D": "domain"}[h[0]]
        ax.scatter(*pos[h], s=10 + 2500 * deg[h] / smax, color=KIND_COLOR[kind], alpha=0.85,
                   edgecolor="white", lw=0.6, zorder=3)
    comps = sorted([h for h in hubs if h[0] == "C"], key=lambda h: -deg[h])[:label_components]
    for h in [h for h in hubs if h[0] != "C"] + comps:
        name = h[2:]
        name = prop(name) if h[0] == "P" else domain(name) if h[0] == "D" else nice(name)
        ax.text(*pos[h], name, ha="center", va="center", fontsize=6.5 + 8 * math.sqrt(deg[h] / smax),
                zorder=4, path_effects=[patheffects.withStroke(linewidth=2.2, foreground="white")])
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=9, label=l) for l, c in
               [("Component", KIND_COLOR["component"]), ("DES property", KIND_COLOR["property"]),
                ("Application domain", KIND_COLOR["domain"]), ("DES (one dot each)", "#8a8a8a")]]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=9)
    ax.set_title(f"{fmt(len(mixpos))} DES linked to {fmt(len(hubs))} hubs (components in "
                 f"≥{min_component_des} DES); hub area ∝ number of DES; ForceAtlas2 layout",
                 fontsize=10, loc="left", color="#555")
    ax.autoscale()
    save(fig, out, "fig5_network", formats)


README = """# Figures — {tag}

Regenerate: `/opt/anaconda3/bin/python tools/paper_figures.py --tag {tag} --refresh`
(graph as loaded on {date}).

| figure | what is counted |
|---|---|
| fig1_workflow | pipeline stages with this run's counts; model names are constants at the top of tools/paper_figures.py |
| fig2a_overview_values | every extracted DES property row (tables + prose candidates) by the publisher of the paper it came from; rows whose status is not `ok` stay in the CSVs ("held for review"); loaded rows with the same DES, property, value, temperature and source merge into one value ("merged duplicates"); unique values split by property, then whether a measuring temperature and an uncertainty are recorded |
| fig2b_overview_applications | the same for application rows, ending in application domain |
| fig2c_overview_components | graph components by role in their DES (HBA/HBD), PubChem identity, rule-based category, and whether PubChem/NIST gave pure-component values |
| fig2d_overview_numbers | headline numbers |
| fig3a_top_des | component sets (water ignored), share of all records (DES values + application records); papers = extracted papers reporting the set |
| fig3b_top_components | number of DES each component is part of; share of all component-in-DES links |
| fig3b2_component_roles | the same, split by whether it was the HBA or HBD (water excluded) |
| fig3c_component_categories | rule-based from PubChem SMILES (tools/figure_data.py `component_category`); unidentified components excluded |
| fig3d_des_types | rule-based from component categories (`des_type`); DES with an unidentified component are "unclassified" and excluded |
| fig3e/3f | DES property values / component property values (inner ring property, outer ring database) |
| fig3f2 | sources PubChem credits for the component values (a value can credit several) |
| fig3g_application_roles | the free-text role of the DES, grouped by keyword (`ROLE_GROUPS`); only records that state one |
| fig3h_application_domains | application records per domain; "other" is unclassified |
| fig3i | H-bond acceptor/donor counts, TPSA, XLogP of PubChem-identified components |
| fig3j | pure-component melting point, boiling point, density: median per component, central 98% |
| fig3k | DES property values inside the plausible ranges of des_pipeline/config.py |
| fig3l | per DES, mean XLogP vs mean TPSA of its components (only DES whose components all have both) |
| fig3m/3n | extracted papers by year / publisher (from DOI prefix) |
| fig4_applications_radial | domains (inner dots, sized by records) and their most frequent stated targets (bars, length ∝ sqrt(records)); "other" omitted |
| fig5_network | each grey dot is one DES, linked to its components (pink; only those in ≥ N DES), the properties measured on it (blue) and its application domains (green); hub area ∝ number of DES; hubs are laid out by shared DES (cosine-normalised), each DES sits among its hubs |
"""


FIGURES = {"fig1": fig_workflow, "fig2": fig_overview, "fig3": fig_distributions,
           "fig4": fig_radial, "fig5": fig_network}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", default=os.environ.get("DES_RUN_TAG", ""))
    ap.add_argument("--refresh", action="store_true", help="re-read the graph first")
    ap.add_argument("--only", help="comma-separated, e.g. fig2,fig5")
    ap.add_argument("--formats", default="png,svg,pdf")
    args = ap.parse_args()
    out = (ROOT / "runs" / args.tag if args.tag else ROOT) / "figures"
    data = out / "figure_data.json"
    if args.refresh or not data.exists():
        env = dict(os.environ, DES_RUN_TAG=args.tag)
        subprocess.run([GRAPH_PYTHON, str(ROOT / "tools" / "figure_data.py"), "--out", str(out)],
                       env=env, check=True, cwd=ROOT)
    d = json.loads(data.read_text())
    formats = [f.strip() for f in args.formats.split(",")]
    wanted = args.only.split(",") if args.only else list(FIGURES)
    for name in wanted:
        FIGURES[name.strip()](d, out, formats)
    (out / "README.md").write_text(README.format(tag=d["run_tag"], date=d["date"]))
    print(f"wrote {out}/")


if __name__ == "__main__":
    main()
