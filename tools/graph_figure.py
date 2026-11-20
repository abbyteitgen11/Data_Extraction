#!/usr/bin/env python3
"""
Draw the graph schema with every circle and arrow sized by how much data it holds.

    DES_RUN_TAG=corpus999 python tools/graph_report.py           # writes figure_counts.json
    /opt/anaconda3/bin/python tools/graph_figure.py runs/corpus999/graph_report

Needs only matplotlib (the GraphDataBase env has none; the base Anaconda python does),
because it reads the counts graph_report.py saved rather than querying Neo4j.

Writes, next to figure_counts.json:
  graph_figure.png / .svg / .pdf     no text at all
  graph_figure_key.png               the same figure labelled, to annotate from

Circle AREA is proportional to the number of nodes, so a type with twice the data has
twice the ink; arrow width is proportional to the square root of the number of
relationships (linear width would make PART_OF 7,000 times wider than the thinnest).
DES measurements and component properties (PubChem/NIST values for pure components)
are split into one circle per property, and application records into one per domain,
each a shade of the group's colour, darkest = most data. Component properties and
Sources are not in the hand-drawn schema, so they get their own colours.
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch   # noqa: E402

# Colours and positions read off the hand-drawn schema (positions in its pixel frame,
# y pointing down, 2000 x 1122).
COLOR = {"Component": "#def1d3", "Mixture": "#9fd77f", "Measurement": "#bab3f4",
         "Paper": "#ed8274", "ApplicationRecord": "#dba1d9", "Application": "#489cd0",
         # not in the hand-drawn schema; chosen to stay clear of the six above
         "ComponentProperty": "#f6cf8a", "Source": "#86d0c6"}
POS = {"Component": (430, 470), "Mixture": (860, 300), "Measurement": (1340, 290),
       "Paper": (1720, 640), "ApplicationRecord": (930, 830), "Application": (1530, 960),
       "ComponentProperty": (330, 830), "Source": (580, 1080)}
WIDTH, HEIGHT = 2000, 1122
R_MAX = 135          # radius of the largest circle or cluster, in that frame
R_MIN = 5            # so a type with 1 node is still visible
W_MAX = 16           # width (points) of the busiest arrow
GAP = 3              # between circles inside a cluster


def shades(base, n):
    """n shades of `base`, dark to light, staying pastel: blended in RGB from a muted
    darker version of it (most data) to near-white (least)."""
    rgb = [int(base[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    if n == 1:
        return [tuple(rgb)]
    dark = [c * 0.82 for c in rgb]          # lower = darker largest circles
    light = [c + (1 - c) * 0.65 for c in rgb]
    return [tuple(d + (l - d) * i / (n - 1) for d, l in zip(dark, light)) for i in range(n)]


def pack(radii):
    """Greedy circle packing around (0, 0): each circle, largest first, goes to the
    free spot touching an already placed circle that is closest to the centre."""
    placed = []
    for r in radii:
        if not placed:
            placed.append((0.0, 0.0, r))
            continue
        best = None
        for (x, y, R) in placed:
            for k in range(72):
                a = 2 * math.pi * k / 72
                cx, cy = x + (R + r + GAP) * math.cos(a), y + (R + r + GAP) * math.sin(a)
                if all(math.hypot(cx - px, cy - py) >= pr + r + GAP - 1e-6
                       for px, py, pr in placed):
                    d = math.hypot(cx, cy)
                    if best is None or d < best[0]:
                        best = (d, cx, cy)
        placed.append((best[1], best[2], r))
    # centre the cluster on its bounding circle
    mx = sum(x * r * r for x, _, r in placed) / sum(r * r for *_, r in placed)
    my = sum(y * r * r for _, y, r in placed) / sum(r * r for *_, r in placed)
    return [(x - mx, y - my, r) for x, y, r in placed]


def layout(counts):
    """-> {node id: (x, y, r, colour)} and the scale used."""
    groups = {"Measurement": counts["measurements"], "ApplicationRecord": counts["app_records"],
              "ComponentProperty": counts.get("component_properties") or {}}
    groups = {g: parts for g, parts in groups.items() if parts}
    # One scale for everything, chosen so the biggest single node or cluster gets R_MAX.
    biggest = max(list(counts["nodes"].values()) +
                  [sum(v.values()) / 0.75 for v in groups.values()])   # packing wastes ~25%
    k = R_MAX / math.sqrt(biggest)
    rad = lambda n: max(R_MIN, k * math.sqrt(n))                       # noqa: E731

    nodes = {}
    for name, n in counts["nodes"].items():
        x, y = POS[name]
        nodes[name] = (x, y, rad(n), COLOR[name])
    for group, parts in groups.items():
        items = sorted(parts.items(), key=lambda kv: -kv[1])
        cols = shades(COLOR[group], len(items))
        cx, cy = POS[group]
        for (sub, n), (dx, dy, r), c in zip(items, pack([rad(n) for _, n in items]), cols):
            nodes[f"{group}:{sub}"] = (cx + dx, cy + dy, r, c)
    return nodes


def edges(counts):
    e = counts["edges"]
    out = [("Component", "Mixture", e["PART_OF"], 0.0),
           ("Mixture", "Paper", e["Mixture_REPORTED_IN"], 0.0)]
    for p, n in e["HAS_PROPERTY"].items():
        out.append(("Mixture", f"Measurement:{p}", n, 0.0))
    for p, n in e["Measurement_REPORTED_IN"].items():
        out.append((f"Measurement:{p}", "Paper", n, 0.0))
    for d, n in e["HAS_APPLICATION"].items():
        out.append(("Mixture", f"ApplicationRecord:{d}", n, 0.0))
    for d, n in e["IN_DOMAIN"].items():
        out.append((f"ApplicationRecord:{d}", "Application", n, 0.0))
    for d, n in e["AppRecord_REPORTED_IN"].items():
        out.append((f"ApplicationRecord:{d}", "Paper", n, 0.0))
    for p, n in e.get("HAS_COMPONENT_PROPERTY", {}).items():
        out.append(("Component", f"ComponentProperty:{p}", n, 0.0))
    for p, n in e.get("ComponentProperty_REPORTED_IN", {}).items():
        out.append((f"ComponentProperty:{p}", "Source", n, 0.0))
    return [x for x in out if x[2]]


def draw(counts, labels=False):
    nodes = layout(counts)
    rels = edges(counts)
    wk = W_MAX / math.sqrt(max(n for *_, n, _ in rels))

    fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)                 # y down, as in the hand-drawn frame
    ax.set_aspect("equal")
    ax.axis("off")

    for a, b, n, bend in rels:
        xa, ya, ra, _ = nodes[a]
        xb, yb, rb, _ = nodes[b]
        d = math.hypot(xb - xa, yb - ya)
        ux, uy = (xb - xa) / d, (yb - ya) / d
        w = max(0.4, wk * math.sqrt(n))
        ax.add_patch(FancyArrowPatch(
            (xa + ux * ra, ya + uy * ra), (xb - ux * rb, yb - uy * rb),
            connectionstyle=f"arc3,rad={bend}",
            arrowstyle=f"-|>,head_length={min(0.9, 0.3 + w * 0.04):.2f},"
                       f"head_width={min(0.5, 0.15 + w * 0.025):.2f}",
            mutation_scale=20, lw=w, color="black", alpha=0.55 if ":" in a + b else 0.85, zorder=1,
            shrinkA=0, shrinkB=2))

    for name, (x, y, r, c) in nodes.items():
        ax.add_patch(Circle((x, y), r, facecolor=c, edgecolor="black",
                            lw=1.6 if ":" not in name else 0.8, zorder=2))
        if labels:
            ax.text(x, y, name.split(":")[-1], ha="center", va="center", fontsize=6, zorder=3)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("report_dir", type=Path, help="the graph_report folder with figure_counts.json")
    args = ap.parse_args()
    counts = json.loads((args.report_dir / "figure_counts.json").read_text())

    fig = draw(counts)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(args.report_dir / f"graph_figure.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight", pad_inches=0.1,
                    transparent=ext != "png", facecolor="white")
    plt.close(fig)
    fig = draw(counts, labels=True)
    fig.savefig(args.report_dir / "graph_figure_key.png", dpi=200, facecolor="white",
                bbox_inches="tight", pad_inches=0.1)
    print(f"wrote {args.report_dir}/graph_figure.png .svg .pdf and graph_figure_key.png")


if __name__ == "__main__":
    main()
