#!/usr/bin/env python3
"""
Summarise the graph that is loaded in Neo4j and draw it in ways Bloom can't.

    DES_RUN_TAG=corpus999 python tools/graph_report.py     # -> runs/corpus999/graph_report/
    python tools/graph_report.py --out some/dir            # anywhere else
    python tools/graph_report.py --mixture "<key or name>" # pick the example DES yourself

Everything is read from the running database (whatever was loaded last), so re-run it
after every `--steps graph` -- e.g. once the full corpus is loaded.

Writes:
  summary.md / summary.json   node and relationship counts, unique components, DES,
                              properties, applications, component properties, sources
  schema.html                 the schema (one box per node type, every property with its
                              fill rate and an example) and one real DES drawn the same way
  schema.mmd                  the schema as a Mermaid ER diagram (renders on GitHub/VS Code)
  component_network.html      interactive overview: components linked when they appear in
                              the same DES, sized by how many DES use them
  component_network.gexf      the same network, for Gephi
  figure_counts.json          the counts tools/graph_figure.py draws

Property nodes come in two kinds that share their labels (:Density, :Melting_point...):
a value measured on a DES hangs off a Mixture, and a value for a pure compound from
PubChem or NIST hangs off a Component (origin = 'component'). Everything here keeps the
two apart -- as "Measurement" and "ComponentProperty" -- because counting them together
would report urea's melting point as DES data.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from des_pipeline import build_graph, config     # noqa: E402

# Where each node type sits in the diagrams, left to right. Types not listed here (a
# label added later) are appended to the last column, so the diagram never drops one.
COLUMNS = [["Source"], ["ComponentProperty", "Component"], ["Mixture"],
           ["Measurement", "ApplicationRecord"], ["Paper", "Application"]]
ROLE_OF = {"Component": "a compound (PubChem identity where found)",
           "Mixture": "one DES: a set of components at one ratio (+ water/additive)",
           "Measurement": "one measured value of one property of a DES",
           "ComponentProperty": "one value of one property of a pure component, from PubChem or NIST",
           "ApplicationRecord": "one reported use of one DES in one paper",
           "Application": "an application domain",
           "Paper": "an extracted paper or a reference it cites",
           "Source": "the database a component property came from, or the source it credits"}
# Categorical fields worth listing value by value; identifiers and free text are not.
NOT_CATEGORICAL = {"key", "name", "doi", "title", "authors", "raw", "source_text", "smiles",
                   "inchikey", "cas", "formula", "matched_name", "member_keys", "pages",
                   "volume", "issue", "subject", "merged_from", "plausibility_note",
                   "ref_numbers", "journal", "year", "pressure_raw", "condition_note"}
PROPS = list(config.PROPERTY_NAMES)

# A node's kind: its label, except property nodes, which are "Measurement:<label>" or
# "ComponentProperty:<label>" depending on what they describe.
KIND = ("CASE WHEN labels({v})[0] IN $props THEN "
        "(CASE WHEN {v}.origin = 'component' THEN 'ComponentProperty:' ELSE 'Measurement:' END)"
        " + labels({v})[0] ELSE labels({v})[0] END")


def group_of(kind):
    return kind.split(":")[0]


def label_of(kind):
    return kind.split(":")[-1]


def display(kind):
    """'Measurement:Density' -> 'Density'; 'ComponentProperty:Density' -> 'Density (component)'."""
    if kind.startswith("ComponentProperty:"):
        return f"{label_of(kind)} (component)"
    return label_of(kind)


def _match(kind, var="n"):
    """MATCH clause selecting exactly the nodes of one kind."""
    label = label_of(kind)
    if group_of(kind) == "ComponentProperty":
        return f"MATCH ({var}:`{label}`) WHERE {var}.origin = 'component'"
    if group_of(kind) == "Measurement":
        return f"MATCH ({var}:`{label}`) WHERE coalesce({var}.origin, '') <> 'component'"
    return f"MATCH ({var}:`{label}`) WHERE true"


def _short(v, n=70):
    if isinstance(v, list):
        v = "[" + ", ".join(map(str, v[:3])) + (", …" if len(v) > 3 else "") + "]"
    if isinstance(v, float):
        v = f"{v:.6g}"
    s = str(v).replace("\n", " ")
    return s if len(s) <= n else s[:n - 1] + "…"


def _type(v):
    return {bool: "bool", int: "int", float: "float", str: "text", list: "list"}.get(type(v), "?")


class Graph:
    def __init__(self):
        self.driver = build_graph._driver()

    def q(self, cypher, **params):
        return [dict(r) for r in self.driver.execute_query(
            cypher, params, database_=config.NEO4J_DATABASE).records]

    def close(self):
        self.driver.close()


# --------------------------------------------------------------------------- inventory
def inventory(g):
    """-> nodes {kind: n}, rels [{t, a, b, n}] between kinds, DES measurement labels,
    component property labels -- each list sorted by count."""
    nodes = {r["k"]: r["n"] for r in g.q(
        f"MATCH (n) RETURN {KIND.format(v='n')} AS k, count(*) AS n ORDER BY n DESC", props=PROPS)}
    if not nodes:
        sys.exit("the database is empty -- run `--steps graph` first")
    rels = g.q(f"MATCH (a)-[r]->(b) RETURN type(r) AS t, {KIND.format(v='a')} AS a, "
               f"{KIND.format(v='b')} AS b, count(*) AS n ORDER BY n DESC", props=PROPS)
    measure = [label_of(k) for k in nodes if group_of(k) == "Measurement"]
    cprops = [label_of(k) for k in nodes if group_of(k) == "ComponentProperty"]
    return nodes, rels, measure, cprops


def field_stats(g, kinds, total):
    """Every property on nodes of these kinds: fill rate, type, example, values."""
    filled, samples = Counter(), defaultdict(list)
    for kind in kinds:
        for r in g.q(f"{_match(kind)} UNWIND keys(n) AS k "
                     "WITH n, k WHERE NOT n[k] IN ['', []] "
                     "WITH k, count(*) AS c, collect(n[k])[..40] AS s RETURN k, c, s"):
            filled[r["k"]] += r["c"]
            samples[r["k"]] += r["s"]
    out = []
    for k in sorted(filled, key=lambda k: (-filled[k], k)):
        vals = [v for v in samples[k] if v not in ("", None, [])]
        example = next((v for v in vals if not isinstance(v, str) or 3 <= len(v) <= 90),
                       vals[0] if vals else "")
        f = {"name": k, "filled": filled[k], "pct": round(100 * filled[k] / total, 1),
             "type": _type(example), "example": _short(example)}
        if k not in NOT_CATEGORICAL and f["type"] in ("text", "bool"):
            values = Counter()
            for kind in kinds:
                for v in g.q(f"{_match(kind)} AND n[$k] IS NOT NULL AND n[$k] <> '' "
                             "RETURN n[$k] AS v, count(*) AS c ORDER BY c DESC LIMIT 9", k=k):
                    values[v["v"]] += v["c"]
            if len(values) <= 8:
                f["values"] = [[_short(v, 40), c] for v, c in values.most_common()]
        out.append(f)
    return out


def rel_field_stats(g, rtype):
    total = g.q(f"MATCH ()-[r:`{rtype}`]->() RETURN count(r) AS n")[0]["n"]
    out = []
    for r in g.q(f"MATCH ()-[r:`{rtype}`]->() UNWIND keys(r) AS k "
                 "WITH r, k WHERE NOT r[k] IN ['', []] "
                 "WITH k, count(*) AS c, collect(r[k])[..40] AS s RETURN k, c, s ORDER BY c DESC"):
        vals = [v for v in r["s"] if v not in ("", None, [])]
        f = {"name": r["k"], "filled": r["c"], "pct": round(100 * r["c"] / total, 1),
             "type": _type(vals[0]) if vals else "?",
             "example": _short(vals[0]) if vals else ""}
        if f["type"] in ("text", "bool") and r["k"] not in NOT_CATEGORICAL:
            values = g.q(f"MATCH ()-[r:`{rtype}`]->() WHERE r[$k] IS NOT NULL "
                         "RETURN r[$k] AS v, count(*) AS c ORDER BY c DESC LIMIT 9", k=r["k"])
            if len(values) <= 8:
                f["values"] = [[_short(v["v"], 40), v["c"]] for v in values]
        out.append(f)
    return out


def schema(g, nodes, rels):
    groups = defaultdict(list)
    for kind in nodes:
        groups[group_of(kind)].append(kind)
    boxes = {}
    for grp, kinds in groups.items():
        total = sum(nodes[k] for k in kinds)
        split = grp in ("Measurement", "ComponentProperty")
        boxes[grp] = {"id": grp, "count": total, "role": ROLE_OF.get(grp, ""),
                      "labels": [[label_of(k), nodes[k]] for k in kinds] if split else [],
                      "fields": field_stats(g, kinds, total)}
    edges = {}
    for r in rels:
        a, b = group_of(r["a"]), group_of(r["b"])
        t = ("HAS_<PROPERTY>" if b in ("Measurement", "ComponentProperty")
             and r["t"].startswith("HAS_") else r["t"])
        e = edges.setdefault((a, t, b), {"from": a, "to": b, "type": t, "count": 0})
        e["count"] += r["n"]
    rel_fields = {t: rel_field_stats(g, t) for t in sorted({r["t"] for r in rels})
                  if not t.startswith("HAS_")}
    for e in edges.values():
        e["fields"] = rel_fields.get(e["type"], [])
    return boxes, list(edges.values())


# --------------------------------------------------------------------------- headline
def headline(g, nodes, rels, measure, cprops):
    one = lambda c, **p: g.q(c, **p)[0]                                  # noqa: E731
    comp = one("MATCH (c:Component) RETURN count(c) AS total, "
               "count(c.cid) AS pubchem, count(c.smiles) AS smiles, count(c.qm9_id) AS qm9")
    roles = one("MATCH (c:Component)-[r:PART_OF]->() WITH c, collect(DISTINCT r.role) AS rs "
                "RETURN sum(CASE WHEN rs = ['HBA'] THEN 1 ELSE 0 END) AS hba_only, "
                "sum(CASE WHEN rs = ['HBD'] THEN 1 ELSE 0 END) AS hbd_only, "
                "sum(CASE WHEN size(rs) > 1 THEN 1 ELSE 0 END) AS both")
    with_cprops = one("MATCH (c:Component) WHERE EXISTS { MATCH (c)-[h]->() "
                      "WHERE type(h) STARTS WITH 'HAS_' } RETURN count(c) AS n")["n"]
    mix = one("""
        MATCH (m:Mixture)
        RETURN count(m) AS total,
          sum(CASE WHEN EXISTS { MATCH (m)-[h]->(x) WHERE type(h) STARTS WITH 'HAS_'
                                 AND NOT x:ApplicationRecord } THEN 1 ELSE 0 END) AS with_props,
          sum(CASE WHEN EXISTS { (m)-[:HAS_APPLICATION]->() } THEN 1 ELSE 0 END) AS with_apps,
          sum(CASE WHEN EXISTS { MATCH (m)-[h]->(x) WHERE type(h) STARTS WITH 'HAS_'
                                 AND NOT x:ApplicationRecord }
                    AND EXISTS { (m)-[:HAS_APPLICATION]->() } THEN 1 ELSE 0 END) AS with_both""")
    origin = {r["o"]: r["n"] for r in g.q("MATCH (m:Mixture) RETURN m.origin AS o, count(*) AS n")}
    # The same DES at different ratios or water contents is several Mixture nodes; the
    # "systems" count collapses those to one per component set.
    systems = one("""
        MATCH (c:Component)-[:PART_OF]->(m:Mixture)
        WITH m, c.name AS n ORDER BY n
        WITH m, collect(n) AS all, [x IN collect(n) WHERE toLower(x) <> 'water'] AS dry
        RETURN count(DISTINCT all) AS systems, count(DISTINCT dry) AS systems_ignoring_water""")
    sizes = {r["k"]: r["n"] for r in g.q(
        "MATCH (m:Mixture) WITH COUNT { (m)<-[:PART_OF]-() } AS k RETURN k, count(*) AS n ORDER BY k")}
    props = []
    for label in measure:
        r = one(f"""
            MATCH (m:Mixture)-[]->(x:`{label}`)
            OPTIONAL MATCH (x)-[:REPORTED_IN]->(p:Paper)
            RETURN count(DISTINCT x) AS values, count(DISTINCT m) AS mixtures,
                   count(DISTINCT p) AS papers, collect(DISTINCT x.unit)[..3] AS units,
                   min(x.value) AS lo, max(x.value) AS hi,
                   count(DISTINCT CASE WHEN x.uncertainty IS NOT NULL THEN x END) AS with_uncertainty,
                   count(DISTINCT CASE WHEN x.plausible = false THEN x END) AS implausible""")
        props.append({"property": label, **r})
    comp_props = []
    for label in cprops:
        r = one(f"""
            MATCH (c:Component)-[]->(x:`{label}`)
            OPTIONAL MATCH (x)-[:REPORTED_IN]->(s:Source)
            RETURN count(DISTINCT x) AS values, count(DISTINCT c) AS components,
                   count(DISTINCT s) AS sources, collect(DISTINCT x.unit)[..3] AS units,
                   min(x.value) AS lo, max(x.value) AS hi""")
        comp_props.append({"property": label, **r})
    sources = g.q("""
        MATCH (s:Source)<-[:REPORTED_IN]-(x)
        RETURN s.name AS source, s.kind AS kind, count(DISTINCT x) AS values
        ORDER BY values DESC""")
    apps = g.q("""
        MATCH (a:Application)<-[:IN_DOMAIN]-(r:ApplicationRecord)<-[:HAS_APPLICATION]-(m:Mixture)
        OPTIONAL MATCH (r)-[:REPORTED_IN]->(p:Paper)
        RETURN a.domain AS domain, count(DISTINCT r) AS records, count(DISTINCT m) AS mixtures,
               count(DISTINCT p) AS papers ORDER BY records DESC""")
    papers = one("""
        MATCH (p:Paper)
        RETURN count(p) AS total,
          sum(CASE WHEN p.extracted THEN 1 ELSE 0 END) AS extracted,
          sum(CASE WHEN p.extracted AND p.role = 'review' THEN 1 ELSE 0 END) AS extracted_review,
          sum(CASE WHEN p.extracted AND p.role <> 'review' THEN 1 ELSE 0 END) AS extracted_primary,
          sum(CASE WHEN p.extracted THEN 0 ELSE 1 END) AS cited_only,
          sum(CASE WHEN EXISTS { ()-[:REPORTED_IN]->(p) } THEN 1 ELSE 0 END) AS with_data""")
    return {
        "date": date.today().isoformat(), "run_tag": config.RUN_TAG or "(default data/)",
        "nodes_total": sum(nodes.values()), "relationships_total": sum(r["n"] for r in rels),
        "nodes": {display(k): n for k, n in nodes.items()},
        "relationships": [{**r, "a": display(r["a"]), "b": display(r["b"])} for r in rels],
        "components": {**comp, **roles, "with_component_properties": with_cprops},
        "mixtures": {**mix, "by_origin": origin, **systems, "by_n_components": sizes},
        "measurements_total": sum(n for k, n in nodes.items() if group_of(k) == "Measurement"),
        "properties": props,
        "component_properties_total": sum(n for k, n in nodes.items()
                                          if group_of(k) == "ComponentProperty"),
        "component_properties": comp_props,
        "sources": sources,
        "applications": apps,
        "papers": papers,
    }


# --------------------------------------------------------------------------- example
def pick_example(g, wanted=None):
    if wanted:
        rows = g.q("MATCH (m:Mixture) WHERE m.key = $w OR m.name = $w RETURN m.key AS k LIMIT 1",
                   w=wanted)
        if not rows:
            sys.exit(f"no Mixture with key or name {wanted!r}")
        return rows[0]["k"]
    # The DES that shows the most of the schema at once: measured properties, an
    # application, an extracted paper, a component with database properties, and few
    # enough components to stay readable.
    rows = g.q("""
        MATCH (m:Mixture)-[h]->(x)
        WHERE type(h) STARTS WITH 'HAS_' AND NOT x:ApplicationRecord
          AND EXISTS { (m)-[:HAS_APPLICATION]->() }
          AND EXISTS { (m)-[:REPORTED_IN]->(:Paper {extracted: true}) }
          AND COUNT { (m)<-[:PART_OF]-() } <= 3
        WITH m, count(DISTINCT labels(x)[0]) AS kinds,
             COUNT { MATCH (m)<-[:PART_OF]-(:Component)-[c]->()-[:REPORTED_IN]->(:Source)
                     WHERE type(c) STARTS WITH 'HAS_' } > 0 AS has_db
        RETURN m.key AS k ORDER BY has_db DESC, kinds DESC, m.name LIMIT 1""")
    if not rows:     # nothing has both; fall back to any mixture with a measurement
        rows = g.q("MATCH (m:Mixture)-[h]->(x) WHERE type(h) STARTS WITH 'HAS_' "
                   "RETURN m.key AS k LIMIT 1")
    return rows[0]["k"] if rows else None


def _kind_of(node):
    label = list(node.labels)[0]
    if label in PROPS:
        return ("ComponentProperty:" if node.get("origin") == "component"
                else "Measurement:") + label
    return label


def example(g, key):
    nodes, edges = {}, []

    def add(n):
        nid = n.element_id
        if nid not in nodes:
            kind = _kind_of(n)
            props = dict(n)
            title = (props.get("name") or props.get("domain") or props.get("title")
                     or props.get("property") or props.get("key") or label_of(kind))
            if group_of(kind) in ("Measurement", "ComponentProperty"):
                title = (f"{label_of(kind)} = {_short(props.get('value'))} "
                         f"{props.get('unit') or ''}").strip()
            nodes[nid] = {"id": nid, "label": label_of(kind), "group": group_of(kind),
                          "title": _short(title, 60),
                          "props": [[k, _short(v)] for k, v in sorted(props.items())]}
        return nid

    def link(a, b, rel):
        edges.append({"from": a, "to": b, "type": rel.type,
                      "props": [[k, _short(v, 40)] for k, v in sorted(dict(rel).items())]})

    run = lambda c, **p: g.driver.execute_query(c, p, database_=config.NEO4J_DATABASE).records  # noqa: E731
    rec = run("""
        MATCH (m:Mixture {key: $k})
        OPTIONAL MATCH (c:Component)-[pr:PART_OF]->(m)
        RETURN m, collect([c, pr]) AS comps""", k=key)[0]
    m = add(rec["m"])
    comps = []
    for c, pr in rec["comps"]:
        if c is not None:
            comps.append(add(c))
            link(comps[-1], m, pr)
    # One database value per component (and the sources it credits), one DES value per
    # property type (up to 3), one application record, and the papers they cite --
    # enough to show every relationship once.
    for cid in comps:
        for r in run("""
            MATCH (c)-[h]->(x)-[rs:REPORTED_IN]->(s:Source)
            WHERE elementId(c) = $id AND type(h) STARTS WITH 'HAS_'
            WITH h, x, collect([rs, s])[..2] AS srcs
            RETURN h, x, srcs LIMIT 1""", id=cid):
            x = add(r["x"])
            link(cid, x, r["h"])
            for rs, s in r["srcs"]:
                link(x, add(s), rs)
    for r in run("""
        MATCH (m:Mixture {key: $k})-[h]->(x)
        WHERE type(h) STARTS WITH 'HAS_' AND NOT x:ApplicationRecord
        WITH labels(x)[0] AS l, collect([h, x])[0] AS hx
        RETURN hx[0] AS h, hx[1] AS x LIMIT 3""", k=key):
        x = add(r["x"])
        link(m, x, r["h"])
        for p in run("MATCH (x)-[rp:REPORTED_IN]->(p) WHERE elementId(x) = $id "
                     "RETURN rp, p LIMIT 2", id=x):
            link(x, add(p["p"]), p["rp"])
    for r in run("""
        MATCH (m:Mixture {key: $k})-[ha:HAS_APPLICATION]->(a)-[d:IN_DOMAIN]->(dom)
        OPTIONAL MATCH (a)-[rp:REPORTED_IN]->(p:Paper)
        RETURN ha, a, d, dom, rp, p LIMIT 1""", k=key):
        a = add(r["a"])
        link(m, a, r["ha"])
        link(a, add(r["dom"]), r["d"])
        if r["p"] is not None:
            link(a, add(r["p"]), r["rp"])
    for r in run("MATCH (m:Mixture {key: $k})-[rp:REPORTED_IN]->(p:Paper) RETURN rp, p LIMIT 2",
                 k=key):
        link(m, add(r["p"]), r["rp"])
    cypher = (f"MATCH (m:Mixture {{key: '{key}'}})\n"
              "OPTIONAL MATCH a = (:Component)-[:PART_OF]->(m)\n"
              "OPTIONAL MATCH b = (m)-->(x)-->(y) WHERE NOT y:Mixture\n"
              "OPTIONAL MATCH c = (m)<-[:PART_OF]-(:Component)-->(:Melting_point)-->(:Source)\n"
              "RETURN a, b, c LIMIT 80")
    return {"key": key, "nodes": list(nodes.values()), "edges": edges, "cypher": cypher}


# --------------------------------------------------------------------------- figure
def figure_counts(g, nodes, rels):
    """Counts for tools/graph_figure.py: node types; DES measurements and component
    properties split by property; application records split by domain; and every
    relationship at that same split."""
    def by(cypher):
        return {r["k"]: r["n"] for r in g.q(cypher)}

    def rel(t, a, b):
        return sum(r["n"] for r in rels if r["t"] == t and r["a"] == a and r["b"] == b)

    single = {k: nodes[k] for k in ("Component", "Mixture", "Paper", "Application", "Source")
              if k in nodes}
    measure = [k for k in nodes if group_of(k) == "Measurement"]
    cprops = [k for k in nodes if group_of(k) == "ComponentProperty"]
    return {
        "nodes": single,
        "measurements": {label_of(k): nodes[k] for k in measure},
        "component_properties": {label_of(k): nodes[k] for k in cprops},
        "app_records": by("MATCH (r:ApplicationRecord) RETURN r.domain AS k, count(*) AS n"),
        "edges": {
            "PART_OF": rel("PART_OF", "Component", "Mixture"),
            "Mixture_REPORTED_IN": rel("REPORTED_IN", "Mixture", "Paper"),
            "HAS_PROPERTY": {label_of(k): rel(f"HAS_{label_of(k).upper()}", "Mixture", k)
                             for k in measure},
            "Measurement_REPORTED_IN": {label_of(k): rel("REPORTED_IN", k, "Paper")
                                        for k in measure},
            "HAS_COMPONENT_PROPERTY": {label_of(k): rel(f"HAS_{label_of(k).upper()}",
                                                        "Component", k) for k in cprops},
            "ComponentProperty_REPORTED_IN": {label_of(k): rel("REPORTED_IN", k, "Source")
                                              for k in cprops},
            "HAS_APPLICATION": by("MATCH (:Mixture)-[e:HAS_APPLICATION]->(r) "
                                  "RETURN r.domain AS k, count(e) AS n"),
            "IN_DOMAIN": by("MATCH (r:ApplicationRecord)-[e:IN_DOMAIN]->() "
                            "RETURN r.domain AS k, count(e) AS n"),
            "AppRecord_REPORTED_IN": by("MATCH (r:ApplicationRecord)-[e:REPORTED_IN]->() "
                                        "RETURN r.domain AS k, count(e) AS n"),
        },
    }


# --------------------------------------------------------------------------- network
def component_network(g):
    nodes = g.q("""
        MATCH (c:Component)-[r:PART_OF]->(m:Mixture)
        WITH c, count(DISTINCT m) AS mixtures,
             sum(CASE r.role WHEN 'HBA' THEN 1 ELSE 0 END) AS hba,
             sum(CASE r.role WHEN 'HBD' THEN 1 ELSE 0 END) AS hbd,
             collect(DISTINCT m) AS ms
        UNWIND ms AS m
        OPTIONAL MATCH (m)-[h]->(x) WHERE type(h) STARTS WITH 'HAS_' AND NOT x:ApplicationRecord
        OPTIONAL MATCH (m)-[:HAS_APPLICATION]->(a)
        WITH c, mixtures, hba, hbd, count(DISTINCT x) AS values, count(DISTINCT a) AS apps
        OPTIONAL MATCH (c)-[hc]->(y) WHERE type(hc) STARTS WITH 'HAS_'
        RETURN c.name AS id, c.cid AS cid, c.formula AS formula, c.molecular_weight AS mw,
               c.melting_point_C AS mp, c.boiling_point_C AS bp, c.qm9_id IS NOT NULL AS qm9,
               mixtures, hba, hbd, values, apps, count(DISTINCT y) AS db_values""")
    links = g.q("""
        MATCH (a:Component)-[:PART_OF]->(m:Mixture)<-[:PART_OF]-(b:Component)
        WHERE a.name < b.name
        RETURN a.name AS source, b.name AS target, count(DISTINCT m) AS w""")
    return {"nodes": nodes, "links": links}


def write_gexf(net, path):
    import networkx as nx
    G = nx.Graph()
    for n in net["nodes"]:
        G.add_node(n["id"], label=n["id"], mixtures=n["mixtures"], values=n["values"],
                   applications=n["apps"], hba=n["hba"], hbd=n["hbd"],
                   database_values=n["db_values"], pubchem=n["cid"] is not None,
                   qm9=bool(n["qm9"]))
    for e in net["links"]:
        G.add_edge(e["source"], e["target"], weight=e["w"])
    nx.write_gexf(G, path)


# --------------------------------------------------------------------------- outputs
def summary_md(h):
    L = [f"# Graph summary — {h['run_tag']} ({h['date']})", "",
         f"**{h['nodes_total']:,} nodes, {h['relationships_total']:,} relationships.**", ""]
    c, m, p = h["components"], h["mixtures"], h["papers"]
    L += ["| | count |", "|---|---:|",
          f"| Unique components | {c['total']:,} |",
          f"| — identified in PubChem (CID) | {c['pubchem']:,} |",
          f"| — with database property values / with QM9 data | {c['with_component_properties']:,} / {c['qm9']:,} |",
          f"| — used only as HBA / only as HBD / as both | {c['hba_only']:,} / {c['hbd_only']:,} / {c['both']:,} |",
          f"| DES (Mixture nodes: components + ratio + water/additive) | {m['total']:,} |",
          f"| — distinct component sets | {m['systems']:,} |",
          f"| — distinct component sets, water ignored | {m['systems_ignoring_water']:,} |",
          f"| — with measured properties / applications / both | {m['with_props']:,} / {m['with_apps']:,} / {m['with_both']:,} |",
          f"| DES property types | {len(h['properties'])} |",
          f"| DES property values (measurement nodes) | {h['measurements_total']:,} |",
          f"| — with an uncertainty | {sum(r['with_uncertainty'] for r in h['properties']):,} |",
          f"| Component property values (PubChem/NIST) | {h['component_properties_total']:,} |",
          f"| Sources (databases and credited sources) | {len(h['sources']):,} |",
          f"| Application domains | {len(h['applications'])} |",
          f"| Application records | {sum(a['records'] for a in h['applications']):,} |",
          f"| Papers extracted (primary / review) | {p['extracted']:,} ({p['extracted_primary']:,} / {p['extracted_review']:,}) |",
          f"| Papers only cited as data sources | {p['cited_only']:,} |", ""]
    L += ["DES by origin: " + ", ".join(f"{k} {v:,}" for k, v in m["by_origin"].items()),
          "DES by number of components: " + ", ".join(f"{k}: {v:,}" for k, v in m["by_n_components"].items()), ""]
    L += ["## Nodes", "", "| label | count |", "|---|---:|"]
    L += [f"| {k} | {v:,} |" for k, v in h["nodes"].items()]
    L += ["", "## Relationships", "", "| type | from → to | count |", "|---|---|---:|"]
    L += [f"| {r['t']} | {r['a']} → {r['b']} | {r['n']:,} |"
          for r in sorted(h["relationships"], key=lambda r: (r["t"], -r["n"]))]
    by_type = Counter()
    for r in h["relationships"]:
        by_type[r["t"]] += r["n"]
    L += ["", "Totals by type: " + ", ".join(f"{t} {n:,}" for t, n in by_type.most_common()), ""]
    L += ["## DES properties", "",
          "| property | values | with uncertainty | DES | papers | unit | min | max | implausible |",
          "|---|---:|---:|---:|---:|---|---:|---:|---:|"]
    L += [f"| {r['property']} | {r['values']:,} | {r['with_uncertainty']:,} | {r['mixtures']:,} | "
          f"{r['papers']:,} | {', '.join(u for u in r['units'] if u) or '—'} | {_short(r['lo'])} | "
          f"{_short(r['hi'])} | {r['implausible']} |" for r in h["properties"]]
    L += ["", "## Component properties (PubChem / NIST)", "",
          "| property | values | components | sources | unit | min | max |",
          "|---|---:|---:|---:|---|---:|---:|"]
    L += [f"| {r['property']} | {r['values']:,} | {r['components']:,} | {r['sources']:,} | "
          f"{', '.join(u for u in r['units'] if u) or '—'} | {_short(r['lo'])} | {_short(r['hi'])} |"
          for r in h["component_properties"]]
    L += ["", "## Sources", "", "| source | kind | values |", "|---|---|---:|"]
    L += [f"| {s['source']} | {s['kind']} | {s['values']:,} |" for s in h["sources"]]
    L += ["", "## Applications", "", "| domain | records | DES | papers |", "|---|---:|---:|---:|"]
    L += [f"| {a['domain']} | {a['records']:,} | {a['mixtures']:,} | {a['papers']:,} |"
          for a in h["applications"]]
    return "\n".join(L) + "\n"


def schema_mmd(boxes, edges):
    L = ["erDiagram"]
    for e in edges:
        L.append(f'    {e["from"]} }}o--o{{ {e["to"]} : "{e["type"]} ({e["count"]:,})"')
    for b in boxes.values():
        L.append(f"    {b['id']} {{")
        for f in b["fields"]:
            L.append(f'        {f["type"].replace("?", "unknown")} {f["name"]} "{f["pct"]}% filled"')
        L.append("    }")
    return "\n".join(L) + "\n"


def _embed(obj):
    return json.dumps(obj, default=str).replace("</", "<\\/")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path,
                    default=(config.RUN_DIR or config.ROOT) / "graph_report")
    ap.add_argument("--mixture", help="key or exact name of the DES to draw as the example")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    g = Graph()
    try:
        nodes, rels, measure, cprops = inventory(g)
        h = headline(g, nodes, rels, measure, cprops)
        boxes, edges = schema(g, nodes, rels)
        key = pick_example(g, args.mixture)
        ex = example(g, key) if key else None
        net = component_network(g)
        fig = figure_counts(g, nodes, rels)
    finally:
        g.close()

    placed = {b for col in COLUMNS for b in col}
    columns = [[b for b in col if b in boxes] for col in COLUMNS]
    columns[-1] += [b for b in boxes if b not in placed]
    data = {"headline": h, "boxes": boxes, "edges": edges, "columns": columns, "example": ex}

    (args.out / "summary.json").write_text(json.dumps(h, indent=1, default=str))
    (args.out / "summary.md").write_text(summary_md(h))
    (args.out / "figure_counts.json").write_text(json.dumps(fig, indent=1))
    (args.out / "schema.mmd").write_text(schema_mmd(boxes, edges))
    (args.out / "schema.html").write_text(SCHEMA_HTML.replace("/*DATA*/", _embed(data)))
    (args.out / "component_network.html").write_text(
        NETWORK_HTML.replace("/*DATA*/", _embed(net)).replace("/*TAG*/", h["run_tag"]))
    write_gexf(net, args.out / "component_network.gexf")
    print(summary_md(h))
    print(f"example DES: {ex and ex['key']}")
    print(f"wrote {args.out}/: summary.md summary.json figure_counts.json schema.html "
          "schema.mmd component_network.html component_network.gexf")


SCHEMA_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DES Graph Schema</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1f;--mute:#6b6b70;--line:#d9d9de;--edge:#8a8a93;
--Component:#2f7d6d;--Mixture:#b0472f;--Measurement:#2c5d9e;--ApplicationRecord:#8a5a12;
--Application:#a3741c;--Paper:#5b4b8a;--Source:#2b8a80;--ComponentProperty:#b07d1a;--other:#55606e}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--card:#1e1e22;--ink:#ececf0;--mute:#9d9da6;
--line:#34343a;--edge:#7b7b86;--Component:#5fb8a4;--Mixture:#e0795e;--Measurement:#6f9ee0;
--ApplicationRecord:#d0a050;--Application:#e0b864;--Paper:#a593e0;--Source:#6cc9bd;--ComponentProperty:#e2b252;--other:#a0aab8}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,system-ui,"Segoe UI",sans-serif}
main{max-width:1500px;margin:0 auto;padding:24px 16px 60px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:34px 0 6px}
.sub{color:var(--mute);margin:0 0 18px}
.stats{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.stat b{display:block;font-size:22px;font-variant-numeric:tabular-nums}
.stat span{color:var(--mute);font-size:12px}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--card)}
.diagram{position:relative;display:flex;gap:150px;padding:28px;width:max-content;align-items:center}
.col{display:flex;flex-direction:column;gap:60px;width:330px;justify-content:center}
.box{position:relative;z-index:1;background:var(--card);border:1.5px solid var(--c);border-radius:8px;
font-size:12px;overflow:hidden}
.box header{background:var(--c);color:#fff;padding:6px 9px;font-weight:600;display:flex;justify-content:space-between;gap:8px}
.box header small{font-weight:400;opacity:.9}
.box .role{color:var(--mute);padding:5px 9px 2px;font-style:italic}
.box table{width:100%;border-collapse:collapse}
.box td{padding:2px 9px;vertical-align:top;border-top:1px solid var(--line)}
.box td.k{font-family:ui-monospace,Menlo,monospace;white-space:nowrap}
.box td.p{color:var(--mute);text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}
.box td.v{color:var(--mute);word-break:break-word}
.box .labels{padding:4px 9px;color:var(--mute);line-height:1.5}
svg.edges{position:absolute;inset:0;width:100%;height:100%;z-index:2;pointer-events:none;overflow:visible}
svg.edges path{fill:none;stroke:var(--edge);stroke-width:1.6}
svg.edges text{font:11px ui-monospace,Menlo,monospace;fill:var(--ink)}
svg.edges rect{fill:var(--card);stroke:var(--line)}
.note{color:var(--mute);font-size:12px;margin:8px 2px}
pre{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;overflow-x:auto;font-size:12px}
table.list{border-collapse:collapse;background:var(--card);font-size:13px}
table.list td,table.list th{border:1px solid var(--line);padding:4px 10px;text-align:left}
table.list td.n{text-align:right;font-variant-numeric:tabular-nums}
.tables{display:flex;flex-wrap:wrap;gap:24px;align-items:flex-start}
</style></head><body><main>
<h1>DES knowledge graph</h1>
<p class="sub" id="sub"></p>
<div class="stats" id="stats"></div>

<h2>Schema</h2>
<p class="note">One box per node type, with every property stored on it: share of nodes that carry it, and an example (or its values, for categorical fields). Property labels (Density, Viscosity…) are used by two kinds of node, kept apart here: <i>Measurement</i> is a value measured on a DES, <i>ComponentProperty</i> a value for a pure component from PubChem or NIST. Each is drawn as one box listing its labels. Arrows give the relationship type, its count, and its own properties.</p>
<div class="wrap"><div class="diagram" id="schema"></div></div>

<h2>One real DES</h2>
<p class="note" id="exnote"></p>
<div class="wrap"><div class="diagram" id="example"></div></div>
<p class="note">Same subgraph in Neo4j Browser or Bloom (Cypher):</p>
<pre id="cypher"></pre>

<h2>Counts</h2>
<div class="tables" id="tables"></div>
</main>
<script>
const D = /*DATA*/;
const H = D.headline, fmt = n => (n ?? 0).toLocaleString('en-US');
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const color = g => `var(--${['Component','Mixture','Measurement','ComponentProperty','ApplicationRecord','Application','Paper','Source'].includes(g)?g:'other'})`;

document.getElementById('sub').textContent =
  `${H.run_tag} · ${H.date} · ${fmt(H.nodes_total)} nodes, ${fmt(H.relationships_total)} relationships`;
const apps = H.applications.reduce((s,a)=>s+a.records,0);
const stats = [
  [H.components.total,'unique components'],[H.mixtures.total,'DES (mixture nodes)'],
  [H.mixtures.systems_ignoring_water,'distinct DES systems (component sets, water ignored)'],
  [H.properties.length,'DES property types'],[H.measurements_total,'DES property values'],
  [H.component_properties_total,'component property values (PubChem/NIST)'],[H.sources.length,'sources'],
  [H.applications.length,'application domains'],[apps,'application records'],
  [H.papers.extracted,'papers extracted'],[H.papers.cited_only,'cited papers (data sources)']];
document.getElementById('stats').innerHTML = stats.map(([n,t])=>`<div class="stat"><b>${fmt(n)}</b><span>${t}</span></div>`).join('');

function fieldRows(fields){
  return fields.map(f=>{
    const v = f.values ? f.values.map(([x,c])=>`${esc(x)} ${fmt(c)}`).join(' · ') : esc(f.example);
    return `<tr><td class="k">${esc(f.name)}</td><td class="p">${f.pct}%</td><td class="v">${v}</td></tr>`}).join('');
}
function drawDiagram(el, columns, edgeList, renderBox){
  el.innerHTML = '';
  for (const col of columns){
    const c = document.createElement('div'); c.className='col';
    for (const item of col){ c.appendChild(renderBox(item)); }
    el.appendChild(c);
  }
  const svg = document.createElementNS('http://www.w3.org/2000/svg','svg');
  svg.setAttribute('class','edges'); el.appendChild(svg);
  const draw = () => {
    const base = el.getBoundingClientRect();
    svg.innerHTML = `<defs><marker id="ah-${el.id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" fill="var(--edge)" stroke="none"/></marker></defs>`;
    const seen = {};
    for (const e of edgeList){
      const a = el.querySelector(`[data-id="${CSS.escape(e.from)}"]`), b = el.querySelector(`[data-id="${CSS.escape(e.to)}"]`);
      if (!a || !b) continue;
      const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect();
      const pair = e.from+'>'+e.to; seen[pair] = (seen[pair]||0)+1;
      const off = (seen[pair]-1)*16;
      // leave from the side facing the target, enter on the side facing the source
      let x1, y1, x2, y2;
      if (rb.left > ra.right) { x1 = ra.right; x2 = rb.left; }
      else if (ra.left > rb.right) { x1 = ra.left; x2 = rb.right; }
      else { x1 = ra.left + ra.width/2; x2 = rb.left + rb.width/2; }
      y1 = Math.min(Math.max(rb.top + rb.height/2, ra.top+14), ra.bottom-14) + off;
      y2 = Math.min(Math.max(ra.top + ra.height/2, rb.top+14), rb.bottom-14) + off;
      if (x1 === x2) { if (rb.top >= ra.bottom) { y1 = ra.bottom; y2 = rb.top; } else { y1 = ra.top; y2 = rb.bottom; } }
      x1-=base.left; x2-=base.left; y1-=base.top; y2-=base.top;
      const mx=(x1+x2)/2, my=(y1+y2)/2;
      const p = document.createElementNS(svg.namespaceURI,'path');
      p.setAttribute('d', `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`);
      p.setAttribute('marker-end', `url(#ah-${el.id})`); svg.appendChild(p);
      const lines = e.label;
      const g = document.createElementNS(svg.namespaceURI,'g'); svg.appendChild(g);
      const w = Math.max(...lines.map(l=>l.length))*6.7+10, h = lines.length*13+6;
      g.innerHTML = `<rect x="${mx-w/2}" y="${my-h/2}" width="${w}" height="${h}" rx="4"/>` +
        lines.map((l,i)=>`<text x="${mx}" y="${my-h/2+14+i*13}" text-anchor="middle">${esc(l)}</text>`).join('');
    }
  };
  requestAnimationFrame(draw);
  new ResizeObserver(draw).observe(el);
}

// ---- schema
drawDiagram(document.getElementById('schema'), D.columns,
  D.edges.map(e=>({...e, label:[`${e.type} (${fmt(e.count)})`, ...e.fields.map(f=>`{${f.name}${f.values?': '+f.values.map(v=>v[0]).join('|'):''}}`)] })),
  id => {
    const b = D.boxes[id], d = document.createElement('div');
    d.className='box'; d.dataset.id=id; d.style.setProperty('--c', color(id));
    d.innerHTML = `<header><span>:${esc(id)}</span><small>${fmt(b.count)}</small></header>` +
      (b.role?`<div class="role">${esc(b.role)}</div>`:'') +
      (b.labels.length?`<div class="labels">labels: ${b.labels.map(([l,n])=>`${esc(l)} ${fmt(n)}`).join(' · ')}</div>`:'') +
      `<table>${fieldRows(b.fields)}</table>`;
    return d;
  });

// ---- example
const X = D.example;
if (X){
  const order = ['Component','Mixture','Measurement','ApplicationRecord','Paper','Application'];
  const cols = [[],[],[],[]];
  const colOf = {};
  D.columns.forEach((col,i)=>col.forEach(g=>colOf[g]=i));
  cols.length = 0; D.columns.forEach(()=>cols.push([]));
  for (const n of X.nodes) cols[colOf[n.group] ?? cols.length-1].push(n.id);
  const byId = Object.fromEntries(X.nodes.map(n=>[n.id,n]));
  const mixName = X.nodes.find(n=>n.group==='Mixture')?.title;
  document.getElementById('exnote').innerHTML = `<b>${esc(mixName)}</b> with all its components (each with one database property value and its source), one value of up to three DES properties, one application record, and the papers they are reported in. Every property stored on each node and relationship is shown with its real value.`;
  document.getElementById('cypher').textContent = X.cypher;
  drawDiagram(document.getElementById('example'), cols,
    X.edges.map(e=>({...e, label:[e.type, ...e.props.map(([k,v])=>`${k}: ${v}`)]})),
    id => {
      const n = byId[id], d = document.createElement('div');
      d.className='box'; d.dataset.id=id; d.style.setProperty('--c', color(n.group));
      d.innerHTML = `<header><span>:${esc(n.label)}</span></header><div class="role" style="font-style:normal;color:var(--ink);font-weight:600">${esc(n.title)}</div>` +
        `<table>${n.props.map(([k,v])=>`<tr><td class="k">${esc(k)}</td><td class="v">${esc(v)}</td></tr>`).join('')}</table>`;
      return d;
    });
}

// ---- count tables
const tbl = (head, rows) => `<table class="list"><tr>${head.map(h=>`<th>${h}</th>`).join('')}</tr>${rows.map(r=>`<tr>${r.map(c=>typeof c==='number'?`<td class="n">${fmt(c)}</td>`:`<td>${esc(c)}</td>`).join('')}</tr>`).join('')}</table>`;
document.getElementById('tables').innerHTML =
  tbl(['node label','count'], Object.entries(H.nodes)) +
  tbl(['relationship','from → to','count'], H.relationships.map(r=>[r.t, `${r.a} → ${r.b}`, r.n])) +
  tbl(['DES property','values','± given','DES','papers'], H.properties.map(p=>[p.property,p.values,p.with_uncertainty,p.mixtures,p.papers])) +
  tbl(['component property','values','components','sources'], H.component_properties.map(p=>[p.property,p.values,p.components,p.sources])) +
  tbl(['source','kind','values'], H.sources.map(s=>[s.source,s.kind||'',s.values])) +
  tbl(['application domain','records','DES','papers'], H.applications.map(a=>[a.domain,a.records,a.mixtures,a.papers]));
</script></body></html>
"""

NETWORK_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DES Component Network</title>
<script src="https://cdn.jsdelivr.net/npm/force-graph@1.43.5/dist/force-graph.min.js"></script>
<style>
:root{--bg:#f7f7f5;--panel:#fff;--ink:#1d1d1f;--mute:#6b6b70;--line:#d9d9de}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--panel:#1e1e22;--ink:#ececf0;--mute:#9d9da6;--line:#34343a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:13px/1.4 -apple-system,system-ui,sans-serif;overflow:hidden}
#g{position:fixed;inset:0}
#panel{position:fixed;top:12px;left:12px;width:min(320px,calc(100vw - 24px));max-height:calc(100vh - 24px);overflow:auto;
background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px;z-index:2}
h1{font-size:15px;margin:0 0 4px}.mute{color:var(--mute)}
label{display:block;margin:8px 0 2px}input[type=text]{width:100%;padding:5px 7px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink)}
input[type=range]{width:100%}
.key span{display:inline-block;width:10px;height:10px;border-radius:50%;margin:0 4px 0 8px;vertical-align:-1px}
#info{margin-top:10px;border-top:1px solid var(--line);padding-top:8px}
#info li{margin:1px 0}
</style></head><body>
<div id="g"></div>
<div id="panel">
  <h1>Components linked by shared DES</h1>
  <div class="mute">/*TAG*/ · node size = number of DES using it; link width = DES they share. Click a node for its partners; scroll to zoom.</div>
  <div class="key mute" style="margin-top:6px"><span style="background:#3b7dd8"></span>mostly HBA<span style="background:#e07b39"></span>mostly HBD<span style="background:#8e8e96"></span>not in PubChem</div>
  <label>Find component <input type="text" id="q" list="names" placeholder="e.g. Choline chloride"></label>
  <datalist id="names"></datalist>
  <label>Hide links shared by fewer than <b id="wv">2</b> DES <input type="range" id="w" min="1" max="10" value="2"></label>
  <label>Hide components in fewer than <b id="nv">2</b> DES <input type="range" id="n" min="1" max="20" value="2"></label>
  <label><input type="checkbox" id="water" checked> hide water (it joins almost everything)</label>
  <div id="counts" class="mute"></div>
  <div id="info"></div>
</div>
<script>
const NET = /*DATA*/;
const byId = Object.fromEntries(NET.nodes.map(n=>[n.id,n]));
const partners = {};
for (const l of NET.links){ (partners[l.source] ||= []).push([l.target,l.w]); (partners[l.target] ||= []).push([l.source,l.w]); }
document.getElementById('names').innerHTML = NET.nodes.map(n=>`<option value="${n.id.replace(/"/g,'&quot;')}">`).join('');
const col = n => n.cid == null ? '#8e8e96' : (n.hba >= n.hbd ? '#3b7dd8' : '#e07b39');
let focus = null;
const G = ForceGraph()(document.getElementById('g'))
  .nodeId('id').nodeVal(n=>Math.max(1,n.mixtures)).nodeRelSize(1.6)
  .nodeLabel(n=>`${n.id} — ${n.mixtures} DES, ${n.values} property values, ${n.apps} application records`)
  .nodeColor(n=> !focus || n.id===focus || (partners[focus]||[]).some(p=>p[0]===n.id) ? col(n) : 'rgba(140,140,150,.15)')
  .linkWidth(l=>Math.min(8, Math.sqrt(l.w)))
  .linkColor(l=> focus && (l.source.id===focus || l.target.id===focus) ? 'rgba(220,60,60,.8)' : 'rgba(140,140,150,.25)')
  .nodeCanvasObjectMode(()=> 'after')
  .nodeCanvasObject((n, ctx, scale) => {
    if (n.mixtures * scale < 40 && n.id !== focus) return;
    ctx.font = `${11/scale}px sans-serif`; ctx.fillStyle = getComputedStyle(document.body).color;
    ctx.textAlign='center'; ctx.fillText(n.id, n.x, n.y + Math.sqrt(Math.max(1,n.mixtures))*1.6 + 10/scale);
  })
  .onNodeClick(n => select(n.id))
  .onBackgroundClick(() => select(null))
  .cooldownTicks(300)
  .onEngineStop(() => { if (!focus) G.zoomToFit(400, 40); });
G.d3Force('charge').strength(-40);
function select(id){
  focus = id; const info = document.getElementById('info');
  if (!id){ info.innerHTML=''; G.nodeColor(G.nodeColor()); return; }
  const n = byId[id], ps = (partners[id]||[]).sort((a,b)=>b[1]-a[1]);
  info.innerHTML = `<b>${id}</b>${n.formula?` (${n.formula})`:''}<div class="mute">${n.mixtures} DES · ${n.values} DES property values · ${n.apps} application records · HBA ${n.hba} / HBD ${n.hbd}${n.cid?` · CID ${n.cid}`:' · not in PubChem'}</div>` +
    `<div class="mute">${[n.mw!=null?`MW ${(+n.mw).toFixed(1)}`:'', n.mp!=null?`mp ${n.mp} °C`:'', n.bp!=null?`bp ${n.bp} °C`:'', `${n.db_values} PubChem/NIST values`, n.qm9?'QM9 data':''].filter(Boolean).join(' · ')}</div>` +
    `<div style="margin-top:6px">Paired with (shared DES):</div><ol>${ps.slice(0,25).map(([p,w])=>`<li>${p} <span class="mute">${w}</span></li>`).join('')}</ol>${ps.length>25?`<div class="mute">…and ${ps.length-25} more</div>`:''}`;
  G.nodeColor(G.nodeColor());
  const node = G.graphData().nodes.find(x=>x.id===id);
  if (node && node.x != null){ G.centerAt(node.x, node.y, 600); G.zoom(3, 600); }
}
function update(){
  const wmin = +document.getElementById('w').value, nmin = +document.getElementById('n').value;
  const noWater = document.getElementById('water').checked;
  document.getElementById('wv').textContent = wmin; document.getElementById('nv').textContent = nmin;
  const keep = new Set(NET.nodes.filter(n=>n.mixtures>=nmin && !(noWater && n.id.toLowerCase()==='water')).map(n=>n.id));
  const links = NET.links.filter(l=>l.w>=wmin && keep.has(l.source) && keep.has(l.target)).map(l=>({...l}));
  const linked = new Set(links.flatMap(l=>[l.source,l.target]));
  const nodes = NET.nodes.filter(n=>keep.has(n.id) && linked.has(n.id));
  document.getElementById('counts').textContent = `showing ${nodes.length} of ${NET.nodes.length} components, ${links.length} of ${NET.links.length} links`;
  G.graphData({nodes, links});
}
for (const id of ['w','n','water']) document.getElementById(id).addEventListener('input', update);
document.getElementById('q').addEventListener('change', e => { if (byId[e.target.value]) select(e.target.value); });
update();
setTimeout(() => { if (!focus) G.zoomToFit(400, 40); }, 2500);
</script></body></html>
"""

if __name__ == "__main__":
    main()
