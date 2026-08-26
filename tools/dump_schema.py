#!/usr/bin/env python3
"""
Print the graph's schema as the Markdown tables in README's "Schema reference".

    python tools/dump_schema.py > /tmp/schema.md

Read out of the running database rather than written by hand, because a schema
reference that is maintained by hand is wrong within a week. Regenerate it whenever a
property or node type is added, and paste over the two tables in the README.

Field *meanings* stay hand-written above those tables -- only the inventory is generated.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from des_pipeline import build_graph, config     # noqa: E402


def main():
    driver = build_graph._driver()
    try:
        run = lambda cypher: driver.execute_query(          # noqa: E731
            cypher, database_=config.NEO4J_DATABASE).records

        counts = {r["l"]: r["n"] for r in
                  run("MATCH (n) UNWIND labels(n) AS l RETURN l, count(*) AS n")}
        if not counts:
            sys.exit("the database is empty -- run `--steps graph` first")

        node_props = {tuple(r["nodeLabels"]): sorted(r["props"]) for r in
                      run("CALL db.schema.nodeTypeProperties() "
                          "YIELD nodeLabels, propertyName "
                          "RETURN nodeLabels, collect(propertyName) AS props")}

        print("| label | n | properties |")
        print("|---|---|---|")
        for label, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            fields = ", ".join(f"`{p}`" for p in node_props.get((label,), []))
            print(f"| `:{label}` | {n} | {fields} |")

        rel_props = {r["relType"].strip(":`"): sorted(p for p in r["props"] if p)
                     for r in run("CALL db.schema.relTypeProperties() "
                                  "YIELD relType, propertyName "
                                  "RETURN relType, collect(propertyName) AS props")}

        endpoints = {}
        for r in run("MATCH (a)-[r]->(b) RETURN type(r) AS t, labels(a)[0] AS a, "
                     "labels(b)[0] AS b, count(*) AS n ORDER BY t, n DESC"):
            endpoints.setdefault(r["t"], []).append((r["a"], r["b"], r["n"]))

        print("\n| relationship | n | endpoints | properties |")
        print("|---|---|---|---|")
        for rel, pairs in sorted(endpoints.items(),
                                 key=lambda kv: -sum(p[2] for p in kv[1])):
            total = sum(p[2] for p in pairs)
            shown = "<br>".join(f"`(:{a})`→`(:{b})` {n}" for a, b, n in pairs[:4])
            if len(pairs) > 4:
                shown += f"<br>…{len(pairs) - 4} more"
            fields = ", ".join(f"`{p}`" for p in rel_props.get(rel, [])) or "—"
            print(f"| `{rel}` | {total} | {shown} | {fields} |")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
