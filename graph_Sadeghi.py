import pandas as pd, hashlib, json
from neo4j import GraphDatabase


driver = GraphDatabase.driver("neo4j://127.0.0.1:7687", auth=("neo4j", "@n11XZ10duEJohmc"))
driver.verify_connectivity()
driver.execute_query("MATCH (n) DETACH DELETE n")   # Delete database and reset 

df = pd.read_csv("table2_full.csv")
df = df.replace({r"¬∑": "·"}, regex=True) 
df = df.astype(object).where(pd.notna(df), None)   # NaN -> None, so IS NOT NULL works


rows = df.to_dict("records")

# Use coalesce to prevent future blank SMILES from removing previously stored SMILES
# Only create property nodes when there is a measured value

driver.execute_query("""
    UNWIND $rows AS row
    MERGE (ca:Component {name: row.Component_1})
      SET ca.smiles = coalesce(row.Component_1_SMILES, ca.smiles) 
    MERGE (cb:Component {name: row.Component_2})
      SET cb.smiles = coalesce(row.Component_2_SMILES, cb.smiles)
    MERGE (mix:Mixture {name: row.Mixture})
    MERGE (ca)-[ra:PART_OF]->(mix) SET ra.molar_ratio = row.Ratio_component_1
    MERGE (cb)-[rb:PART_OF]->(mix) SET rb.molar_ratio = row.Ratio_component_2
    MERGE (paper:Paper {doi: row.DOI}) SET paper.reference = row.Ref

    
    FOREACH (_ IN CASE WHEN row.Density IS NOT NULL AND row.Density = row.Density THEN [1] ELSE [] END |
        MERGE (d:Density {mixture: row.Mixture})
          SET d.value = row.Density, d.unit = row.Units_density
        MERGE (mix)-[rd:HAS_DENSITY]->(d)
          SET rd.temperature_C = row.Temperature_density
        MERGE (d)-[rep:REPORTED_IN]->(paper)
          SET rep.reference = row.Ref
    )
    FOREACH (_ IN CASE WHEN row.Viscosity IS NOT NULL AND row.Viscosity = row.Viscosity THEN [1] ELSE [] END |
        MERGE (v:Viscosity {mixture: row.Mixture})
          SET v.value = row.Viscosity, v.unit = row.Units_viscosity
        MERGE (mix)-[rv:HAS_VISCOSITY]->(v)
          SET rv.temperature_C = row.Temperature_viscosity
        MERGE (v)-[rep:REPORTED_IN]->(paper)
          SET rep.reference = row.Ref
    )
    FOREACH (_ IN CASE WHEN row.Conductivity IS NOT NULL AND row.Conductivity = row.Conductivity THEN [1] ELSE [] END |
        MERGE (c:Conductivity {mixture: row.Mixture})
          SET c.value = row.Conductivity, c.unit = row.Units_conductivity
        MERGE (mix)-[rc:HAS_CONDUCTIVITY]->(c)
          SET rc.temperature_C = row.Temperature_conductivity
        MERGE (c)-[rep:REPORTED_IN]->(paper)
          SET rep.reference = row.Ref
    )
    FOREACH (_ IN CASE WHEN row.Surface_tension IS NOT NULL AND row.Surface_tension = row.Surface_tension THEN [1] ELSE [] END |
        MERGE (st:Surface_tension {mixture: row.Mixture})
          SET st.value = row.Surface_tension, st.unit = row.Units_surface_tension
        MERGE (mix)-[rst:HAS_SURFACE_TENSION]->(st)
          SET rst.temperature_C = row.Temperature_surface_tension
        MERGE (st)-[rep:REPORTED_IN]->(paper)
          SET rep.reference = row.Ref
    )
    FOREACH (_ IN CASE WHEN row.Refractive_index IS NOT NULL AND row.Refractive_index = row.Refractive_index THEN [1] ELSE [] END |
        MERGE (ri:Refractive_index {mixture: row.Mixture})
          SET ri.value = row.Refractive_index, ri.unit = row.Units_refractive_index
        MERGE (mix)-[rri:HAS_REFRACTIVE_INDEX]->(ri)
          SET rri.temperature_C = row.Temperature_refractive_index
        MERGE (ri)-[rep:REPORTED_IN]->(paper)
          SET rep.reference = row.Ref
    )
""", rows=rows, database_="neo4j")