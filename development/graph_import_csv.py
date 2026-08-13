import pandas as pd
from neo4j import GraphDatabase

"""

Import data from csv

"""


### Connect to the graph database
driver = GraphDatabase.driver("neo4j://127.0.0.1:7687", auth=("neo4j", "@n11XZ10duEJohmc"))
driver.verify_connectivity()

### Load data into Neo4j
# Data preprocessing in pandas 
df = pd.read_csv("toy_data.csv")

identity   = ["Component_1","Component_1_SMILES","Component_2","Component_2_SMILES",
              "Mixture","Ratio_component_1","Ratio_component_2","DOI"]
conditions = ["Liquid_solid_ratio_g_g","Temperature_C","Time_h"]
results    = ["Leaching_efficiency_percent"]

PROP_META = {
    "Leaching_efficiency_percent": ("Leaching efficiency", "%"),
    # add a line here to add another properry/column, e.g.:
    # "Density_g_cm3": ("Density", "g/cm3"),
}

long = df.melt(id_vars=identity+conditions, value_vars=results,
               var_name="prop_col", value_name="value")
long["property"] = long["prop_col"].map(lambda c: PROP_META[c][0])
long["unit"]     = long["prop_col"].map(lambda c: PROP_META[c][1])
rows = long.to_dict("records")
print(long)

long.to_csv('long.csv', index=False)
                            
# Constraints
for c in [
    "CREATE CONSTRAINT component_name IF NOT EXISTS FOR (c:Component) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT mixture_name  IF NOT EXISTS FOR (m:Mixture)   REQUIRE m.name IS UNIQUE",
    "CREATE CONSTRAINT property_name IF NOT EXISTS FOR (p:Property)  REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT paper_doi     IF NOT EXISTS FOR (p:Paper)     REQUIRE p.doi  IS UNIQUE",
]:
    driver.execute_query(c, database_="neo4j")

# Load 
driver.execute_query("""
    UNWIND $rows AS row
    MERGE (ca:Component {name: row.Component_1}) SET ca.smiles = row.Component_1_SMILES
    MERGE (cb:Component {name: row.Component_2}) SET cb.smiles = row.Component_2_SMILES
    MERGE (mix:Mixture {name: row.Mixture})
    MERGE (ca)-[ra:PART_OF]->(mix) SET ra.molar_ratio = row.Ratio_component_1
    MERGE (cb)-[rb:PART_OF]->(mix) SET rb.molar_ratio = row.Ratio_component_2
    MERGE (prop:Property {name: row.property})
    MERGE (paper:Paper {doi: row.DOI})
    MERGE (meas:Measurement {
        mixture: row.Mixture,
        property: row.property,
        temperature_C: row.Temperature_C,
        time_h: row.Time_h,
        liquid_solid_ratio: row.Liquid_solid_ratio_g_g,
        doi: row.DOI
    })
      SET meas.value = row.value, meas.unit = row.unit
    MERGE (mix)-[:HAS_MEASUREMENT]->(meas)
    MERGE (meas)-[:OF_PROPERTY]->(prop)
    MERGE (meas)-[:REPORTED_IN]->(paper)
""", rows=rows, database_="neo4j")