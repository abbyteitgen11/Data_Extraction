from bs4 import BeautifulSoup
from chemicals import CAS_from_any
from chemicals.phase_change import Tm as melting_temperature
from chemicals.miscdata import lookup_VDI_tabular_data
from chemicals.identifiers import search_chemical
from lxml import etree
from neo4j import GraphDatabase
from neo4j import Result
import pubchempy as pcp
import re
import requests

from component import Component
from paper import Paper
from process_property import obtain_NIST_property
#import pdb; pdb.set_trace()

"""

Toy graph model to learn Neo4j and Cypher

"""

### Connect to the graph database
driver = GraphDatabase.driver("neo4j://127.0.0.1:7687", auth=("neo4j", "!@xZ8LiuK29oB@iC"))

driver.verify_connectivity()

"""
paper = {"doi": "10.1039/d4gc01418a", "title": "Machine learning models accelerate deep eutectic solvent discovery for the recycling of lithium-ion ..."
"battery cathodes", "author": "Zhou et al.", "year": 2024, "journal": "Green Chemistry", "volume": 26, "pages": "7857-7868"}

"""

### Create database 
# Adding components, mixtures and relationships one at a time
#cypher = """
#MERGE (c:Component {name: 'Choline Chloride'})
#"""
#driver.execute_query(cypher)

#records, summary, keys = driver.execute_query(
#    "MERGE (c:Component {name: 'Choline Chloride'}) RETURN c.name AS name"
#)
#print(summary.counters.nodes_created, records[0]["name"])

#cypher = """
#MATCH (c1:Component {name: 'Choline Chloride'})
#MATCH (m:Mixture {name: 'Mixture 1'})
#MERGE (c1)-[:PART_OF]->(m)
#"""
#driver.execute_query(cypher)



# Delete entire database 
#cypher = """
#MATCH (n)
#DETACH DELETE n
#"""
#driver.execute_query(cypher)




# Adding components, mixtures, and relationships in a loop
components = [
    "Choline Chloride", "Glycerol", "Ethylene Glycol", "Urea", "Oxalic acid",
    "P toluenesulfonic acid", "L Ascorbic acid", "Malic acid",
    "Formic acid", "Acetic acid",
]

records, summary, _ = driver.execute_query(
    """
    UNWIND $names AS name
    MERGE (c:Component {name: name})
    """,
    names=components,
)
print(summary)


mixtures = [
    {"name": "Mixture 1", "a": "Choline Chloride", "ra": 1, "b": "Glycerol ", "rb": 1},
    {"name": "Mixture 2", "a": "Choline Chloride", "ra": 2, "b": "Ethylene Glycol", "rb": 1},
    {"name": "Mixture 3", "a": "Choline Chloride", "ra": 1, "b": "Urea", "rb": 2},
    {"name": "Mixture 4", "a": "Choline Chloride", "ra": 1, "b": "Oxalic acid", "rb": 1},
    {"name": "Mixture 5", "a": "Choline Chloride", "ra": 1, "b": "P toluenesulfonic acid", "rb": 1},
    {"name": "Mixture 6", "a": "Choline Chloride", "ra": 2, "b": "L Ascorbic acid", "rb": 1},
    {"name": "Mixture 7", "a": "Choline Chloride", "ra": 1, "b": "Malic acid", "rb": 1},
    {"name": "Mixture 8", "a": "Choline Chloride", "ra": 1, "b": "Formic acid", "rb": 2},
    {"name": "Mixture 9", "a": "Choline Chloride", "ra": 1, "b": "Acetic acid", "rb": 2},
]

driver.execute_query(
    """
    UNWIND $mixtures AS m
    MERGE (mix:Mixture {name: m.name})
    WITH mix, m
    MATCH (ca:Component {name: m.a})
    MATCH (cb:Component {name: m.b})
    MERGE (ca)-[ra:PART_OF]->(mix) SET ra.molar_ratio = m.ra
    MERGE (cb)-[rb:PART_OF]->(mix) SET rb.molar_ratio = m.rb
    """,
    mixtures=mixtures,
)




### Searching database 

# Getting info on a specific component
#cypher = """
#MATCH (c:Component {name: $name})-[r:PART_OF]->(m:Mixture)
#RETURN r.molar_ratio AS molar_ratio, m.name AS mixture_name, c.name AS component_name
#"""
#name = "Urea"
 
#records, summary, keys = driver.execute_query( 
#    cypher,    
#    name=name  
#)
#print(summary)
#print(keys)
#for record in records:
#    print(record["component_name"], record["molar_ratio"], record["mixture_name"])


# Print all component nodes in the database
#cypher = """
#MATCH (c:Component)
#RETURN c.name AS component_name
#"""
 
#records, summary, keys = driver.execute_query( 
#    cypher    
#)
#print(records)


# Print property of specific component
#cypher = """
#MATCH (c:Component {name: $name})
#RETURN c.name AS component_name
#"""
#name = "Oxalic acid"

#records, summary, keys = driver.execute_query( 
#    cypher,    
#    name=name    
#)
#print(records)


# Print all mixtures choline chloride is a part of where it has a molar ratio of 1
#cypher = """
#MATCH (c:Component {name: 'Choline Chloride'})-[r:PART_OF]->(m)
#WHERE r.molar_ratio = 1 
#RETURN m.name AS mixture_name
#"""
## Can also have >= 1 etc 

#records, _, _ = driver.execute_query(
#    cypher,       
#)
#print(records)


# Return all components that end with 'acid'
#cypher = """
#MATCH (c:Component)
#WHERE c.name ENDS WITH 'acid'
#RETURN c.name
#"""

#records, _, _ = driver.execute_query(
#    cypher,       
#)
#print(records)
## Case sensitive


# Return all components that do not have a PART_OF relationship
#cypher = """
#MATCH (c:Component)
#WHERE NOT exists ((c)-[:PART_OF]->())
#RETURN c.name
#"""

#records, summary, keys = driver.execute_query(
#    cypher,       
#)
#print(records)
#print(summary)
#print(keys)


# Result transformer
#cypher = """
#MATCH (c:Component)
#WHERE NOT exists ((c)-[:PART_OF]->())
#RETURN c.name as component_name 
#"""

#result = driver.execute_query(
#    cypher,
#    result_transformer_= lambda result: [f"These components: {record['component_name']} do not belong to any mixtures"
 #                                        for record in result
#                                         ]       
#)
#print(result)


# Transform to dataframe 
#cypher = """
#MATCH (c:Component)
#WHERE c.name ENDS WITH 'acid'
#RETURN c.name
#"""

#result = driver.execute_query(
#    cypher,
#    result_transformer_=Result.to_df     
#)
#print(result[0])


# Node properties
mix = "Mixture 3"

records, summary, keys = driver.execute_query(""" 
    MATCH path = (c:Component)-[r:PART_OF]->(m:Mixture {name: $mixture}) 
    RETURN path, c, r, m
    """, mixture=mix
)

for record in records:
    node  = record["m"]
    print(node.element_id)
    print(node.labels)
    print(node.items())
    print(node["name"])

    part_of = record["r"]
    print(part_of.element_id)
    print(part_of.type)
    print(part_of.items())
    print(part_of["molar_ratio"])
    print(part_of.start_node)
    print(part_of.end_node)

    comp = record["c"]
    print(comp.labels)
    print(comp["name"])

    path = record["path"]
    print(path.start_node)
    print(path.end_node)
    print(len(path))
    print(path.relationships)
    # iter(path)



# Return all property keys defined in the graph (retained even if no current nodes/relationships use them)
#records, summary, keys = driver.execute_query(
#    """
#CALL db.propertyKeys()
#"""
#)
#print(records)

# Return all labels
#records, summary, keys = driver.execute_query(
#    """
#CALL db.labels()
#"""
#)
#print(records)











#with driver.session() as s:
   # s.execute_write(lambda tx: tx.run("MATCH (r) DETACH DELETE r"))
   # s.execute_write(lambda tx: tx.run("MATCH (n) DETACH DELETE n"))

    #s.execute_write(lambda tx: tx.run("MERGE (c:Component {name: 'Choline Chloride'})"))
    #s.execute_write(lambda tx: tx.run("MERGE (c:Component {name: 'Glycerol'})"))
    #s.execute_write(lambda tx: tx.run("MERGE (c:Component {name: 'Ethylene Glycol'})"))
    #s.execute_write(lambda tx: tx.run("MERGE (c:Component {name: 'Urea'})"))
    #s.execute_write(lambda tx: tx.run("MERGE (c:Component {name: 'Oxalic acid'})"))
    #s.execute_write(lambda tx: tx.run("MERGE (c:Component {name: 'P Toluenesulfonic acid'})"))
   # s.execute_write(lambda tx: tx.run("MERGE (c:Component {name: 'L Ascorbic acid'})"))
   # s.execute_write(lambda tx: tx.run("MERGE (c:Component {name: 'Malic acid'})"))
   # s.execute_write(lambda tx: tx.run("MERGE (c:Component {name: 'Formic acid'})"))
   # s.execute_write(lambda tx: tx.run("MERGE (c:Component {name: 'Acetic acid'})"))

    #s.execute_write(lambda tx: tx.run("MERGE (m:Mixture {name: 'Mixture 1'})"))
    #s.execute_write(lambda tx: tx.run("MERGE (m:Mixture {name: 'Mixture 2'})"))
    #s.execute_write(lambda tx: tx.run("MERGE (m:Mixture {name: 'Mixture 3'})"))
    #s.execute_write(lambda tx: tx.run("MERGE (m:Mixture {name: 'Mixture 4'})"))
    #s.execute_write(lambda tx: tx.run("MERGE (m:Mixture {name: 'Mixture 5'})"))
   # s.execute_write(lambda tx: tx.run("MERGE (m:Mixture {name: 'Mixture 6'})"))
   # s.execute_write(lambda tx: tx.run("MERGE (m:Mixture {name: 'Mixture 7'})"))
   # s.execute_write(lambda tx: tx.run("MERGE (m:Mixture {name: 'Mixture 8'})"))
   # s.execute_write(lambda tx: tx.run("MERGE (m:Mixture {name: 'Mixture 9'})"))

    
    #s.execute_write(lambda tx: tx.run("MATCH (c:Component {name: 'Choline Chloride'}), (m:Mixture {name: 'Mixture 1'}) MERGE (c)-[:PART_OF {ratio: 1}]->(m)"))
    #s.execute_write(lambda tx: tx.run("MATCH (c:Component {name: 'Choline Chloride'}), (m:Mixture {name: 'Mixture 2'}) MERGE (c)-[:PART_OF {ratio: 2}]->(m)"))
    #s.execute_write(lambda tx: tx.run("MATCH (c:Component {name: 'Choline Chloride'}), (m:Mixture {name: 'Mixture 3'}) MERGE (c)-[:PART_OF {ratio: 1}]->(m)"))
    #s.execute_write(lambda tx: tx.run("MATCH (c:Component {name: 'Choline Chloride'}), (m:Mixture {name: 'Mixture 4'}) MERGE (c)-[:PART_OF {ratio: 1}]->(m)"))
    #s.execute_write(lambda tx: tx.run("MATCH (c:Component {name: 'Choline Chloride'}), (m:Mixture {name: 'Mixture 5'}) MERGE (c)-[:PART_OF {ratio: 1}]->(m)"))
    #s.execute_write(lambda tx: tx.run("MATCH (c:Component {name: 'Choline Chloride'}), (m:Mixture {name: 'Mixture 6'}) MERGE (c)-[:PART_OF {ratio: 2}]->(m)"))
    #s.execute_write(lambda tx: tx.run("MATCH (c:Component {name: 'Choline Chloride'}), (m:Mixture {name: 'Mixture 7'}) MERGE (c)-[:PART_OF {ratio: 1}]->(m)"))
    #s.execute_write(lambda tx: tx.run("MATCH (c:Component {name: 'Choline Chloride'}), (m:Mixture {name: 'Mixture 8'}) MERGE (c)-[:PART_OF {ratio: 1}]->(m)"))
   # s.execute_write(lambda tx: tx.run("MATCH (c:Component {name: 'Choline Chloride'}), (m:Mixture {name: 'Mixture 9'}) MERGE (c)-[:PART_OF {ratio: 1}]->(m)"))


    #s.execute_write(lambda tx: tx.run("MATCH (c:Component {name: 'Glycerol'}), (m:Mixture {name: 'Mixture 1'}) MERGE (c)-[:PART_OF {ratio: 1}]->(m)"))
    #s.execute_write(lambda tx: tx.run("MATCH (c:Component {name: 'Ethylene Glycol'}), (m:Mixture {name: 'Mixture 2'}) MERGE (c)-[:PART_OF {ratio: 1}]->(m)"))
    #s.execute_write(lambda tx: tx.run("MATCH (c:Component {name: 'Urea'}), (m:Mixture {name: 'Mixture 3'}) MERGE (c)-[:PART_OF {ratio: 2}]->(m)"))
    #s.execute_write(lambda tx: tx.run("MATCH (c:Component {name: 'Oxalic acid'}), (m:Mixture {name: 'Mixture 4'}) MERGE (c)-[:PART_OF {ratio: 1}]->(m)"))
    #s.execute_write(lambda tx: tx.run("MATCH (c:Component {name: 'P Toluenesulfonic acid'}), (m:Mixture {name: 'Mixture 5'}) MERGE (c)-[:PART_OF {ratio: 1}]->(m)"))
    #s.execute_write(lambda tx: tx.run("MATCH (c:Component {name: 'L Ascorbic acid'}), (m:Mixture {name: 'Mixture 6'}) MERGE (c)-[:PART_OF {ratio: 1}]->(m)"))
    #s.execute_write(lambda tx: tx.run("MATCH (c:Component {name: 'Malic acid'}), (m:Mixture {name: 'Mixture 7'}) MERGE (c)-[:PART_OF {ratio: 1}]->(m)"))
    #s.execute_write(lambda tx: tx.run("MATCH (c:Component {name: 'Formic acid'}), (m:Mixture {name: 'Mixture 8'}) MERGE (c)-[:PART_OF {ratio: 2}]->(m)"))
    #s.execute_write(lambda tx: tx.run("MATCH (c:Component {name: 'Acetic acid'}), (m:Mixture {name: 'Mixture 9'}) MERGE (c)-[:PART_OF {ratio: 2}]->(m)"))

    #s.execute_write(lambda tx: tx.run("MATCH (m:Mixture {name: 'Mixture 10'}) RETURN m"))
    
