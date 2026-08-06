
"""

This Python script is a first step towards creating a Graph Database for Deep Eutectic Solvents

We will use the data that is reported in the paper by Omar and Sadeghi, J. Mol. Liq. 384, (2023) 121899,
particularly in table 2 of this paper, listing a large number of DES and some of their 
measured properties. 

"""

from bs4 import BeautifulSoup
from chemicals import CAS_from_any
from chemicals.phase_change import Tm as melting_temperature
from chemicals.miscdata import lookup_VDI_tabular_data
from chemicals.identifiers import search_chemical
from lxml import etree
from neo4j import GraphDatabase
import pubchempy as pcp
import re
import requests

from component import Component
from paper import Paper
from process_property import obtain_NIST_property
#import pdb; pdb.set_trace()




def extract_values(node):
    values = []

    for info in node.get("Information", []):
        value = info.get("Value", {})

        # Most common case
        for item in value.get("StringWithMarkup", []):
            values.append(item.get("String"))

        # Numeric case sometimes
        if "Number" in value:
            values.append(value["Number"])

    return values

def find_heading(node, heading):
    # Is this node the one we're looking for?
    if node.get("TOCHeading") == heading:
        return node

    # Recurse through any child sections
    for child in node.get("Section", []):
        result = find_heading(child, heading)
        if result is not None:
            return result

    return None

# read the paper and generate an instance of Paper with it

file = "SadeghiDESReview.xml"

with open(file, "r", encoding="utf-8") as infile:
     document = infile.read()

tree = etree.fromstring(document.encode())

paper = Paper.from_xml(tree)

# In this paper the important information is in table 2;
# first we will create a unique list of HBAs and another of HBDs, 
# contained in the first and second column of table 2. Since
# many are repeated, we make sure that we only include unique entries

molecules = []

for n, row in enumerate(paper.tables[1].rows):

    if len(row) > 1:

       acceptor = row[0]
       donor = row[1]

       if acceptor not in molecules:
           molecules.append(acceptor)

       if donor not in molecules:
           molecules.append(donor)

# now loop over the distinct molecules and create a dictionary of with the 
# molecule name as key and an instance of class Component as value

# We need to retrieve data from different places... first from CAS

search_url = "https://commonchemistry.cas.org/api/search"
detail_url = "https://commonchemistry.cas.org/api/detail"
headers = {
    "X-API-KEY": "QGPNaL0ztr9KDu2nLNovhTnwhUdFRin5KGVNA8D3"  
}

components = []

for molecule in molecules[:3]:

    # obtain the CAS identifier from chemicals

    cas_id = CAS_from_any(molecule)

    molecule_mdata = search_chemical(cas_id)

    # get the CID for this molecule

    compounds = pcp.get_compounds(molecule_mdata.smiles, "smiles")

    cid = compounds[0].cid

    # next let's try to obtain some physical data from NIST

    property_list = []

    properties = [
        "Melting Point",
        "Boiling Point",
        "Density",
        "Viscosity",
    ]

    for prop in properties:

        request = requests.get(
      f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON",
           params={"heading": prop}
                                 )
        data = request.json()

        try: 
        
           mp_node = find_heading(data["Record"]["Section"][0], prop)
           mp_values = extract_values(mp_node)

           property_list += \
               obtain_NIST_property(key=prop, info=mp_values)

        except Exception as e:
           print(f"No {prop} data in NIH PubChem for {molecule}")

        # if we found some data, process it


        try:
        
        # here we query to get the properties available at CAS

           response = requests.get(
                              detail_url,
                              headers = headers,
                              params={"cas_rn": cas_id}
                           )

           data = response.json()
           properties = data.get("experimentalProperties", [])

           if properties:

               for prop in properties:

                   property_list.append(obtain_CAS_property(prop))
 
        except Exception as e:
           print(f"Error searching for {molecule}: {e}")


        # now from NIST

        nist_id = "C" + cas_id.replace("-", "")

        nist_request = requests.get(
              "https://webbook.nist.gov/cgi/cbook.cgi",
              params={
                   "ID": nist_id,
                   "Units": "SI"
                  }
        )

        soup = BeautifulSoup(nist_request.text, "html.parser")

        text = soup.get_text(" ", strip=True)

        m = re.search(r'Normal melting point\s+([0-9.]+)\s*K', text)

        if m:
           melting_nist = float(m.group(1))

        m = re.search(r'Density\s+([0-9.]+)\s*g/cm3', text)

        if m:
            density_nist = float(m.group(1))



    components.append(Component(
                   name = molecule,
                   smiles = compounds[0].smiles,
                   inchi = compounds[0].inchi,
                   cid = compounds[0].cid,
                   cas = cas_id,
                   chemical_formula = compounds[0].molecular_formula,
                   properties = property_list
            )
    )
print(components)
print('Got here!')
