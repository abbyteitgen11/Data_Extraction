
from physical_property import PhysicalProperty

class Component:
    """
    This class represents a component in a Deep Eutectic Solvent. The 
    class members are essentially data values of physico-chemical 
    properties of interest as well as identifiers (CAS numbers, etc)
    purpose of the class is to create instances for the purpose of 
    serialisation through json so as not to have to re-create new
    instances every time we want to create a new database.

    name: str
 
    The following properties can be obtained from PubChemPy
 
    smiles: str
    inchi: str
    molecular_weight: float
    chemical_formula: str
    h_bond_acceptor_count: int (number of hydrogen-bond acceptor atoms)
    h_bond_donor_count: int (number of hydrogen-bond donor atoms)

    The following from CAS
 
    CAS: str (a CAS molecule identifier)

    properties: list[PhysicalProperty]

    This is a list of instances of the PhysicalProperty class, each having a name 
    (e.g. density, viscosity, etc), a value, units and a source for traceability.

    ...

    It is possible that we eventually want to include additional data, 
    so the constructor is designed to be flexible in this respect.

    """

    def __init__(self, 
                *,
                name = str,
                **kwargs
        ) -> None:

        """ Create an instance of Component. """

        self.name = name

        self.cas = kwargs.get("cas", "")
        self.smiles = kwargs.get("smiles", "")
        self.inchi = kwargs.get("inchi", "")
        self.cid = kwargs.get("cid", "")
        self.chemical_formula = kwargs.get("chemical_formula", "")
        self.properties = kwargs.get("properties", [])

    def to_dict(self) -> dict:

        """ Construct a dictionary representation of this instance. """

        return {
                 "name": self.name,
                 "cas": self.cas,
                 "smiles": self.smiles,
                 "inchi": self.inchi,
                 "cid": self.cid,
                 "chemical_formula": self.chemical_formula,
                 "properties": self.properties
        }

    @classmethod
    def from_dict(cls, compdict: dict):

        """ Reconstruct an instance from its dictionary representation. """

        return cls(
                    name = compdict["name"],
                    cas = compdict["cas"],
                    smiles = compdict["smiles"],
                    inchi = compdict["inchi"],
                    cid = compdict["cid"],
                    chemical_formula = compdict["chemical_formula"],
                    properties = compdict["properties"]
        )
