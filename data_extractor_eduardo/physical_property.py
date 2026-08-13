
from abc import ABC, abstractmethod
import re

class PhysicalProperty(ABC):

    """
    This class represents a physical or chemical property of a molecule or component. It is an abstract 
    base class because different properties may need different treatment to convert their units to a 
    standard one throughout the database. We want to keep
    traceability of the data that we store in the database, so each property must come with a name
    (e.g. density), a value, units, and origin, i.e. CAS, NIST, PubMed, or a doi if obtained from a
    scientific paper, or any other source.

    This is meant to serve as a base class of more specific properties, such as density, melting temperature, 
    etc, which may have special methods of their own, for example to perform unit conversion to a database
    standard and so on.

    This class, being an ABC (Abstract Base Class) cannot be directly instantiated; rather one would
    create an instance of one of its derived classes according to the kind of magnitude (e.g Temperature,
    Density, Mass, or Viscosity), defined below.

    """

    def __init__(
                  self,
                  *,
                  name: str,
                  value: float,
                  units: str,
                  comment: str,
                  source: str
       ) -> None:

       """ Create an instance of Property. """

       self.name = name
       self.value = value
       self.units = units
       self.comment = comment
       self.source = source

    def to_dict(self) -> dict:

       """ Create a dictionary representation. """

       return {
          "name": self.name,
          "value": self.value,
          "units": self.units,
          "comment": self.comment,
          "source": self.source
       }
                  
    @classmethod
    def from_dict(cls, propdict: dict):
        
        """ Create an instance from its dictionary representation. """

        return cls(
                   name = propdict["name"],
                   value = propdict["value"],
                   units = propdict["units"],
                   comment = propdict["comment"],
                   source = propdict["source"]
               )

    @abstractmethod
    def transform_units(self):

        """ Must be defined in children classes. """

        raise NotImplementedError

class UnrecognisedUnit(Exception):
    pass

class Conductivity(PhysicalProperty):

    """ A class to represent conductivity. """

    def __init__(
                  self, 
                  *,
                  name: str = "Conductivity",
                  value: float = 0.0,
                  units: str = "mS/cm",
                  comment: str = "",
                  source: str = None
    ) -> None:

        """ Construct an instance of Conductivity. """

        super().__init__(
                          name = name,
                          value = value,
                          units = units,
                          comment = comment,
                          source = source
                        )

        # now make sure that we use the standard units

        self.transform_units()

    def transform_units(self) -> None:

        """ This method transforms input units to mS/cm. """

        if re.match("^S/m|^Sm-", self.units, flags = re.IGNORECASE):

            self.value *= 10
            self.units = "mS/cm"

        elif re.match("^mS", self.units, flags = re.IGNORECASE):

            self.units = "mS/cm"
            return

        else:

            raise UnrecognisedUnit(f'Conductivity unit {self.units} not contemplated!')

class Density(PhysicalProperty):

    """ A class to represent density properties. """

    def __init__(
                  self, 
                  *,
                  name: str = "Density",
                  value: float = 0.0,
                  units: str = "g/cm3",
                  comment: str = "",
                  source: str = None
    ) -> None:

        """ Construct an instance of Density. """

        super().__init__(
                          name = name,
                          value = value,
                          units = units,
                          comment = comment,
                          source = source
                        )

        # now make sure that we use the standard units

        self.transform_units()

    def transform_units(self) -> None:

        """ This method transforms input units to g/cm3. """

        if re.match("^kg/m", self.units, flags = re.IGNORECASE):

            self.value *= 1.0e-3
            self.units = "g/cm3"

        elif re.match("^g/l", self.units, flags = re.IGNORECASE):

            self.value *= 1.0e-3
            self.units = "g/cm3"

        elif re.match("^g/c", self.units, flags = re.IGNORECASE):

            self.units = "g/cm3"
            return

        else:

            raise UnrecognisedUnit(f'Density unit {self.units} not contemplated!')

class DimensionlessProperty(PhysicalProperty):

    """ A class to represent dimensionless properties. """

    def __init__(
                  self, 
                  *,
                  name: str = "",
                  value: float = 0.0,
                  units: str = "dimensionless",
                  comment: str = "",
                  source: str = None
    ) -> None:

        """ Construct an instance of Density. """

        super().__init__(
                          name = name,
                          value = value,
                          units = units,
                          comment = comment,
                          source = source
                        )

        # now make sure that we use the standard units

        self.transform_units()

    def transform_units(self) -> None:
        pass

class SurfaceTension(PhysicalProperty):

    """ A class to represent Surface Tension. """

    def __init__(
                  self, 
                  *,
                  name: str = "Surface Tension",
                  value: float = 0.0,
                  units: str = "mN/m",
                  comment: str = "",
                  source: str = None
    ) -> None:

        """ Construct an instance of Surface Tension. """

        super().__init__(
                          name = name,
                          value = value,
                          units = units,
                          comment = comment,
                          source = source
                        )

        # now make sure that we use the standard units

        self.transform_units()

    def transform_units(self) -> None:

        """ This method transforms input units to mN/m. """

        if re.match("^N/m|^Nm-", self.units, flags = re.IGNORECASE):

            self.value *= 1.0e3
            self.units = "mN/m"

        elif re.match("^d|^dyne", self.units, flags = re.IGNORECASE):

            self.units = "mN/m"

        elif re.match("^mN", self.units, flags = re.IGNORECASE):

            self.units = "mN/m"
            return

        else:

            raise UnrecognisedUnit(f'Surface Tension unit {self.units} not contemplated!')

class Temperature(PhysicalProperty):

    """ A class to represent temperatures (e.g. melting, boiling, etc.) """

    def __init__(
                  self, 
                  *,
                  name: str = "Melting Temperature",
                  value: float = 0.0,
                  units: str = "C",
                  comment: str = "",
                  source: str = None
    ) -> None:

        """ Construct an instance of Temperature. """

        super().__init__(
                          name = name,
                          value = value,
                          units = units,
                          comment = comment,
                          source = source
                        )

        # now make sure that we use the standard units

        self.transform_units()

    def transform_units(self) -> None:

        """ This method transforms input units to centigrade. """

        if re.match("^k|^\u00B0k", self.units, flags = re.IGNORECASE):

            self.value -= 273.15
            self.units = "C"

        elif re.match("^f|^\u00B0f", self.units, flags = re.IGNORECASE):

            self.value = (self.value - 32.) * 5./9.
            self.units = "C"

        elif re.match("^c|^\u00B0c", self.units, flags = re.IGNORECASE):

            self.units = "C"
            return

        else:

            raise UnrecognisedUnit(f'Temperature unit {self.units} not contemplated!')

class Viscosity(PhysicalProperty):

    """ A class to represent viscosity. """

    def __init__(
                  self, 
                  *,
                  name: str = "Viscosity",
                  value: float = 0.0,
                  units: str = "mPa s",
                  comment: str = "",
                  source: str = None
    ) -> None:

        """ Construct an instance of Viscosity. """

        super().__init__(
                          name = name,
                          value = value,
                          units = units,
                          comment = comment,
                          source = source
                        )

        # now make sure that we use the standard units

        self.transform_units()

    def transform_units(self) -> None:

        """ This method transforms input units to centigrade. """

        if re.match("^pa", self.units, flags = re.IGNORECASE):

            self.value *= 1.0e3  # if units = Pa s
            self.units = "mPa s"

        elif re.match("^p^|^po", self.units, flags = re.IGNORECASE):

            self.value *= 1.0e-2  # if units = Poise
            self.units = "mPa s"

        elif re.match("^cp|^mpa", self.units, flags = re.IGNORECASE):

            self.units = "mPa s"
            return

        else:

            raise UnrecognisedUnit(f'Viscosity unit {self.units} is not contemplated!')

