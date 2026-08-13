
import re

from physical_property import Conductivity
from physical_property import Density
from physical_property import DimensionlessProperty
from physical_property import PhysicalProperty
from physical_property import SurfaceTension
from physical_property import Temperature
from physical_property import Viscosity

def obtain_NIST_property(
       key: str,
       info: list[str]
    ) -> PhysicalProperty:

    """
    This function processes a list of string constants, where each string 
    may (in some cases may not) contain data (temperature, density, etc) and
    appropriate units, obtained from the NIH PubChem database, and transforms
    it to an appropriate instance of PhysicalProperty, required
    for the corresponding instance of Component.

    key: provides information on the type of magnitude for which info is being provided
    info: list of string constants containing information to be processed

    """

    if re.match("^melting", key, re.IGNORECASE):

        name = "melting point"

        data_points = []

        for item in info:  # often info contains more than one set of data

            if not isinstance(item, str): # sometimes items can be lists; ignore
                continue

        # the following regex matches "100-110", "100.0-110.0", etc, to handle range cases

            match = re.match("^(-?\d*\.?\d*)(\s*-\s*|\s*to\s*)(-?\d*\.?\d*)", item)

            if match:
                value = (float(match.group(1)) + float(match.group(3)))/2.
            elif re.match("^(-?\d*\.?\d*)", item):
                value = float(re.match("^(-?\d*\.?\d*)", item).group(0))
            else:
                value = False

            if re.search("\s+\u00B0?C\s+", item):
                units = "C"
            elif re.search("\s+\u00B0?F\s+", item):
                units = "F"
            elif re.search("\s+\u00B0?K\s+", item):
                units = "K"

            comment = item  # we keep the entire string as comment; useful for checking

            if value:
               data_points.append(Temperature(
                     name = name,
                     value = value,
                     units = units,
                     comment = comment,
                     source = "NIST-NIH"
                  )
               )

        return data_points

    if re.match("^boiling", key, re.IGNORECASE):

        name = "boiling point"

        data_points = []

        for item in info:  # often info contains more than one set of data

            if not isinstance(item, str): # sometimes items can be lists; ignore
                continue

            # the following regex matches "100-110", "100.0-110.0", etc, 

            match = re.match("^(-?\d*\.?\d*)(\s*-\s*|\s*to\s*)(-?\d*\.?\d*)", item)
        
            if match:
                value = (float(match.group(1)) + float(match.group(3)))/2.
            elif re.match("^(-?\d*\.?\d*)", item):
                value = float(re.match("^(-?\d*\.?\d*)", item).group(0))
            else:
                value = False

            if re.search("\s+\u00B0?C\s+", item):
                units = "C"
            elif re.search("\s+\u00B0?F\s+", item):
                units = "F"
            elif re.search("\s+\u00B0?K\s+", item):
                units = "K"

            comment = item

            if value:
                data_points.append(Temperature(
                     name = name,
                     value = value,
                     units = units,
                     comment = comment,
                     source = "NIST-NIH"
                    )
                )

        return data_points

    elif re.match("^density", key, re.IGNORECASE):

        name = "density"

        data_points = []

        for item in info: # often info contains more than one set of data
       
            if not isinstance(item, str): # sometimes items can be lists; ignore
                continue

        # the following regex matches range cases

            match = re.match("^(\d*\.?\d*)(\s*-\s*|\s*to\s*)(\d*\.?\d*)", item)

            if match:
                value = (float(match.group(1)) + float(match.group(3)))/2.
            elif re.match("^(-?\d*\.{1}\d*)", item):
                value = float(re.match("^(-?\d*\.?\d*)", item).group(0))
            else:
                value = False

            # we are going to try to deduce the units from the value, because
            # the format of density entries is too general 

            dens = repr(value)

            if re.match("^\d?\.\d*", dens):  # matches 1.34, units must be g/cm^3
                units = "g/cm3"
            elif re.match("^\d{3,4}\.\d*", dens): # matches 1340.0, units must be kg/m3
                units = "kg/m3"

            comment = item

            if value:
               data_points.append(Density(
                     name = name,
                     value = value,
                     units = units,
                     comment = comment,
                     source = "NIST-NIH"
                   )
               )

        return data_points

    elif re.match("^viscosity", key, re.IGNORECASE):

        name = "viscosity"

        data_points = []

        for item in info: # often info contains more than one set of data
       
            if not isinstance(item, str): # sometimes items can be lists; ignore
                continue

        # the following regex matches range cases

            match = re.match("^(\d*\.?\d*)(\s*-\s*|\s*to\s*)(\d*\.?\d*)", item)

            if match:
                value = (float(match.group(1)) + float(match.group(3)))/2.
            elif re.match("^(-?\d*\.?\d*)", item):
                value = float(re.match("^(-?\d*\.?\d*)", item).group(0))
            else:
                value = False

            if re.search("\s+Pa\s?s\s+", item):
                units = "Pa s"
            elif re.search("\s+mPa(-?|\s?)s\s+", item):
                units = "mPa s"
            elif re.search("\s+Po\s+|\s+P\s+", item):
                units = "Poise"
            elif re.search("\s+cP\s+", item): # centipoise == mPa s
                units = "mPa s"

            comment = item

            if value:
               data_points.append(Density(
                     name = name,
                     value = value,
                     units = units,
                     comment = comment,
                     source = "NIST-NIH"
                   )
               )

        return data_points

def obtain_CAS_property(info: dict) -> PhysicalProperty:

    """
    This function processes an input dictionary containing data obtained from CAS web 
    resource, and transforms it to an appropriate instance of PhysicalProperty, required
    for the corresponding instance of Component.

    info: dict containing a description of the physical property.

    """

    if re.match("^melting", info["name"], re.IGNORECASE):

        name = "melting point"

        words = info["property"].split()

        # the following regex matches "100-110", "100.0-110.0", etc, to handle range cases

        if re.match("^(-?\d*\.?\d*)-(-?\d*\.?\d*)", words[0]):
            value = (float(match.group(1)) + float(match.group(2)))/2.
        elif re.match("^(-?\d*\.?\d*)", words[0]):
            value = float(words[0])
        else:
            value = NaN

        units = words[1]

        if len(words) > 2:
            comment = ""
            for word in words[2:]:
                comment += word

        return Temperature(
                  name = name,
                  value = value,
                  units = units,
                  comment = comment,
                  source = "CAS"
               )  

    if re.match("^boiling", info["name"], re.IGNORECASE):

        name = "boiling point"

        words = info["property"].split()

        # the following regex matches "100-110", "100.0-110.0", etc, to handle range cases

        if re.match("^(-?\d*\.?\d*)-(-?\d*\.?\d*)", words[0]):
            value = (float(match.group(1)) + float(match.group(2)))/2.
        elif re.match("^(-?\d*\.?\d*)", words[0]):
            value = float(words[0])
        else:
            value = NaN

        units = words[1]

        if len(words) > 2:
            comment = ""
            for word in words[2:]:
                comment += word

        return Temperature(
                  name = name,
                  value = value,
                  units = units,
                  comment = comment,
                  source = "CAS"
               )

    elif re.match("^density", info["name"], re.IGNORECASE):

        name = "density"
       
        words = info["property"].split()

        # the following regex matches "100-110", "100.0-110.0", etc, to handle range cases

        if re.match("^(-?\d*\.?\d*)-(-?\d*\.?\d*)", words[0]):
            value = (float(match.group(1)) + float(match.group(2)))/2.
        elif re.match("^(-?\d*\.?\d*)", words[0]):
            value = float(words[0])
        else:
            value = NaN

        if re.match("x", words[1], re.IGNORECASE): # probably means there is an exponent 
            exponent = float(words[2])
            units = words[3]
            if len(words) > 4:
               comment = ""
               for word in words[4:]:
                   comment += word
        else:
           units = words[1]
           if len(words) > 2:
               comment = ""
               for word in words[2:]:
                   comment += word

        return Density(
                 name = name,
                 value = value,
                 units = units,
                 comment=comment,
                 source = "CAS"
               )




