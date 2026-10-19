"""
Turn the number a table printed into the number the graph stores.

Until now nothing converted a property VALUE. `extract_table.read_condition` converted
K to Celsius for *condition* columns, and `_unit_for` canonicalised the unit's
SPELLING -- so a melting point printed in kelvin reached the graph as
`Melting_point = 285.15, unit = "K"`, sitting in the same distribution as one in
Celsius, and passing the -150..400 plausible range without a murmur. One property, two
scales, no flag. Any model trained on that column learns the mixture of the two.

Three things a header does that have to be undone before a value means anything:

    10-3rho (kg m-3)     a multiplier written into the header, and a unit nothing
                         recognised. 1.1979 is 1197.9 kg m-3 = 1.1979 g cm-3; read the
                         multiplier wrong and the answer is out by 1000.
    gamma (mN m-1) 298 K the measurement temperature, in the slot a unit would occupy.
    viscosity (cSt)      not viscosity at all -- KINEMATIC viscosity, mm2/s, which needs
                         the density to become mPa s. Refused rather than relabelled.

Conversions are affine, `canonical = value * scale + offset`, which covers kelvin as
well as kilograms per cubic metre and keeps one code path. The unconverted number is
kept alongside as `Value_as_written`: that is what `validate.check_fidelity` compares
against the cell, so a broken conversion shows up as a changed value beside an unchanged
as-written one, rather than as nothing at all.
"""
import re

from . import config

# canonical unit -> {written unit (normalised): (scale, offset)}
#
# Keys are normalised by `_key` below: lower-cased, unicode minus and middot folded,
# spaces and the multiplication signs removed. So "kg m-3", "kg/m3" and "kg·m−3" are one
# key. The identity entry for each canonical unit is added automatically.
CONVERSIONS = {
    "C": {
        "k": (1.0, -273.15),
        "f": (5.0 / 9.0, -32.0 * 5.0 / 9.0),
        "c": (1.0, 0.0), "°c": (1.0, 0.0), "℃": (1.0, 0.0), "oc": (1.0, 0.0),
    },
    "g*cm^-3": {
        "kgm-3": (1e-3, 0.0), "kg/m3": (1e-3, 0.0), "kgm3": (1e-3, 0.0),
        "gm-3": (1e-6, 0.0),
        "gl-1": (1e-3, 0.0), "g/l": (1e-3, 0.0),
        "kgl-1": (1.0, 0.0), "kg/l": (1.0, 0.0), "kg/dm3": (1.0, 0.0),
        "g/ml": (1.0, 0.0), "gml-1": (1.0, 0.0), "g/cm3": (1.0, 0.0),
        "gcm-3": (1.0, 0.0), "g/cc": (1.0, 0.0),
    },
    "mPa*s": {
        "pas": (1e3, 0.0), "pa.s": (1e3, 0.0),
        "cp": (1.0, 0.0), "mpas": (1.0, 0.0),
        # Poise: 1 P = 0.1 Pa s = 100 mPa s.
        "p": (100.0, 0.0),
    },
    "mS*cm^-1": {
        "sm-1": (10.0, 0.0), "s/m": (10.0, 0.0),
        "scm-1": (1e3, 0.0), "s/cm": (1e3, 0.0),
        "μscm-1": (1e-3, 0.0), "uscm-1": (1e-3, 0.0),
        "μs/cm": (1e-3, 0.0), "us/cm": (1e-3, 0.0),
        "msm-1": (1e-2, 0.0), "ms/m": (1e-2, 0.0),
        "mscm-1": (1.0, 0.0), "ms/cm": (1.0, 0.0),
    },
    "mN*m^-1": {
        "nm-1": (1e3, 0.0), "n/m": (1e3, 0.0),
        # 1 dyn/cm = 1 mN/m exactly.
        "dyncm-1": (1.0, 0.0), "dyn/cm": (1.0, 0.0),
        "mnm-1": (1.0, 0.0), "mn/m": (1.0, 0.0),
    },
    "W*m^-1*K^-1": {
        "mwm-1k-1": (1e-3, 0.0),
        "wm-1k-1": (1.0, 0.0), "w/(mk)": (1.0, 0.0), "w/mk": (1.0, 0.0),
    },
    "wt%": {
        "%": (1.0, 0.0), "wt%": (1.0, 0.0), "w/w%": (1.0, 0.0), "%w/w": (1.0, 0.0),
        "ppm": (1e-4, 0.0),
        "g/100g": (1.0, 0.0),
    },
    "mV": {"mv": (1.0, 0.0), "v": (1e3, 0.0)},
    "mg*mL^-1": {
        "mgml-1": (1.0, 0.0), "mg/ml": (1.0, 0.0), "g/l": (1.0, 0.0), "gl-1": (1.0, 0.0),
        "μgml-1": (1e-3, 0.0), "ugml-1": (1e-3, 0.0),
        "μg/ml": (1e-3, 0.0), "ug/ml": (1e-3, 0.0),
        "mg/l": (1e-3, 0.0), "g/100ml": (10.0, 0.0), "g/ml": (1e3, 0.0),
    },
    "J*g^-1*K^-1": {
        "jg-1k-1": (1.0, 0.0), "j/gk": (1.0, 0.0), "j/(gk)": (1.0, 0.0),
        "kjkg-1k-1": (1.0, 0.0), "kj/kgk": (1.0, 0.0),
    },
    "kcal*mol^-1": {"kcalmol-1": (1.0, 0.0), "kcal/mol": (1.0, 0.0)},
}

# Units that name a DIFFERENT PHYSICAL QUANTITY from the property they are printed
# under. Converting these needs data we do not have (kinematic -> dynamic viscosity
# needs the density; a relative density needs the reference), so the value is kept with
# `status = "unhandled_unit"` and does not load. Silently treating cSt as mPa s would
# put a number 1000x off into the middle of the distribution.
FOREIGN_UNITS = {
    "Viscosity": {"cst": "kinematic viscosity (mm2/s); needs the density to convert",
                  "mm2s-1": "kinematic viscosity; needs the density to convert",
                  "mm2/s": "kinematic viscosity; needs the density to convert",
                  "m2s-1": "kinematic viscosity; needs the density to convert"},
    "Density": {"relativedensity": "relative density, not an absolute one",
                "specificgravity": "specific gravity, not an absolute density"},
}


def _key(raw):
    """Normalise a written unit so its many spellings collapse to one key."""
    text = str(raw or "").strip().lower()
    text = (text.replace("−", "-").replace("–", "-").replace("—", "-")
                .replace("⋅", "").replace("·", "").replace("×", "").replace("*", ""))
    # Superscript digits, as JATS prints them when <sup> is flattened.
    for sup, plain in zip("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-"):
        text = text.replace(sup, plain)
    text = re.sub(r"[()\[\]{}]", "", text)
    return re.sub(r"\s+", "", text)


def to_canonical(value, unit_as_written, prop):
    """One printed value -> (value, unit, status) in the property's canonical unit.

    `status` is "" when the value is usable. Otherwise it names what stopped it, using
    the same vocabulary as `schema.COMPONENT_PROPERTY_STATUS_ORDER`, and the caller
    keeps the row but does not load it.

    A property whose canonical unit is "" (refractive index, pH) is dimensionless: the
    header's "nD" or "-" is a symbol, not a unit, so there is nothing to convert.
    """
    canonical = config.PROPERTY_UNITS.get(prop, "")
    if value is None:
        return None, canonical or None, ""
    if not canonical:
        return value, None, ""

    written = _key(unit_as_written)
    foreign = FOREIGN_UNITS.get(prop, {})
    if written and written in foreign:
        return value, str(unit_as_written).strip(), "unhandled_unit"

    table = CONVERSIONS.get(canonical, {})
    if not written:
        # No unit printed. The property's own unit is the only available reading, and
        # asserting a scale nobody stated would be worse than saying so.
        return value, canonical, ""
    if written == _key(canonical):
        return value, canonical, ""
    if written in table:
        scale, offset = table[written]
        return round(value * scale + offset, 6), canonical, ""
    if written in config.UNIT_ALIASES:
        alias = config.UNIT_ALIASES[written]
        if _key(alias) == _key(canonical):
            return value, canonical, ""
        if _key(alias) in table:
            scale, offset = table[_key(alias)]
            return round(value * scale + offset, 6), canonical, ""
    return value, str(unit_as_written).strip(), "unhandled_unit"


# A power-of-ten multiplier written into the header rather than into the unit:
# "10-3rho (kg m-3)", "1014 sigma", "106 x bv". The exponent may be a unicode minus, and
# the digits may run together with the base ("1014" is 10^14, printed with a superscript
# that flattening lost). Only a leading factor counts -- "10-3 Pa s" inside the unit
# parenthesis is part of the unit and is handled by CONVERSIONS.
_SCALE = re.compile(r"^\s*10\s*([-−–]?)\s*(\d{1,2})\s*(?:[×x·*]|\s)?\s*(?=[^\d]|$)")


def header_scale(header):
    """The multiplier a header writes in front of the quantity. -> float (1.0 if none).

    Getting this wrong is a factor-of-1000 error, so it deliberately fires only on a
    leading "10^n" and never on a bare number: a header reading "25 C" or "298 K" states
    a temperature, and "1st step" states a sequence.
    """
    text = str(header or "").strip()
    # Superscripts survive flattening as their own characters often enough to matter.
    for sup, plain in zip("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-"):
        text = text.replace(sup, plain)
    match = _SCALE.match(text)
    if not match:
        return 1.0
    sign = -1 if match.group(1) else 1
    exponent = sign * int(match.group(2))
    if abs(exponent) > 20:
        return 1.0
    # The header names the quantity that was TABULATED: "10-3 rho" means the printed
    # number IS 10^-3 times rho, so rho is the printed number times 10^+3. Hence the
    # negated exponent -- 1.1979 under "10-3 rho (kg m-3)" is 1197.9 kg m-3, and 4.477
    # under "103 RMSD" is 0.004477.
    return 10.0 ** -exponent


# A temperature sitting where a unit would be: "gamma (mN m-1) 298 K",
# "Viscosity/10-3 Pa s 25 C", "eta (mPa s) at 180 C".
_HEADER_TEMPERATURE = re.compile(
    r"(?:\bat\s*)?(\d{2,3}(?:\.\d+)?)\s*(?:°|˚|º)?\s*(K|C|℃)\b", re.I)


def temperature_in_header(header):
    """A measurement temperature stated in a column header. -> Celsius, or None.

    Only fires on a plausible laboratory temperature so a stray number cannot become
    one: 298 K and 25 C are read, 10 K and 900 C are not.
    """
    for number, unit in _HEADER_TEMPERATURE.findall(str(header or "")):
        value = float(number)
        if unit.upper() == "K":
            if not 200 <= value <= 500:
                continue
            return round(value - 273.15, 3)
        if not -50 <= value <= 400:
            continue
        return round(value, 3)
    return None


def basis_of(prop, printed):
    """Which basis a property was reported on. -> (basis, status).

    Heat capacity per gram and per mole are different quantities with no conversion
    between them short of the molar mass; "polarity" names E_T(30) in kcal/mol and the
    dimensionless Kamlet-Taft parameters. A property declaring `basis` in
    config.PROPERTIES therefore has to have it established from what was printed, and
    gets `status = "ambiguous_basis"` when it cannot be.
    """
    patterns = config.PROPERTY_BASIS.get(prop)
    if not patterns:
        return "", ""
    text = str(printed or "").lower()
    for basis, pattern in patterns.items():
        if re.search(pattern, text):
            return basis, ""
    return "", "ambiguous_basis"


# What a unit measures, when it plainly measures something no DES property is. These
# are the units models have been seen forcing into the vocabulary: a critical pressure in
# bar labelled Density, a conversion in % labelled pH, a solvation energy in kcal/mol
# labelled Solubility, a cavity volume in cubic angstroms labelled Solubility.
_DIMENSIONS = (
    ("pressure", r"^(bar|kbar|mbar|kpa|mpa|gpa|pa|atm|psi|mmhg|torr)$"),
    ("energy", r"kcal|kj|^j$|jmol|j/mol|^ev$|kt$"),
    ("percentage", r"%|percent|ppm|ppb"),
    ("length/area/volume", r"^(å|a|å3|a3|å2|a2|nm|pm|μm|um|mm|cm|m|nm2|nm3|å\^3|ang.*)$"),
    ("wavenumber", r"cm-1$|cm\^-1$"),
    ("time", r"^(s|sec|min|h|hr|hours?|days?)$"),
    ("mass", r"^(g|mg|kg|μg|ug|da|kda|g/mol|gmol-1)$"),
    ("voltage", r"^(v|mv)$"),
    ("concentration", r"^(m|mm|mol/l|moll-1|mmol/l|mg/l|μg/l|ug/l)$"),
    ("temperature", r"^(k|°c|c|℃|°f|f)$"),
)


def dimension_of(unit_as_written):
    """A coarse name for what a unit measures. -> str, or "" when unrecognised."""
    key = _key(unit_as_written)
    for name, pattern in _DIMENSIONS:
        if re.search(pattern, key):
            return name
    return ""


def unit_conflict(unit_as_written, prop):
    """Does the printed unit contradict the property? -> reason, or "" when it fits.

    Converts-cleanly and no-unit-at-all both fit. A unit naming a related quantity the
    converter refuses on purpose (cSt for viscosity) also fits: the value is kept and
    flagged `unhandled_unit` later, which is the right outcome. What does NOT fit is a
    unit that plainly measures something else -- a pressure, an energy, a percentage --
    because then the column is a different quantity and every value in it would land in
    the wrong distribution.
    """
    written = _key(unit_as_written)
    if not written or written in ("-", "nd", "n", "nd20", "nd25", "ph", "dimensionless",
                                  "unitless", "a.u.", "au"):
        return ""
    _value, _unit, status = to_canonical(1.0, unit_as_written, prop)
    if not status:
        canonical = config.PROPERTY_UNITS.get(prop, "")
        if canonical or not dimension_of(unit_as_written):
            return ""
    elif written in FOREIGN_UNITS.get(prop, {}):
        return ""
    dimension = dimension_of(unit_as_written)
    canonical = config.PROPERTY_UNITS.get(prop, "")
    expected = "temperature" if prop in config.PHASE_TRANSITION_PROPERTIES else \
        (dimension_of(canonical) or ("dimensionless" if not canonical else canonical))
    if dimension and dimension != expected:
        wanted = "dimensionless" if not canonical else f"measured in {canonical}"
        return (f"the unit {unit_as_written!r} measures {dimension}, but {prop} is "
                f"{wanted}")
    return ""
