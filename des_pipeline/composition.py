"""
What a DES is made of, read from however the paper chose to write it.

Every route used to assume one layout: one component per column and the ratio in a
column of its own. Papers do not cooperate. Across the thirteen papers reviewed:

    ChCl:Gly (1:2)                 the whole DES in one cell, ratio in brackets
    Urea, 1:2                      ratio after a comma, in the component cell
    Glycerol/Ethylene glycol       two components in one slot
    ChCl:LA  |  1:1 +20% water     water written into the ratio cell
    ChCl:LA  |  4% water           ...or into a component cell
    L-Pro | PGL | 2                a single number meaning HBA:HBD = 2:1
    DES 1:4 + 10 %wt EC            the paper's own DES, an additive by weight
    CEM5, DES5, [ChCl][Gl]3        a code that only the paper defines

Each of those was either dropped or stored as one long "component" name, so the same
solvent became several unrelated mixtures and water vanished from compositions it
belongs to. This module turns any of them into one `Composition`.

Two rules matter more than the parsing:

  * Water is a COMPONENT. "ChCl:LA 1:1 + 20% water" is a different solvent from the
    dry one, and recording the water as a separate property split one solvent into a
    dry mixture plus a number.
  * What the paper wrote is never overwritten. A molar ratio computed from weight
    fractions is stored BESIDE the ratio as written, flagged `computed`, because the
    conversion uses PubChem molecular weights that the paper never stated.
"""
import copy
import json
import re
from dataclasses import dataclass, field

from . import config, xml_utils

WATER_MW = 18.015
_WATER = re.compile(r"^\s*(?:water|h2o|h₂o|h 2 o|deionized water|deionised water|"
                    r"di water|distilled water|milli-?q water)\s*$", re.I)

# A percentage written with a basis: "20%", "20 wt%", "20 wt.%", "20%wt", "20% (w/w)",
# "20 vol%", "20% v/v". The basis decides whether a molar ratio can be computed.
_PCT = (r"(\d+(?:\.\d+)?)\s*(wt\s*\.?\s*%|%\s*wt\.?|%\s*\(?\s*w\s*/\s*w\s*\)?|"
        r"w\s*/\s*w\s*%|vol\s*\.?\s*%|%\s*\(?\s*v\s*/\s*v\s*\)?|v\s*/\s*v\s*%|"
        r"mol\s*\.?\s*%|%\s*\(?\s*mol\s*\)?|%)")
# "+20% water", "+ 10 %wt EC", "with 5 wt% H2O", "4% water"
_ADDITIVE_AFTER = re.compile(
    r"(?:^|[+,;&]|\bwith\b|\band\b)\s*" + _PCT + r"\s*(?:of\s+)?"
    r"([A-Za-z][\w\-\[\]()₂]*(?:\s+water)?)?", re.I)
# "water (20 wt%)", "H2O 20%"
_WATER_BEFORE = re.compile(r"\b(water|h2o)\s*\(?\s*" + _PCT + r"\s*\)?", re.I)

# A ratio is digits separated by colons: "1:2", "1 : 1.5", "1:1:2".
_RATIO = r"\d+(?:\.\d+)?(?:\s*:\s*\d+(?:\.\d+)?)+"
# "(2:1, 1:1, 1:2)" -- several ratios of one DES. Kept as written, never picked from.
_RATIO_LIST = re.compile(r"\(\s*(" + _RATIO + r"(?:\s*[,;]\s*" + _RATIO + r")+)\s*\)")
# The ratio at the end of a mixture string: "(1:2)", ", 1:2", " 1:2", "(molar ratio 1:2)"
_RATIO_TAIL = re.compile(r"[\s,;(]*(?:molar\s+ratio\s*[:=]?\s*|ratio\s*[:=]?\s*)?"
                         r"(" + _RATIO + r")\s*\)?\s*$", re.I)
# A header saying the column holds HBA:HBD (or the reverse) as one number.
_HBA_HBD = re.compile(r"hba\s*[:/]\s*hbd|hba\s+to\s+hbd", re.I)
_HBD_HBA = re.compile(r"hbd\s*[:/]\s*hba|hbd\s+to\s+hba", re.I)
_RATIO_HEADER = re.compile(r"ratio", re.I)
# Group-contribution fragments and other notation that is not a chemical name:
# "-CH3", ">N-", "=O", "-OH (Alcohol)".
_FRAGMENT = re.compile(r"^\s*[-–>=<]")
# A stoichiometric coefficient on a name: "2PTSA" (see extract_table.split_coefficient).
_COEFFICIENT = re.compile(r"^(\d{1,2})([A-Za-z\[].*)$")


@dataclass
class Composition:
    """One DES as the paper described it, plus what can be derived from that."""

    written: str = ""                     # the text it was read from, verbatim
    components: list = field(default_factory=list)      # names, water included
    components_written: list = field(default_factory=list)  # before resolution
    parts: list = field(default_factory=list)  # molar parts aligned with components
    ratio_raw: str = ""                   # the ratio exactly as written
    ratio_molar: str = ""                 # molar ratio over ALL components, when known
    ratio_source: str = ""                # as_written | computed | inferred | coefficients
    basis: str = ""                       # molar | wt | vol | mol | "" (unknown)
    water_fraction: float = None          # 0.2 for "20% water", on `water_basis`
    water_basis: str = ""
    additives: dict = field(default_factory=dict)        # name -> (fraction, basis)
    aliases: dict = field(default_factory=dict)  # "ChCl" -> "Choline Chloride", as written
    code: str = ""                        # the paper's code for it, when one resolved it
    source: str = ""                      # where a dictionary entry came from
    notes: list = field(default_factory=list)

    # --- views the CSV writers want ---------------------------------------------------
    def slots(self, n=3):
        """-> ([n names], [n parts], flag). Overflow beyond n is flagged, not dropped."""
        names = list(self.components)
        parts = list(self.parts) + [None] * (len(names) - len(self.parts))
        flag = "quaternary+" if len(names) > n else ""
        return ((names + [None] * n)[:n], (parts + [None] * n)[:n], flag)

    @property
    def n_non_water(self):
        return sum(1 for c in self.components if not is_water(c))

    @property
    def key_ratio(self):
        """The ratio that identifies this solvent: molar over all components if known."""
        return self.ratio_molar or self.ratio_raw

    def label(self):
        names = ":".join(c for c in self.components if c)
        ratio = self.ratio_molar or self.ratio_raw
        return f"{names} ({ratio})" if ratio else names

    def note(self, text):
        if text not in self.notes:
            self.notes.append(text)


def is_water(name):
    return bool(_WATER.match(str(name or "")))


def _ratio_parts(text):
    return [float(p) for p in re.split(r"\s*:\s*", text.strip()) if p.strip()]


def _fmt(x):
    return f"{x:.3g}" if x is not None else "?"


def basis_of(unit_text):
    """'wt%' -> 'wt', '% (v/v)' -> 'vol', '%' -> 'wt' (the common default, noted)."""
    t = re.sub(r"\s+", "", str(unit_text or "").lower())
    if "v/v" in t or "vol" in t:
        return "vol"
    if "mol" in t:
        return "mol"
    return "wt"


# ---------- molecular weights ----------
_MW = None


def molecular_weight(name):
    """PubChem molecular weight for a component name, from the shared cache. -> float."""
    global _MW
    if is_water(name):
        return WATER_MW
    if _MW is None:
        _MW = {}
        try:
            cache = json.loads(config.COMPONENT_CACHE.read_text())
        except (OSError, ValueError):
            cache = {}
        for key, row in cache.items():
            mw = (row or {}).get("molecular_weight")
            if mw:
                _MW[str(key).strip().lower()] = float(mw)
                matched = str((row or {}).get("matched_name") or "").strip().lower()
                if matched:
                    _MW.setdefault(matched, float(mw))
    return _MW.get(str(name or "").strip().lower())


# ---------- splitting a mixture string into names ----------
def _letters(text):
    return len(re.findall(r"[A-Za-z]", text))


# A stereodescriptor pair: the "/" belongs to one name. "ChCl/U" (Azad) is two
# components; a one-letter abbreviation like U is still a component.
_STEREO = re.compile(r"(?<![A-Za-z])(?:[DLRSEZ]{1,2}|dl|rs)\s*/\s*(?:[DLRSEZ]{1,2}|dl|rs)\s*-")


def split_names(text):
    """'ChCl:Gly' / 'A/B' / 'A + B' / 'ChCl-Urea' -> list of names. -> [text] if one."""
    t = str(text or "").strip(" ,;")
    if not t:
        return []
    # ":" is the commonest DES separator; "/" also appears inside one name as a
    # stereodescriptor ("D/L-Proline"), hence the three-letter rule from extract_table.
    for sep, min_letters in ((":", 1), ("/", 1), (" + ", 2), ("+", 3), (" – ", 2),
                             ("–", 3), (" - ", 2)):
        if sep not in t:
            continue
        if sep == "/" and _STEREO.search(t):
            continue                       # "D/L-Proline", "R/S-": one racemic compound
        parts = [p.strip(" ,;") for p in t.split(sep)]
        parts = [p for p in parts if p]
        if len(parts) >= 2 and all(_letters(p) >= min_letters or is_water(p)
                                   for p in parts):
            return parts
    # A hyphen joins two abbreviations ("ChCl-Urea", "LiTFSI-TFA") but is part of
    # most chemical names ("L-Proline", "N-methylurea", "1,2-propanediol"). Split only
    # when the left side carries two capitals and the right side starts with one.
    match = re.match(r"^([A-Za-z0-9\[\]()]*[A-Z][A-Za-z0-9\[\]()]*[A-Z][A-Za-z0-9\[\]()]*)"
                     r"-([A-Z\[].*)$", t)
    if match and _letters(match.group(1)) >= 2 and _letters(match.group(2)) >= 2:
        return [match.group(1), match.group(2)]
    return [t]


# A footnote marker left on the end of a name when the cell was flattened:
# "1-methyl-3-octyl imidazolium chloride a", "Urea*", "Glycerol†".
_TRAILING_MARKER = re.compile(r"(?:\s+[a-z]|[*†‡§¶#]+)\s*$")
# "Choline Chloride (ChCl)" -- a full name with the paper's abbreviation for it.
_NAME_WITH_ABBR = re.compile(r"^(.*[A-Za-z].*?)\s*\(\s*([A-Za-z0-9\[\]+\-−,']{1,14})\s*\)\s*$")


def clean_name(name):
    """-> (name, abbreviation the cell defines or ""). Never changes the chemistry.

    Drops a trailing footnote marker, unbalanced brackets left by splitting
    "(ChCl:EG)", and a trailing "(ABBR)" -- returned separately, because "Choline
    Chloride (ChCl)" is a definition the paper is giving us, not part of the name.
    """
    text = str(name or "").strip()
    text = _TRAILING_MARKER.sub("", text).strip()
    for opener, closer in (("(", ")"), ("[", "]")):
        while text.endswith(closer) and text.count(closer) > text.count(opener):
            text = text[:-1].strip()
        while text.startswith(opener) and text.count(opener) > text.count(closer):
            text = text[1:].strip()
    abbr = ""
    match = _NAME_WITH_ABBR.match(text)
    if match and len(match.group(2)) < len(match.group(1)) \
            and re.search(r"[A-Za-z]", match.group(2)) \
            and not re.fullmatch(r"[a-z]", match.group(2)):
        text, abbr = match.group(1).strip(), match.group(2).strip()
    return text, abbr


def split_coefficient(name):
    """'2PTSA' -> ('PTSA', 2). -> (name, None) when there is none."""
    match = _COEFFICIENT.match(str(name or "").strip())
    if not match:
        return name, None
    return match.group(2).strip(), int(match.group(1))


# ---------- the parser ----------
def _strip_additives(text, comp):
    """Pull "+ 20% water" / "+ 10 wt% EC" out of the text. -> what is left."""
    work = text
    match = _WATER_BEFORE.search(work)
    if match:
        comp.water_fraction = float(match.group(2)) / 100
        comp.water_basis = basis_of(match.group(3))
        work = (work[:match.start()] + work[match.end():]).strip(" +,;")

    def repl(match):
        value, unit, name = match.group(1), match.group(2), (match.group(3) or "").strip()
        basis = basis_of(unit)
        if name and not is_water(name) and not re.match(r"^water\b", name, re.I):
            comp.additives[name] = (float(value) / 100, basis)
            return " "
        if name or re.search(r"water|h2o", text, re.I):
            comp.water_fraction = float(value) / 100
            comp.water_basis = basis
            return " "
        return match.group(0)            # a bare "20%" with nothing named: not ours

    work = _ADDITIVE_AFTER.sub(repl, work)
    return re.sub(r"\s+", " ", work).strip(" +,;")


def parse_mixture(text, dictionary=None, resolve=None):
    """One string naming a DES -> Composition. Never raises; notes what it could not do.

    `dictionary` is the paper's own DES dictionary (paper_dictionary.PaperDictionary):
    a code like "CEM5" or "DES 1:4" means what the paper says it means and nothing else.
    `resolve` maps a written component name onto a canonical one (abbreviation -> full
    name); a name it cannot resolve is kept as written, never guessed at.
    """
    raw = re.sub(r"\s+", " ", str(text or "")).strip()
    comp = Composition(written=raw)
    if not raw or raw in config.DASH:
        return comp

    work = _strip_additives(raw, comp)

    # Several ratios of one DES: recorded as written, not resolved to one.
    listed = _RATIO_LIST.search(work)
    if listed:
        comp.ratio_raw = listed.group(1).strip()
        comp.note("several ratios listed; not resolved to one")
        work = (work[:listed.start()] + work[listed.end():]).strip(" ,;")
    else:
        tail = _RATIO_TAIL.search(work)
        if tail and tail.start() > 0:
            comp.ratio_raw = re.sub(r"\s+", "", tail.group(1))
            work = work[:tail.start()].strip(" ,;(")

    # A code the paper itself defines wins over any parse of the letters.
    if dictionary is not None:
        entry = dictionary.lookup(work) or dictionary.lookup(raw)
        if entry is not None:
            out = copy.deepcopy(entry)
            out.written = raw
            if comp.ratio_raw and not out.ratio_raw:
                out.ratio_raw = comp.ratio_raw
                out.parts = _ratio_parts(comp.ratio_raw)
                out.ratio_source = "as_written"
            out.water_fraction = comp.water_fraction if comp.water_fraction is not None \
                else out.water_fraction
            out.water_basis = comp.water_basis or out.water_basis
            out.additives = {**out.additives, **comp.additives}
            finish(out)
            return out

    names, coefficients = [], []
    for piece in split_names(work):
        piece, abbr = clean_name(piece)
        name, coefficient = split_coefficient(piece)
        name = name.strip()
        if abbr and name:
            comp.aliases[abbr] = name
        if name and not re.search(r"[A-Za-z]", name):
            comp.note(f"{name!r} is a number, not a component")
            continue
        names.append(name)
        coefficients.append(coefficient)
    names = [n for n in names if n]
    comp.components_written = list(names)
    comp.components = [_resolved(n, resolve) for n in names]

    if comp.ratio_raw and not listed:
        comp.parts = _ratio_parts(comp.ratio_raw)
        comp.ratio_source = "as_written"
    elif any(coefficients):
        comp.parts = [float(c or 1) for c in coefficients]
        comp.ratio_raw = ":".join(str(c or 1) for c in coefficients)
        comp.ratio_source = "coefficients"
    finish(comp)
    return comp


def _resolved(name, resolve):
    if is_water(name):
        return "Water"
    if resolve is None:
        return name
    found = resolve(name)
    return found or name


def ratio_from_cell(text, header=""):
    """A ratio column's cell -> (ratio_raw, source). Handles the bare-number shorthand.

    "2" under "HBA:HBD Molar Ratio" means 2:1. That inference is labelled `inferred`,
    because the paper wrote one number and we wrote two.
    """
    raw = re.sub(r"\s+", "", str(text or ""))
    if not raw or raw in config.DASH:
        return "", ""
    if re.fullmatch(_RATIO, raw):
        return raw, "as_written"
    number = xml_utils.clean_number(raw)
    if number is not None and (_HBA_HBD.search(header) or _HBD_HBA.search(header)
                               or _RATIO_HEADER.search(header)):
        if _HBD_HBA.search(header):
            return f"1:{number:g}", "inferred"
        return f"{number:g}:1", "inferred"
    return raw, "as_written"


def compose(cells, ratio_text="", ratio_header="", water_text="", water_header="",
            dictionary=None, resolve=None, implied=()):
    """Several table cells describing one DES -> one Composition.

    `cells` are the component cells (each may hold one name, several, a code, or a
    whole formula with its ratio). The ratio cell and a water column, when the table
    has them, are folded in. `implied` are components the caption states and no
    column lists ("PTSA based DES").
    """
    out = Composition(written=" | ".join(c for c in cells if c))
    from_code = None
    for cell in cells:
        if not str(cell or "").strip() or str(cell).strip() in config.DASH:
            continue
        part = parse_mixture(cell, dictionary, resolve)
        if part.code and from_code is None:
            from_code = part
        out.components += part.components
        out.components_written += part.components_written or part.components
        if part.ratio_raw and not out.ratio_raw:
            out.ratio_raw, out.parts, out.ratio_source = (part.ratio_raw, part.parts,
                                                          part.ratio_source)
        if part.water_fraction is not None:
            out.water_fraction, out.water_basis = part.water_fraction, part.water_basis
        out.additives.update(part.additives)
        out.aliases.update(part.aliases)
        out.notes += [n for n in part.notes if n not in out.notes]
    if from_code is not None and len(cells) == 1:
        out.code, out.source = from_code.code, from_code.source

    present = {re.sub(r"\W+", "", c).lower() for c in out.components}
    for name in implied or ():
        if re.sub(r"\W+", "", str(name)).lower() not in present:
            out.components.append(_resolved(name, resolve))
            out.components_written.append(name)
            out.note(f"{name} implied by the caption")

    if ratio_text:
        cell = Composition()
        rest = _strip_additives(str(ratio_text), cell)
        if cell.water_fraction is not None:
            out.water_fraction, out.water_basis = cell.water_fraction, cell.water_basis
        out.additives.update(cell.additives)
        ratio, source = ratio_from_cell(rest, ratio_header)
        if ratio and re.fullmatch(_RATIO, ratio):
            out.ratio_raw, out.parts, out.ratio_source = ratio, _ratio_parts(ratio), source
        elif ratio and re.fullmatch(r"(\d+(?:\.\d+)?:)+\(?\d+(?:\.\d+)?[–—-]\d+(?:\.\d+)?\)?",
                                    ratio):
            # "1:4:(1–11)": the last part is a RANGE of variants, each its own DES
            # ("CEM1".."CEM11"). Kept as written; the dictionary reads a code's suffix.
            out.ratio_raw = ratio
            fixed = ratio.rsplit(":", 1)[0]
            out.parts = _ratio_parts(fixed) + [None]
            out.note("the last ratio part is a range of variants")
        elif ratio:
            out.ratio_raw = ratio
            out.note(f"ratio cell not a molar ratio: {ratio!r}")

    if water_text and str(water_text).strip() not in config.DASH:
        number = xml_utils.clean_number(re.sub(r"[%\s]|wt|w/w|v/v", "", str(water_text)))
        if number is not None:
            # A water column holds a percentage unless its numbers say otherwise.
            out.water_fraction = number / 100 if number > 1 else number
            out.water_basis = basis_of(f"{water_header} {water_text}")
    finish(out)
    return out


def finish(comp):
    """Add water as a component and compute the molar ratio when it can be. In place."""
    if comp.water_fraction is not None and not any(is_water(c) for c in comp.components):
        comp.components.append("Water")
        comp.components_written.append("water")
    for name in comp.additives:
        if name not in comp.components:
            comp.components.append(name)
            comp.components_written.append(name)

    if comp.ratio_raw and comp.parts and not comp.ratio_source:
        comp.ratio_source = "as_written"
    comp.basis = comp.basis or ("molar" if comp.parts else "")

    # Molar parts over the non-water, non-additive components, as written.
    base = [c for c in comp.components if not is_water(c) and c not in comp.additives]
    parts = list(comp.parts[:len(base)]) if comp.parts else []
    if len(parts) != len(base) or any(p is None for p in parts):
        if comp.parts and len(comp.parts) != len(base):
            comp.note(f"{len(comp.parts)} ratio parts for {len(base)} components")
        comp.ratio_molar = ""
        return comp

    extra = {}
    if comp.water_fraction is not None:
        extra["Water"] = (comp.water_fraction, comp.water_basis)
    extra.update(comp.additives)
    if not extra:
        comp.ratio_molar = ":".join(f"{p:g}" for p in parts)
        comp.parts = parts
        return comp

    # Weight fractions of water or an additive -> moles, from molecular weights. Only
    # a weight basis converts; a volume % would need densities we do not have.
    weights = [molecular_weight(c) for c in base]
    if any(w is None for w in weights):
        missing = [c for c, w in zip(base, weights) if w is None]
        comp.note(f"no molecular weight for {', '.join(missing)}; molar ratio not computed")
        return comp
    unit_mass = sum(p * w for p, w in zip(parts, weights))
    molar = list(parts)
    order = list(base)
    for name, (fraction, basis) in extra.items():
        if basis != "wt" or not 0 < fraction < 1:
            comp.note(f"{name} given on a {basis or 'unknown'} basis; molar ratio not computed")
            return comp
        mw = molecular_weight(name)
        if mw is None:
            comp.note(f"no molecular weight for {name}; molar ratio not computed")
            return comp
        # fraction of the TOTAL mass: m_x = f/(1-f) * (mass of everything else)
        molar.append(fraction / (1 - fraction) * unit_mass / mw)
        order.append(name)
    comp.components = order
    comp.parts = [round(m, 3) for m in molar]
    comp.ratio_molar = ":".join(_fmt(m) for m in molar)
    comp.ratio_source = "computed"
    comp.note("molar ratio computed from weight fractions with PubChem molecular weights")
    return comp


# ---------- is this a DES at all? ----------
_ION_ONLY = re.compile(r"^\s*(?:\[[^\]]+\]\s*)+\s*[A-Za-z0-9\[\]\-−+]*\s*$")
_IL_WORDS = re.compile(r"imidazolium|pyridinium|pyrrolidinium|\bil\b|ionic liquid", re.I)


def is_ionic_liquid_only(comp):
    """One salt written as an ionic liquid, with no HBD: '[C2C1im]Cl', '[BMIM][Br]'."""
    base = [c for c in comp.components if not is_water(c)]
    return len(base) == 1 and bool(_ION_ONLY.match(base[0]) or _IL_WORDS.search(base[0]))


def has_ionic_liquid(comp):
    """Does any component look like an ionic liquid by its notation or name?"""
    return any(_ION_ONLY.match(c) or _IL_WORDS.search(c)
               for c in comp.components if c and not is_water(c))


def des_problem(comp, dictionary=None):
    """Why this is not a DES. -> "" when it is one.

    A deep eutectic solvent is at least two constituents (an HBA and an HBD; water
    alone does not make a second). A row that names one compound -- a corrosion
    inhibitor, a group-contribution fragment like "-CH3", a lone ionic liquid -- is
    not one, whatever table it sits in. A code the paper's own dictionary defines is
    always accepted: the paper said what it is.
    """
    names = [c for c in comp.components if c]
    defined = bool(comp.code) or (dictionary is not None
                                  and dictionary.lookup(comp.written) is not None)
    if defined and len(names) >= 2:
        return ""                          # the paper says what it is
    if not names:
        return "no components"
    if any(_FRAGMENT.match(c) for c in names):
        return "a structural fragment, not a compound"
    if is_ionic_liquid_only(comp):
        return "a single ionic liquid, not a DES"
    # Water counts as a constituent when it is listed as one, with its own molar share
    # ("salt:water 1:4", a recognised class of DES) -- not when it is a few percent
    # added to an otherwise single compound.
    constituents = comp.n_non_water + (1 if any(is_water(c) for c in names)
                                       and comp.water_fraction is None else 0)
    if constituents < 2:
        return "a single compound, not a DES"
    return ""
