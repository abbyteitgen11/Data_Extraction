"""
Parse Table 2 (~1500-row DES property table) from the Omar & Sadeghi (2023) in XML format, 
convert to dataframe for creating graph database

"""
import re
import pandas as pd
from lxml import etree

# From footnote 
# "At a40 C, b20 C, c60 C, d45 C, e30 C, f35 C, g50 C, h55 C" ; default is 25 C
TEMP_MAP = {"a": 40, "b": 20, "c": 60, "d": 45, "e": 30, "f": 35, "g": 50, "h": 55}
DEFAULT_TEMP = 25

# Units
PROP_COLUMNS = [
    # (entry index, property name, unit)
    (3, "Melting point",    "C"),
    (4, "Density",          "g*cm^-3"),
    (5, "Viscosity",        "mPa*s"),
    (6, "Conductivity",     "mS*cm^-1"),
    (7, "Surface_tension",  "mN*m^-1"),
    (8, "Refractive_index", ""),
]

DASH = {"–", "—", "-", "−", ""}


def cell_text(entry):
    return re.sub(r"\s+", " ", "".join(entry.itertext())).strip()


def cell_sups(entry):
    return [(s.text or "").strip() for s in entry.iter() if s.tag == "sup"]


def clean_number(s):
    """Return float or None. Handles unicode minus, thousands commas, dashes."""
    s = s.strip().replace(",", "")
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    if s in {"", "-"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_property(entry):
    """Return (value, temperature_C). value None if not reported/unparseable."""
    text = cell_text(entry)
    if text in DASH:
        return None, None
    marker = next((s for s in cell_sups(entry) if s in TEMP_MAP), None)
    temp = TEMP_MAP[marker] if marker else DEFAULT_TEMP
    # strip a leading marker letter (e.g. 'a1.24' -> '1.24')
    if marker and text[:1] == marker:
        text = text[1:]
    val = clean_number(text)
    if val is None:
        return None, None            # e.g. "DT", "140 84.5" -> not a clean value
    return val, temp


def parse_ratio(entry):
    """Return (raw_ratio, [r1, r2, r3], flag)."""
    raw = cell_text(entry)
    sups = cell_sups(entry)
    flag = ""
    if "i" in sups:
        flag = "weight_ratio"
    if "us" in sups or raw.startswith("us"):
        flag = "unstable"
    if raw.startswith("C") or "C" in sups:
        flag = "unknown_ratio"
    # strip known leading letter-markers so "i6:4" -> "6:4", "us2:1" -> "2:1"
    r = re.sub(r"^(i|us|C-?)", "", raw)
    parts = [p.strip() for p in r.split(":")] if r else []
    nums = []
    for p in parts[:3]:
        try:
            nums.append(float(p))
        except ValueError:
            nums.append(None)
    while len(nums) < 3:
        nums.append(None)
    return raw, nums, flag


def parse_components(hba, hbd):
    """Return list of up to 3 component names + a 'complex' flag."""
    comps = [hba.strip()] + [c.strip() for c in hbd.split("/")]
    flag = "quaternary+" if len(comps) > 3 else ""
    comps = (comps + [None, None, None])[:3]
    return comps, flag


def parse_table(xml_path):
    tree = etree.parse(xml_path)
    root = tree.getroot()
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]

    table = root.findall(".//table")[1]          # Table 2
    rows = table.findall(".//row")

    records = []
    for row in rows:
        entries = row.findall("entry")
        if len(entries) != 10:
            continue                              # skip unit sub-header (5 cells)
        hba = cell_text(entries[0])
        hbd = cell_text(entries[1])
        if hba in {"", "HBAs"} or cell_text(entries[2]) == "Molar ratio":
            continue                              # skip repeated header rows

        (c1, c2, c3), comp_flag = parse_components(hba, hbd)
        raw_ratio, (r1, r2, r3), ratio_flag = parse_ratio(entries[2])
        ref = cell_text(entries[9]).strip("[]").replace("–", "-")

        rec = {
            "Component_1": c1, "Component_1_SMILES": None,
            "Component_2": c2, "Component_2_SMILES": None,
            "Component_3": c3, "Component_3_SMILES": None,
            "Ratio_component_1": r1, "Ratio_component_2": r2, "Ratio_component_3": r3,
            "Ratio_raw": raw_ratio,
            "Ratio_flag": ratio_flag, "Component_flag": comp_flag,
            "DOI": "10.1016/j.molliq.2023.121899", "Ref": ref,
        }
        # descriptive, reasonably-unique mixture name (replaces arbitrary "Mixture N")
        names = ":".join(n for n in (c1, c2, c3) if n)
        rec["Mixture"] = f"{names} ({raw_ratio})"

        for idx, prop, unit in PROP_COLUMNS:
            val, temp = parse_property(entries[idx])
            rec[prop] = val
            rec[f"Units_{prop.lower()}"] = unit if val is not None else None
            rec[f"Temperature_{prop.lower()}"] = temp
        records.append(rec)

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = parse_table("SadeghiDESReview.xml")

    # ---- column order: identity first, then value/unit/temp triples ----
    id_cols = ["Component_1", "Component_1_SMILES", "Component_2", "Component_2_SMILES",
               "Component_3", "Component_3_SMILES",
               "Ratio_component_1", "Ratio_component_2", "Ratio_component_3", "Ratio_raw",
               "Mixture", "DOI", "Ref", "Ratio_flag", "Component_flag"]
    prop_cols = []
    for _, prop, _ in PROP_COLUMNS:
        p = prop.lower()
        prop_cols += [prop, f"Units_{p}", f"Temperature_{p}"]
    df = df[id_cols + prop_cols]

    df.to_csv("table2_full.csv", index=False)

    # ---- report ----
    print("Rows parsed:", len(df))
    print("Binary / ternary / quaternary+:",
          (df.Component_3.isna()).sum(),
          (df.Component_3.notna() & (df.Component_flag == "")).sum(),
          (df.Component_flag == "quaternary+").sum())
    print("\nValues reported per property:")
    for _, prop, _ in PROP_COLUMNS:
        n = df[prop].notna().sum()
        print(f"  {prop:16s} {n:5d}")
    print("\nMeasurements not at 25 C (per property):")
    for _, prop, _ in PROP_COLUMNS:
        p = prop.lower()
        off = ((df[prop].notna()) & (df[f"Temperature_{p}"] != 25)).sum()
        print(f"  {prop:16s} {off:5d}")

    # ---- spot-check the known ChCl:Urea 1:2 row (reline) ----
    print("\nSpot check  ChCl:Urea (1:2):")
    r = df[(df.Component_1 == "Choline chloride") & (df.Component_2 == "Urea")
           & (df.Ratio_raw == "1:2")].iloc[0]
    print(f"  density={r.Density} {r.Units_density} @ {r.Temperature_density}C ; "
          f"viscosity={r.Viscosity} @ {r.Temperature_viscosity}C ; Tm={r['Melting point']}")

    print("\nSample rows:")
    print(df[["Component_1", "Component_2", "Component_3", "Ratio_raw",
              "Density", "Temperature_density", "Viscosity", "Ref"]].head(8).to_string(index=False))

