"""
The table route: parse Table 2 of the review into one row per DES mixture.

Table 2 has 10 cells per row:

    HBA | HBD | molar ratio | Tm | rho | eta | kappa | gamma | n | refs

The HBD cell may list several components separated by "/", giving ternary and
occasionally quaternary mixtures. Values carry superscript footnote markers that
encode the measurement temperature (see config.TEMP_MAP); everything unmarked was
measured at 25 C.

Output is deliberately *wide* — one row per mixture, with a value/unit/temperature
triple per property — because that is how the table reads. ``to_long`` derives the
one-row-per-measurement view that the graph loader and any ML training will want.
"""
import re

from . import config, xml_utils
from .extract_references import reference_fields
from .schema import MeasurementRow, MixtureRow, TableRow


def find_table(tables, label="Table 2"):
    """Pick a table by its printed label, falling back to position."""
    for t in tables:
        if xml_utils.text(t.find("label")).strip().lower() == label.lower():
            return t
    index = 1 if label.endswith("2") else 0
    return tables[index] if len(tables) > index else None


def parse_ratio(entry):
    """Read the molar-ratio cell -> (raw text, [r1, r2, r3], flag)."""
    raw = xml_utils.text(entry)
    markers = xml_utils.sups(entry)
    flag = ""
    if "i" in markers:
        flag = "weight_ratio"
    if "us" in markers or raw.startswith("us"):
        flag = "unstable"
    if raw.startswith("C") or "C" in markers:
        flag = "unknown_ratio"

    # Strip the leading marker so "i6:4" -> "6:4", "us2:1" -> "2:1".
    stripped = re.sub(r"^(i|us|C-?)", "", raw)
    parts = [p.strip() for p in stripped.split(":")] if stripped else []
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
    """-> ([up to 3 names], flag). Mixtures beyond ternary are flagged and truncated."""
    comps = [hba.strip()] + [c.strip() for c in hbd.split("/")]
    flag = "quaternary+" if len(comps) > 3 else ""
    return (comps + [None, None, None])[:3], flag


def _sources(ref_numbers, reference_map):
    """Collect aligned per-source metadata for the papers a row cites.

    Only references that resolved to a DOI are included, so every value in
    Source_DOIs matches a Paper node in the graph. Source_ref_numbers keeps the
    full list either way, so nothing is lost.
    """
    keys = ("doi", "authors", "title", "journal", "volume", "issue", "pages", "year")
    collected = {k: [] for k in keys}
    for n in ref_numbers:
        meta = reference_map.get(n)
        if not meta or not meta.get("doi"):
            continue
        fields = reference_fields(meta)
        for k in keys:
            collected[k].append(str(fields[k]).replace(config.SOURCE_SEP, "/"))
    sep = config.SOURCE_SEP
    return {k: sep.join(v) for k, v in collected.items()}


def parse_table2(table, reference_map, review):
    """-> list[MixtureRow]. Header and unit sub-header rows are skipped."""
    review_fields = {
        "Review_DOI": review.get("doi", config.REVIEW_DOI),
        "Review_authors": review.get("authors", ""),
        "Review_title": review.get("title", ""),
        "Review_journal": review.get("journal", ""),
        "Review_volume": str(review.get("volume", "")),
        "Review_issue": str(review.get("issue", "")),
        "Review_year": str(review.get("year", "")),
    }

    rows = []
    for i, row in enumerate(table.findall(".//row")):
        entries = row.findall("entry")
        if len(entries) != 10:
            continue                       # the unit sub-header has only 5 cells
        hba = xml_utils.text(entries[0])
        if hba in {"", "HBAs"} or xml_utils.text(entries[2]) == "Molar ratio":
            continue                       # repeated header rows

        (c1, c2, c3), component_flag = parse_components(hba, xml_utils.text(entries[1]))
        ratio_raw, (r1, r2, r3), ratio_flag = parse_ratio(entries[2])
        ref_text = xml_utils.text(entries[9]).strip("[]").replace("–", "-")
        ref_numbers = xml_utils.expand_ref_field(ref_text)
        sources = _sources(ref_numbers, reference_map)

        names = ":".join(n for n in (c1, c2, c3) if n)
        record = {
            "Row_id": f"T2-{len(rows) + 1:04d}",
            "Component_1": c1, "Component_2": c2, "Component_3": c3,
            "Ratio_component_1": r1, "Ratio_component_2": r2, "Ratio_component_3": r3,
            "Ratio_raw": ratio_raw,
            "Mixture": f"{names} ({ratio_raw})",
            "Ratio_flag": ratio_flag,
            "Component_flag": component_flag,
            "DOI": review_fields["Review_DOI"],
            "Ref": ref_text,
            **review_fields,
            "Source_ref_numbers": ",".join(str(n) for n in ref_numbers),
            "Source_DOIs": sources["doi"],
            "Source_authors": sources["authors"],
            "Source_titles": sources["title"],
            "Source_journals": sources["journal"],
            "Source_volumes": sources["volume"],
            "Source_issues": sources["issue"],
            "Source_pages": sources["pages"],
            "Source_years": sources["year"],
            "_locus": f"row {i}",
        }

        for index, prop, unit in config.PROPERTIES:
            value, temp = xml_utils.value_and_temperature(entries[index])
            suffix = prop.lower()
            record[prop] = value
            record[f"Units_{suffix}"] = unit if value is not None else None
            record[f"Temperature_{suffix}"] = temp

        locus = record.pop("_locus")
        mixture = MixtureRow(**record)
        rows.append((mixture, locus))

    return rows


def to_long(rows):
    """Flatten wide mixture rows into one MeasurementRow per reported value."""
    out = []
    for mixture, locus in rows:
        for prop in config.PROPERTY_NAMES:
            value = getattr(mixture, prop)
            if value is None:
                continue
            suffix = prop.lower()
            out.append(MeasurementRow(
                Measurement_key=f"{mixture.Row_id}:{prop}",
                Row_id=mixture.Row_id,
                Mixture=mixture.Mixture,
                Property=prop,
                Value=value,
                Unit=getattr(mixture, f"Units_{suffix}"),
                Temperature_C=getattr(mixture, f"Temperature_{suffix}"),
                Source="Table 2",
                Locus=locus,
                Source_ref_numbers=mixture.Source_ref_numbers,
                Source_DOIs=mixture.Source_DOIs,
                Review_DOI=mixture.Review_DOI,
            ))
    return out


def unhandled_tables(tables, handled):
    """Record any table we have no parser for, rather than dropping it silently."""
    rows = []
    for t in tables:
        if t is handled:
            continue
        rows.append(TableRow(
            table_id=t.get("id", ""),
            label=xml_utils.text(t.find("label")),
            caption=xml_utils.text(t.find("caption")),
            n_rows=len(t.findall(".//row")),
        ))
    return rows
