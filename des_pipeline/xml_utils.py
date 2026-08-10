"""
Low-level XML helpers. Every other module gets its lxml primitives from here, so
namespace-stripping, cell-text flattening and the footnote-temperature rule each
exist exactly once.
"""
import re

from lxml import etree

from . import config


def load_root(path=None):
    """Parse the paper and strip namespaces from every tag.

    Elsevier XML uses eight namespace prefixes (ce:, ja:, sb:, cals:, ...). Stripping
    them means the rest of the pipeline can write findall(".//table") instead of
    threading an nsmap through every call.

    Note this rewrites *tags* only — attributes keep their namespace, which is why
    the figure link href has to be matched with .endswith("href").
    """
    root = etree.parse(str(path or config.XML)).getroot()
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def text(el):
    """All text under an element, whitespace-normalised to a single line."""
    if el is None:
        return ""
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def sups(el):
    """Text of every <sup> under an element — Table 2's footnote markers."""
    return [(s.text or "").strip() for s in el.iter() if s.tag == "sup"]


def attr_endswith(el, suffix):
    """Find an attribute value by the tail of its name, ignoring any namespace."""
    return next((v for k, v in el.attrib.items() if k.endswith(suffix)), None)


def clean_number(s):
    """Return a float, or None. Handles unicode minus, thousands commas and dashes."""
    s = s.strip().replace(",", "")
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    if s in {"", "-"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def value_and_temperature(entry):
    """Read one Table 2 property cell -> (value, temperature_C).

    Returns (None, None) when the cell is a dash, a footnote-only marker, or
    otherwise not a single clean number (e.g. "DT", "140 84.5").
    """
    raw = text(entry)
    if raw in config.DASH:
        return None, None

    marker = next((s for s in sups(entry) if s in config.TEMP_MAP), None)
    temp = config.TEMP_MAP[marker] if marker else config.DEFAULT_TEMP

    # The marker letter sometimes lands in the text too: "a1.24" -> "1.24".
    if marker and raw[:1] == marker:
        raw = raw[1:]

    value = clean_number(raw)
    return (value, temp) if value is not None else (None, None)


def expand_ref_field(s):
    """'1,26-28' -> [1, 26, 27, 28]. Unparseable fragments are dropped."""
    out = []
    for part in str(s).replace("–", "-").replace("—", "-").split(","):
        part = part.strip()
        if "-" in part:
            try:
                lo, hi = (int(x) for x in part.split("-", 1))
                out += list(range(lo, hi + 1))
            except ValueError:
                pass
        elif part.isdigit():
            out.append(int(part))
    return out


def review_metadata(root):
    """Bibliographic metadata for the paper itself, from Elsevier's <coredata>.

    Used as the offline fallback. When the network is available the review is run
    through the same Crossref lookup as every other paper, because <coredata>
    carries no issue number.
    """
    core = root.find(".//coredata")
    if core is None:
        return {"doi": config.REVIEW_DOI}
    get = lambda tag: text(core.find(tag))          # noqa: E731 - reads better inline
    return {
        "doi": get("doi") or config.REVIEW_DOI,
        "title": get("title"),
        "authors": "; ".join(text(c) for c in core.findall("creator")),
        "journal": get("publicationName"),
        "volume": get("volume"),
        "issue": "",                                 # not present in <coredata>
        "pages": get("pageRange") or get("articleNumber"),
        "year": (get("coverDate") or "")[:4],
    }


def write_csv(records, path, model=None):
    """Write a list of pydantic models (or dicts) to CSV, preserving field order.

    `model` supplies the column names when `records` is empty, so a run that finds
    nothing still produces a readable file rather than a zero-byte one.
    """
    import pandas as pd

    rows = [r.model_dump() if hasattr(r, "model_dump") else r for r in records]
    columns = list(model.model_fields) if (not rows and model is not None) else None
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    print(f"  wrote {path.name}  ({len(rows)} rows)")
    return path
