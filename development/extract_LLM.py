"""
Test extraction from XML file with LLM + pydantic

"""
import os, glob
import pandas as pd
from lxml import etree
from pydantic import BaseModel, ValidationError
from typing import Optional, Literal
from ollama import chat
import json
from pathlib import Path

RAW_DIR = Path("raw_responses")
RAW_DIR.mkdir(exist_ok=True)

MODEL = "qwen3"


class DESMeasurement(BaseModel): # Inhereted from pydantic base model 
    hba: str
    hbd: str
    molar_ratio: Optional[str] = None # If doesn't exist, set to None
    property: Literal["Density", "Viscosity", "Conductivity",
                      "Surface_tension", "Refractive_index", "Melting_point"] # The property must be exactly one of these listed strings
    value: float
    unit: Optional[str] = None
    temperature_C: Optional[float] = None # If doesn't exist, set to None 
    source_text: str

class Extraction(BaseModel):          # Container so we can return a list, inhereted from pydantic base model
    measurements: list[DESMeasurement]


# Convert XML to plain text 
def xml_to_text(path: str) -> str:
    root = etree.parse(path).getroot()
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    for el in root.iter():
        if el.tag in {"bibliography", "references", "ce:bibliography", "bibliogr"}: # Remove reference info
            parent = el.getparent()
            if parent is not None:
                parent.remove(el)
    return " ".join(" ".join(root.itertext()).split())


# Ask local LLM 
PROMPT = (
    "You extract deep eutectic solvent (DES) property measurements from chemistry papers.\n"
    "A valid measurement has TWO chemical component names (an HBA and an HBD, e.g. "
    "'choline chloride' and 'urea') and a numeric physical property.\n"
    "IGNORE author names, citation years, page numbers, and reference lists — these are NOT measurements.\n"
    "If the text contains no genuine DES property data, return an empty list.\n"
    "Only record values explicitly stated; never infer. Copy the exact source text into source_text.\n"
    "Respond as JSON only.\n\nTEXT:\n{text}"
)

def extract(text: str, source_name: str) -> list[DESMeasurement]:
    resp = chat(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT.format(text=text[:12000])}],
        format=Extraction.model_json_schema(),   # uses model_json_schema from pydantic to constrain output to JSON with schema defined above
        options={"temperature": 0},              # temperature = randomness, set to 0 to make model deterministic
    )
    raw = resp.message.content

    # Save JSON
    (RAW_DIR / f"{source_name}.json").write_text(raw, encoding="utf-8")

    try:
        #print(resp.message.content)
        return Extraction.model_validate_json(resp.message.content).measurements # resp.message.content is model's raw answer as JSON, use model_validate_json to validate against schema defined above
    except ValidationError as e:
        print("  couldn't validate model output:", e.errors()[0]["msg"]) # print error message if does not match schema
        return []


# Verification 
def verified(r: DESMeasurement) -> bool:
    return str(r.value) in r.source_text # double check that all returned values actually appear somewhwere in the text and the LLM didn't hallucinate them 


# Loop over all xml files (just one for now)
rows = []
for path in glob.glob("xml/*.xml"): # loop over all xml files in xml directory
    name = Path(path).stem 
    print("processing", os.path.basename(path)) # print file name 
    for r in extract(xml_to_text(path), name):
        if verified(r):
            rows.append({**r.model_dump(), "file": os.path.basename(path)})

pd.DataFrame(rows).to_csv("extracted.csv", index=False)
print(f"\nDone. {len(rows)} verified measurements -> extracted.csv")