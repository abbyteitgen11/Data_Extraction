"""
Match DES components against QM9 and attach its computed descriptors.

QM9 is 134k small organic molecules with DFT properties (B3LYP/6-31G(2df,p)) computed
for ONE ISOLATED MOLECULE IN THE GAS PHASE. That is worth stating twice, because
everything else in this pipeline is a measurement of a real substance:

    a melting point on a Component  is something somebody measured
    a QM9 dipole on a Component     is something a computer calculated about a
                                    single molecule that is not in a solvent at all

So these never become property nodes. They sit on the Component alongside `tpsa`,
`xlogp` and `complexity` -- which are also computed descriptors -- under a `qm9_`
prefix, with the unit in the field name and a (:Source {name:'QM9'}) behind them. They
are ML features, not evidence.

What can possibly match: QM9 covers only C/H/N/O/F and at most 9 heavy atoms, so no
HBA in this corpus will ever be in it -- every quaternary ammonium salt has a chloride
or bromide. It is the HBD side (glycols, small acids, amides, polyols) that matches.

    python run_pipeline.py --steps qm9
"""
import json

from . import config

# The 19 values in Data.y, in order, as torch_geometric returns them. PyG converts the
# energies from Hartree to eV before handing them over, so these units are NOT the ones
# in the raw QM9 files -- hence the unit in every field name.
TARGETS = (
    ("qm9_dipole_D", 0),                 # mu, Debye
    ("qm9_polarizability_a0_3", 1),      # alpha, Bohr^3
    ("qm9_homo_eV", 2),
    ("qm9_lumo_eV", 3),
    ("qm9_gap_eV", 4),
    ("qm9_r2_a0_2", 5),                  # electronic spatial extent, Bohr^2
    ("qm9_zpve_eV", 6),
    ("qm9_cv_cal_mol_K", 11),            # heat capacity at 298.15 K
)
_U0_INDEX = 7                            # internal energy at 0 K, used to break ties


def inchikey_from_smiles(smiles):
    """The FULL InChIKey for a SMILES string, stereochemistry included. -> str | None.

    Used for the QM9 side only. torch_geometric writes its SMILES with
    `isomericSmiles=True`, so QM9's stereochemistry survives into the key.

    Our side does NOT go through here: `enrich_components` stores PubChem's
    *connectivity* SMILES, which has the stereo stripped out, so regenerating a key
    from it loses the very thing that distinguishes glucose from galactose -- 66 of our
    403 components come back as `-UHFFFAOYSA-` if you try. The stored `inchikey` is
    already PubChem's full stereo-aware key and is what `component_key` returns; it
    round-trips against the stored InChI for 403/403 components.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def component_key(row):
    """The full InChIKey PubChem gave for this component. -> str | None."""
    key = str(row.get("inchikey") or "").strip()
    return key or None


def skeleton(key):
    """The connectivity block of an InChIKey -- same atoms and bonds, any stereo."""
    return str(key or "").split("-")[0]


def build_index(refresh=False):
    """QM9 -> {full InChIKey: [record, ...]}, cached in data/qm9_index.json.

    torch_geometric downloads and processes on first use; with rdkit installed it
    parses the raw SDF, so every molecule carries its own SMILES, and it already drops
    the ~3k molecules QM9 itself flags as uncharacterized.
    """
    if config.QM9_INDEX.exists() and not refresh:
        return json.loads(config.QM9_INDEX.read_text())

    from torch_geometric.datasets import QM9

    print("  loading QM9 (first run downloads and processes ~130k molecules)")
    dataset = QM9(root=str(config.QM9_DIR))
    print(f"  {len(dataset)} molecules")

    index, skipped = {}, 0
    for data in dataset:
        key = inchikey_from_smiles(data.smiles)
        if key is None:
            skipped += 1
            continue
        y = data.y.view(-1).tolist()
        record = {"qm9_id": str(data.name), "smiles": data.smiles,
                  "u0": y[_U0_INDEX]}
        record.update({field: round(y[i], 6) for field, i in TARGETS})
        index.setdefault(key, []).append(record)

    config.QM9_INDEX.parent.mkdir(parents=True, exist_ok=True)
    config.QM9_INDEX.write_text(json.dumps(index))
    print(f"  indexed {len(index)} distinct substances"
          f"{f', {skipped} unparseable' if skipped else ''} -> {config.QM9_INDEX.name}")
    return index


def _norm(text):
    """Fold case, spacing and punctuation so 'H2O' and 'h2o' compare equal."""
    import re

    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def contested_components(rows):
    """Component names whose PubChem identification is in doubt. -> {name: [others]}.

    Several names sharing one InChIKey is not by itself a problem -- `Water`, `water`
    and `H2O` are one chemical written three ways, and skipping them would lose real
    matches for nothing. The problem is the other kind: `PEG600`, `PEG2000`, `PEG4000`
    and `PEG6000` also share a key, because PubChem resolved every one of them to
    diethylene glycol. Attaching that molecule's quantum data to all four would carry a
    bad identification into data that looks freshly computed.

    PubChem's own synonym list separates them on evidence rather than on a guess:
    "H2O" IS a synonym of water, "PEG600" is NOT a synonym of diethylene glycol. A name
    the compound does not acknowledge is a resolution we should not trust.

    A group whose members carry no synonym list is treated as contested, because the
    check could not be made -- absence of evidence is not evidence here.
    """
    by_key = {}
    for r in rows:
        key = component_key(r)
        if key and r.get("lookup_status") == "ok":
            by_key.setdefault(key, []).append(r)

    contested = {}
    for group in by_key.values():
        if len(group) < 2:
            continue
        known = set()
        for r in group:
            known.update(_norm(s) for s in str(r.get("synonyms") or "").split(";") if s)
            known.add(_norm(r.get("matched_name")))
            # A paper often writes the formula instead of a name -- "H2O" for water.
            # PubChem lists it as a synonym too, but hundreds deep, past the slice we
            # store, so check the formula we already hold rather than storing more.
            known.add(_norm(r.get("molecular_formula")))
        known.discard("")
        names = [r["name"] for r in group]
        unvouched = [n for n in names if _norm(n) not in known]
        if unvouched:
            for name in names:
                contested[name] = [n for n in names if n != name]
    return contested


def match(rows, index=None):
    """Attach QM9 descriptors to the components that unambiguously match. -> stats."""
    index = index if index is not None else build_index()
    contested = contested_components(rows)

    by_skeleton = {}
    for k in index:
        by_skeleton.setdefault(skeleton(k), []).append(k)

    matched = ambiguous = near = 0
    for row in rows:
        # Clear before deciding, so the step is idempotent. components.csv persists the
        # previous run's answer, and without this a component that matched under an
        # older rule kept its QM9 data even once it became contested -- ending up
        # flagged as doubtful AND carrying the numbers the flag exists to withhold.
        row["qm9_contested"] = ""
        row["qm9_id"] = ""
        row["qm9_n_matches"] = None
        for field, _ in TARGETS:
            row[field] = None

        if row.get("lookup_status") != "ok" or not row.get("inchikey"):
            continue
        if row["name"] in contested:
            row["qm9_contested"] = ";".join(sorted(contested[row["name"]]))[:200]
            continue
        key = component_key(row)
        found = index.get(key or "") or []
        if not found:
            # QM9 may hold a STEREOISOMER of this molecule -- same atoms and bonds,
            # different arrangement. That is a different substance (glucose is not
            # galactose), so nothing is attached; it is counted so the near-miss is
            # visible rather than looking like plain absence.
            near += bool(key and by_skeleton.get(skeleton(key)))
            continue
        # Several QM9 entries can share a skeleton (tautomers, stereoisomers). Take the
        # lowest-energy one and record how many there were, so an ambiguous match is
        # visible rather than silently collapsed to whichever came first.
        best = min(found, key=lambda r: r["u0"])
        row["qm9_id"] = best["qm9_id"]
        row["qm9_n_matches"] = len(found)
        for field, _ in TARGETS:
            row[field] = best[field]
        matched += 1
        ambiguous += len(found) > 1

    return {"matched": matched, "ambiguous": ambiguous, "stereo_near_miss": near,
            "contested": len({n for n in contested if any(
                r["name"] == n for r in rows)}),
            "eligible": sum(1 for r in rows if r.get("lookup_status") == "ok")}


def run(refresh=False):
    """The qm9 step: match every enriched component and rewrite components.csv."""
    import pandas as pd

    from . import xml_utils
    from .schema import ComponentRow

    if not config.COMPONENTS_CSV.exists():
        print("  no components.csv yet -- run --steps components first")
        return []

    df = pd.read_csv(config.COMPONENTS_CSV)
    rows = df.astype(object).where(pd.notna(df), None).to_dict("records")
    stats = match(rows, build_index(refresh=refresh))

    # An empty CSV cell reads back as None, which pydantic rejects for a `str` field.
    model = ComponentRow.model_fields
    text_fields = {k for k, f in model.items() if f.annotation is str}
    out = []
    for r in rows:
        clean = {k: v for k, v in r.items() if k in model}
        for k in text_fields:
            if clean.get(k) is None:
                clean[k] = ""
        out.append(ComponentRow(**clean))
    xml_utils.write_csv(out, config.COMPONENTS_CSV, model=ComponentRow)

    print(f"  QM9: {stats['matched']} of {stats['eligible']} resolved components matched"
          f" ({stats['ambiguous']} matched more than one QM9 entry)")
    print(f"       {stats['contested']} skipped -- their InChIKey is shared by a name the "
          f"compound does not acknowledge, so the identification is in doubt")
    print(f"       {stats['stereo_near_miss']} had a QM9 STEREOISOMER but not the "
          f"substance itself -- not attached, since glucose is not galactose")
    return out
