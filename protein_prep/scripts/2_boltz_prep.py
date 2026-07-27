#!/usr/bin/env python
"""Step 2 — write the Boltz-2 input for the HOLO complex (protein + ligand).

Emits outputs/boltz_7dli.yaml co-folding chain B (with the FFQQFF loop inserted
after residue 395) together with the 7dli ligand, using:
  - the deposited 7DLI mmCIF as a structural template,
  - single-sequence (no-MSA) mode (nothing sent to an external server),
  - a pocket constraint tying the ligand to the crystallographic binding
    residues (auto-detected from the docked ligand) so the co-folded pose lands
    in the right site.

Modelling the loop in the ligand-bound (holo) state matters here because the
loop is only ~7 A from the pocket and shifts ~2 A when the ligand is present.

    python scripts/2_boltz_prep.py
"""

import pathlib
import numpy as np
from rdkit import Chem

REPO = pathlib.Path(__file__).resolve().parent.parent
RECFINAL = REPO / "inputs" / "recfinal_7dli_water.pdb"
LIGAND_SDF = REPO / "inputs" / "cry1_ligands.sdf"
YAML = REPO / "outputs" / "boltz_7dli.yaml"
TEMPLATE = "inputs/7dli.cif"
LIGAND_NAME = "7dli"

LOOP_AFTER_RESID = 395
LOOP_SEQ = "FFQQFF"
POCKET_CUTOFF = 5.0        # A; protein residues within this of the ligand

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

# --- target sequence with the loop inserted ---
seq, seen = [], set()
for line in RECFINAL.read_text().splitlines():
    if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[21] == "B":
        n = int(line[22:26])
        if n not in seen:
            seen.add(n)
            seq.append((n, THREE_TO_ONE[line[17:20].strip()]))
one = "".join(c for _, c in seq)
idx = next(i for i, (n, _) in enumerate(seq) if n == LOOP_AFTER_RESID)
target = one[: idx + 1] + LOOP_SEQ + one[idx + 1:]

# --- ligand SMILES from the input SDF (same molecule that is docked) ---
smiles, lig_xyz = None, None
for m in Chem.SDMolSupplier(str(LIGAND_SDF), removeHs=True):
    if m is not None and m.GetProp("_Name").strip() == LIGAND_NAME:
        smiles = Chem.MolToSmiles(m)
        lig_xyz = np.array([list(m.GetConformer().GetAtomPosition(i))
                            for i in range(m.GetNumAtoms())])
        break
if smiles is None:
    raise SystemExit(f"ligand '{LIGAND_NAME}' not found in {LIGAND_SDF}")

# --- pocket residues within POCKET_CUTOFF of the docked ligand (query numbering
#     == receptor numbering for these pre-loop residues) ---
prot = {}
for line in RECFINAL.read_text().splitlines():
    if line.startswith("ATOM") and line[21] == "B":
        resid = int(line[22:26])
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        prot.setdefault(resid, []).append(xyz)
pocket = sorted(r for r, atoms in prot.items()
                if min(np.min(np.linalg.norm(lig_xyz - a, axis=1)) for a in atoms)
                < POCKET_CUTOFF)
contacts = ", ".join(f"[A, {r}]" for r in pocket)

YAML.parent.mkdir(exist_ok=True)
YAML.write_text(
    "version: 1\n"
    "sequences:\n"
    "  - protein:\n"
    "      id: A\n"
    f"      sequence: {target}\n"
    "      msa: empty\n"
    "  - ligand:\n"
    "      id: B\n"
    f"      smiles: '{smiles}'\n"
    "constraints:\n"
    "  - pocket:\n"
    "      binder: B\n"
    f"      contacts: [{contacts}]\n"
    "      max_distance: 6.0\n"
    "templates:\n"
    f"  - cif: {TEMPLATE}\n"
)
print(f"wrote {YAML.relative_to(REPO)}")
print(f"  protein: {len(target)} residues (loop {LOOP_SEQ} after {LOOP_AFTER_RESID})")
print(f"  ligand:  {smiles}")
print(f"  pocket contacts ({len(pocket)} residues < {POCKET_CUTOFF} A): {pocket}")
