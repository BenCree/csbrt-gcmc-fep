#!/usr/bin/env python
"""Preprocessing step 5 - add explicit hydrogens to a 3D ligand SDF.

The endpoint preparation requires a sanitizable, explicit-hydrogen, 3D ligand
record; crystallographic/docked SDFs frequently carry heavy atoms only and fail
with "Ligand SDF has no explicit hydrogens".

Heavy-atom coordinates are preserved exactly. Hydrogens are added with
coordinates and then relaxed with the heavy atoms fixed, so the pose is
untouched but the added hydrogens are not left at idealised positions that clash.

    python scripts/prepare_ligands.py --help
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def options() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True, help="SDF with 3D heavy atoms")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--names", nargs="*", help="Only these record names (default: all)")
    p.add_argument("--no-minimise", action="store_true",
                   help="Place hydrogens geometrically without relaxing them")
    return p.parse_args()


def heavy_coords(mol: Chem.Mol) -> np.ndarray:
    conf = mol.GetConformer()
    return np.array([list(conf.GetAtomPosition(a.GetIdx()))
                     for a in mol.GetAtoms() if a.GetAtomicNum() > 1])


def main() -> None:
    opt = options()
    supplier = Chem.SDMolSupplier(str(opt.input), removeHs=False)
    writer = Chem.SDWriter(str(opt.output))
    written = 0
    try:
        for mol in supplier:
            if mol is None:
                continue
            name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
            if opt.names and name not in opt.names:
                continue
            if mol.GetNumConformers() == 0:
                raise SystemExit(f"record '{name}' has no 3D conformer")
            before = heavy_coords(mol)

            protonated = Chem.AddHs(mol, addCoords=True)
            if not opt.no_minimise:
                # Relax hydrogens only: constrain every heavy atom.
                field = AllChem.MMFFGetMoleculeForceField(
                    protonated, AllChem.MMFFGetMoleculeProperties(protonated)
                ) if AllChem.MMFFHasAllMoleculeParams(protonated) else \
                    AllChem.UFFGetMoleculeForceField(protonated)
                if field is not None:
                    for atom in protonated.GetAtoms():
                        if atom.GetAtomicNum() > 1:
                            field.AddFixedPoint(atom.GetIdx())
                    field.Minimize(maxIts=500)

            after = heavy_coords(protonated)
            shift = float(np.abs(after - before).max()) if len(before) else 0.0
            if shift > 1e-3:
                raise SystemExit(
                    f"record '{name}': heavy atoms moved {shift:.4f} A; pose not preserved"
                )
            # The endpoint preparation requires 'charge' and 'multiplicity'
            # metadata and cross-checks charge against the molecular graph.
            charge = int(Chem.GetFormalCharge(protonated))
            radicals = sum(a.GetNumRadicalElectrons() for a in protonated.GetAtoms())
            multiplicity = radicals + 1
            protonated.SetProp("_Name", name)
            protonated.SetProp("charge", str(charge))
            protonated.SetProp("multiplicity", str(multiplicity))
            writer.write(protonated)
            written += 1
            added = protonated.GetNumAtoms() - mol.GetNumAtoms()
            print(f"  {name:8s} +{added:3d} H  charge={charge:+d} mult={multiplicity}  "
                  f"(heavy-atom pose preserved, max shift {shift:.1e} A)")
    finally:
        writer.close()
    if not written:
        raise SystemExit("no records written; check --names")
    print(f"wrote {written} record(s) -> {opt.output}")


if __name__ == "__main__":
    main()
