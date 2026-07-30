#!/usr/bin/env python
"""Preprocessing step 2 - write the OpenFold3 query JSON for the holo complex.

Port of Ben's ``2_boltz_prep.py`` from Boltz-2 to OpenFold3. Same idea: co-fold
the receptor chain (with the missing loop inserted) together with the ligand,
using the deposited mmCIF as an explicit structural template so the co-folded
model reproduces the crystal conformation.

Why OpenFold3 rather than Boltz-2: boltz 2.2.1 requires ``numpy<2.0`` while the
sire/somd2/loch stack is built against numpy 2.x, so the two cannot share an
environment. OpenFold3 leaves numpy unpinned and supports explicit template CIFs
(``template_cif_paths``, "CIF-direct mode") plus MSAs via the ColabFold server,
so the whole workflow fits in one environment.

Difference from the Boltz version to be aware of: Boltz accepted an explicit
``pocket`` constraint tying the ligand to named binding residues. OpenFold3's
query schema has no equivalent, so the pocket residues are reported here for the
record but not passed to the model; the template is what localises the pose.

    python scripts/of3_prep.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import Chem

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def options() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--receptor", type=Path, required=True,
                   help="Receptor PDB whose chain is to be loop-completed")
    p.add_argument("--template-cif", type=Path, required=True,
                   help="Deposited mmCIF used as an explicit OpenFold3 template")
    p.add_argument("--ligand-sdf", type=Path, required=True)
    p.add_argument("--ligand-name", required=True,
                   help="_Name of the record to take from --ligand-sdf")
    p.add_argument("--output", type=Path, required=True, help="Query JSON to write")
    p.add_argument("--receptor-chain", default="B")
    p.add_argument("--template-chain", default="B",
                   help="Chain of --template-cif to use as the template")
    p.add_argument("--loop-after-resid", type=int, required=True)
    p.add_argument("--loop-seq", required=True, help="One-letter loop to insert")
    p.add_argument("--pocket-cutoff", type=float, default=5.0,
                   help="Report receptor residues within this distance of the ligand")
    p.add_argument("--query-name", default="holo")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-msas", action="store_true",
                   help="Request MSAs (run predict with --use-msa-server to fetch "
                        "them from the ColabFold server). Default is single-sequence.")
    return p.parse_args()


def chain_residues(pdb: Path, chain: str) -> list[tuple[int, str]]:
    """Ordered (resid, one-letter) for CA atoms of one chain."""
    out, seen = [], set()
    for line in pdb.read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[21] == chain:
            resid = int(line[22:26])
            if resid not in seen:
                seen.add(resid)
                out.append((resid, THREE_TO_ONE[line[17:20].strip()]))
    if not out:
        raise SystemExit(f"No CA atoms for chain {chain} in {pdb}")
    return out


def ligand_from_sdf(sdf: Path, name: str) -> tuple[str, np.ndarray]:
    for mol in Chem.SDMolSupplier(str(sdf), removeHs=True):
        if mol is not None and mol.GetProp("_Name").strip() == name:
            conf = mol.GetConformer()
            xyz = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
            return Chem.MolToSmiles(mol), xyz
    raise SystemExit(f"ligand '{name}' not found in {sdf}")


def pocket_residues(pdb: Path, chain: str, ligand_xyz: np.ndarray, cutoff: float) -> list[int]:
    atoms: dict[int, list[np.ndarray]] = {}
    for line in pdb.read_text().splitlines():
        if line.startswith("ATOM") and line[21] == chain:
            resid = int(line[22:26])
            atoms.setdefault(resid, []).append(
                np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            )
    return sorted(
        resid for resid, coords in atoms.items()
        if min(np.min(np.linalg.norm(ligand_xyz - a, axis=1)) for a in coords) < cutoff
    )


def main() -> None:
    opt = options()
    residues = chain_residues(opt.receptor, opt.receptor_chain)
    sequence = "".join(c for _, c in residues)
    try:
        cut = next(i for i, (resid, _) in enumerate(residues) if resid == opt.loop_after_resid)
    except StopIteration:
        raise SystemExit(f"residue {opt.loop_after_resid} not in chain {opt.receptor_chain}")
    target = sequence[: cut + 1] + opt.loop_seq + sequence[cut + 1:]

    smiles, ligand_xyz = ligand_from_sdf(opt.ligand_sdf, opt.ligand_name)
    pocket = pocket_residues(opt.receptor, opt.receptor_chain, ligand_xyz, opt.pocket_cutoff)

    query = {
        "seeds": [opt.seed],
        "queries": {
            opt.query_name: {
                "query_name": opt.query_name,
                "use_msas": bool(opt.use_msas),
                "chains": [
                    {
                        "molecule_type": "protein",
                        "chain_ids": ["A"],
                        "description": f"{opt.receptor.name} chain {opt.receptor_chain} "
                                       f"+ loop {opt.loop_seq} after {opt.loop_after_resid}",
                        "sequence": target,
                        # explicit template ("CIF-direct" mode), the OpenFold3
                        # equivalent of Boltz's `templates: - cif:`
                        "template_cif_paths": [str(opt.template_cif.resolve())],
                        "template_cif_chain_ids": [opt.template_chain],
                    },
                    {
                        "molecule_type": "ligand",
                        "chain_ids": ["B"],
                        "description": opt.ligand_name,
                        "smiles": smiles,
                    },
                ],
            }
        },
    }
    opt.output.parent.mkdir(parents=True, exist_ok=True)
    opt.output.write_text(json.dumps(query, indent=2) + "\n")

    print(f"wrote {opt.output}")
    print(f"  protein : {len(target)} residues "
          f"(loop {opt.loop_seq} inserted after {opt.loop_after_resid})")
    print(f"  template: {opt.template_cif.name} chain {opt.template_chain}")
    print(f"  ligand  : {smiles}")
    print(f"  MSAs    : {'ColabFold server' if opt.use_msas else 'single-sequence'}")
    print(f"  pocket  : {len(pocket)} residues within {opt.pocket_cutoff} A "
          f"(reported only; OpenFold3 has no pocket-constraint input): {pocket}")


if __name__ == "__main__":
    main()
