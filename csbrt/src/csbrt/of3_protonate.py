#!/usr/bin/env python
"""Preprocessing step 4 - clean the grafted receptor and protonate it at pH.

PDB2PQR refuses structures containing fragmentary residues, and the deposited
receptors here carry stray single-amino-acid chains (e.g. a free lysine in chain
C of the 7DLI receptor) that trip it with:

    ValueError: Too few atoms present to reconstruct or cap residue LYS C 113

Ben's original MD script dropped those chains before protonating; this does the
same, then runs PDB2PQR/PROPKA to assign titratable states (His tautomers,
Asp/Glu/Lys/Cys) at the requested pH.

    python scripts/of3_protonate.py --help
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

from Bio.PDB import PDBIO, PDBParser, Select

AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY", "HIS", "HID",
    "HIE", "HIP", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL",
}
WATERS = {"HOH", "WAT"}


def options() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True, help="Grafted receptor PDB")
    p.add_argument("--output", type=Path, required=True, help="Protonated PDB to write")
    p.add_argument("--ph", type=float, default=7.4)
    p.add_argument("--keep-waters", action="store_true",
                   help="Keep crystallographic waters in the cleaned structure")
    p.add_argument("--rename-chain", default="A",
                   help="Relabel the remaining protein chain to this ID (the "
                        "endpoint preparation expects chain A). Empty to keep.")
    p.add_argument("--pdb2pqr", default="pdb2pqr30")
    return p.parse_args()


class Keep(Select):
    def __init__(self, drop_chains: set[str], keep_waters: bool):
        self.drop_chains = drop_chains
        self.keep_waters = keep_waters

    def accept_chain(self, chain):  # noqa: D102
        return chain.id not in self.drop_chains

    def accept_residue(self, residue):  # noqa: D102
        if residue.get_resname().strip() in WATERS:
            return self.keep_waters
        return True


def fix_ter_records(path: Path) -> int:
    """Rewrite bare ``TER`` lines with the PDB-standard residue identification.

    PDB2PQR emits a bare ``TER`` with no fields, but the endpoint receptor audit
    requires the TER record to name the final residue (chain in column 22,
    residue number in 23-26), so it is reconstructed from the preceding ATOM.
    """
    lines = path.read_text().splitlines()
    fixed = 0
    previous: str | None = None
    for index, line in enumerate(lines):
        if line.startswith(("ATOM", "HETATM")):
            previous = line
        elif line.startswith("TER") and previous is not None:
            if len(line.rstrip()) > 26:
                continue  # already carries the fields
            serial = int(previous[6:11]) + 1
            lines[index] = (
                "TER   "
                + f"{serial:>5}"
                + " " * 6
                + f"{previous[17:20]:>3}"
                + " "
                + previous[21]
                + f"{previous[22:26]:>4}"
                + previous[26]
            )
            fixed += 1
    if fixed:
        path.write_text("\n".join(lines) + "\n")
    return fixed


def main() -> None:
    opt = options()
    structure = PDBParser(QUIET=True).get_structure("rec", str(opt.input))
    model = structure[0]

    drop: set[str] = set()
    for chain in model:
        residues = [r.get_resname().strip() for r in chain
                    if r.get_resname().strip() not in WATERS]
        if len(residues) == 1 and residues[0] in AMINO_ACIDS:
            drop.add(chain.id)
            print(f"  dropping stray single-residue chain {chain.id} ({residues[0]})")
        elif not residues:
            # water-only chain: keep or drop with the waters
            if not opt.keep_waters:
                drop.add(chain.id)

    # The endpoint preparation accepts a single protein chain named A; relabel
    # the one surviving chain rather than making callers pre-edit the PDB.
    # Detach dropped chains up front. Doing this before any relabelling matters:
    # the water chain here is 'A', so renaming the protein chain B -> A while a
    # drop-set still referenced 'A' would filter out the protein itself.
    for chain_id in sorted(drop):
        model.detach_child(chain_id)

    if opt.rename_chain:
        remaining = list(model)
        if len(remaining) == 1 and remaining[0].id != opt.rename_chain:
            chain = remaining[0]
            print(f"  relabelling chain {chain.id} -> {opt.rename_chain}")
            # Reassigning chain.id alone does not update the parent's child
            # dictionary, so the saved file would keep the old ID.
            model.detach_child(chain.id)
            chain.id = opt.rename_chain
            model.add(chain)
        elif len(remaining) > 1:
            print(f"  {len(remaining)} chains remain; not relabelling "
                  f"({[c.id for c in remaining]})")

    cleaned = opt.output.with_name(opt.output.stem + "_cleaned.pdb")
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(cleaned), Keep(set(), opt.keep_waters))
    print(f"  cleaned structure -> {cleaned}")

    # Resolve pdb2pqr from PATH, else from this interpreter's bin directory, so
    # the script works whether or not the environment has been activated.
    binary = shutil.which(opt.pdb2pqr)
    if binary is None:
        sibling = Path(sys.executable).resolve().parent / opt.pdb2pqr
        if not sibling.is_file():
            raise SystemExit(
                f"'{opt.pdb2pqr}' not found on PATH or beside {sys.executable}; "
                "activate the csbrt environment or pass --pdb2pqr"
            )
        binary = str(sibling)
    pqr = opt.output.with_suffix(".pqr")
    command = [binary, "--ff=AMBER", "--ffout=AMBER", "--keep-chain",
               f"--with-ph={opt.ph}", "--titration-state-method=propka",
               f"--pdb-output={opt.output}", str(cleaned), str(pqr)]
    print("  " + " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join((result.stderr or result.stdout).splitlines()[-12:])
        raise SystemExit(f"pdb2pqr failed (exit {result.returncode}):\n{tail}")
    if not opt.output.is_file() or opt.output.stat().st_size == 0:
        raise SystemExit(f"pdb2pqr produced no output at {opt.output}")
    fixed = fix_ter_records(opt.output)
    if fixed:
        print(f"  rewrote {fixed} bare TER record(s) with residue identification")
    print(f"  protonated at pH {opt.ph} -> {opt.output}")


if __name__ == "__main__":
    main()
