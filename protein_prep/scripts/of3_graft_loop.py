#!/usr/bin/env python
"""Preprocessing step 3 - graft the OpenFold3-modelled loop into the receptor.

Port of Ben's ``3_graft_loop.py`` from Boltz-2 to OpenFold3. The method is
unchanged: superpose the predicted model onto the crystal receptor using CA atoms
flanking the loop on both sides, then copy only the modelled loop residues into
the receptor chain as insertion codes (e.g. 395A-395F). Crystal coordinates,
waters and residue numbering are preserved.

The only real difference is the input: OpenFold3 writes mmCIF (Boltz wrote PDB),
so the model is located by glob and parsed with whichever biopython parser suits
the extension.

    python scripts/of3_graft_loop.py --help
"""

from __future__ import annotations

import argparse
from pathlib import Path
import string

import numpy as np
from Bio.PDB import MMCIFParser, PDBIO, PDBParser, Superimposer


def options() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--receptor", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True,
                   help="OpenFold3 --output-dir; the top-ranked model is located by glob")
    p.add_argument("--model", type=Path,
                   help="Explicit model file, bypassing the glob")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--receptor-chain", default="B")
    p.add_argument("--model-chain", default="A")
    p.add_argument("--loop-after-resid", type=int, required=True)
    p.add_argument("--loop-length", type=int, required=True)
    p.add_argument("--flank", type=int, default=16,
                   help="CA atoms on each side used for the superposition")
    p.add_argument("--renumber", action="store_true", default=True,
                   help="Renumber the grafted chain contiguously, removing the "
                        "insertion codes (required downstream: the endpoint "
                        "preparation refuses insertion codes)")
    p.add_argument("--keep-insertion-codes", dest="renumber", action="store_false",
                   help="Preserve crystal numbering with 395A-style insertion codes")
    return p.parse_args()


def renumber_contiguously(chain, mapping_path: Path) -> None:
    """Renumber a chain 1..N without insertion codes, recording the mapping.

    Ben's graft keeps crystal numbering by inserting the loop as 395A-395F, but
    the endpoint preparation refuses insertion codes because downstream stages
    key on unique integer residue IDs. Renumbering here keeps that contract while
    the written mapping preserves the correspondence to the original numbering.
    """
    import json

    residues = list(chain)
    mapping = []
    for residue in residues:
        chain.detach_child(residue.id)
    for index, residue in enumerate(residues, start=1):
        het, old_number, old_icode = residue.id
        mapping.append({
            "new_resid": index,
            "old_resid": old_number,
            "old_icode": old_icode.strip(),
            "resname": residue.get_resname().strip(),
        })
        residue.id = (het, index, " ")
        chain.add(residue)
    mapping_path.write_text(json.dumps(mapping, indent=2) + "\n")
    print(f"renumbered {len(residues)} residues 1..{len(residues)} "
          f"(mapping -> {mapping_path.name})")


def find_model(model_dir: Path) -> Path:
    """Locate the top-ranked OpenFold3 model (mmCIF preferred, PDB accepted)."""
    candidates: list[Path] = []
    for pattern in ("**/*sample_0*.cif", "**/*model_0*.cif", "**/*.cif",
                    "**/*sample_0*.pdb", "**/*model_0*.pdb", "**/*.pdb"):
        candidates = sorted(model_dir.glob(pattern))
        if candidates:
            break
    if not candidates:
        raise SystemExit(f"No .cif/.pdb model found under {model_dir}")
    return candidates[0]


def load(path: Path):
    """Load a predicted model as a biopython structure.

    OpenFold3 writes minimal mmCIF that omits ``_atom_site.occupancy``, which
    biopython's MMCIFParser requires, so CIF input is normalised through gemmi
    first (gemmi ships as an OpenFold3 dependency).
    """
    if path.suffix.lower() != ".cif":
        return PDBParser(QUIET=True).get_structure("m", str(path))[0]
    try:
        return MMCIFParser(QUIET=True).get_structure("m", str(path))[0]
    except KeyError:
        import tempfile
        import gemmi

        structure = gemmi.read_structure(str(path))
        structure.setup_entities()
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        if atom.occ == 0.0:
                            atom.occ = 1.0
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as handle:
            converted = Path(handle.name)
        structure.write_pdb(str(converted))
        return PDBParser(QUIET=True).get_structure("m", str(converted))[0]


def ca(chain, resid: int, icode: str = " "):
    key = (" ", resid, icode)
    return chain[key]["CA"] if key in chain and "CA" in chain[key] else None


def main() -> None:
    opt = options()
    model_path = opt.model or find_model(opt.model_dir)
    print(f"model: {model_path}")

    receptor = PDBParser(QUIET=True).get_structure("rec", str(opt.receptor))[0]
    model = load(model_path)
    rec_chain = receptor[opt.receptor_chain]
    mdl_chain = model[opt.model_chain] if opt.model_chain in model else list(model.get_chains())[0]

    after, n_loop = opt.loop_after_resid, opt.loop_length
    if n_loop > len(string.ascii_uppercase):
        raise SystemExit(f"loop of {n_loop} residues exceeds available insertion codes")
    # The model numbers residues contiguously, so everything after the insertion
    # point is shifted by the loop length relative to the receptor numbering.
    pre = [(r, r) for r in range(after - opt.flank + 1, after + 1)]
    post = [(r, r + n_loop) for r in range(after + 1, after + 1 + opt.flank)]

    fixed, moving = [], []
    for rec_resid, mdl_resid in pre + post:
        a, b = ca(rec_chain, rec_resid), ca(mdl_chain, mdl_resid)
        if a is not None and b is not None:
            fixed.append(a)
            moving.append(b)
    if len(fixed) < 3:
        raise SystemExit(f"only {len(fixed)} flanking CA pairs matched; cannot superpose")
    sup = Superimposer()
    sup.set_atoms(fixed, moving)
    sup.apply(model.get_atoms())
    print(f"superposed on {len(fixed)} flanking CA atoms, RMSD {sup.rms:.2f} A")

    loop = []
    for i, resid in enumerate(range(after + 1, after + 1 + n_loop)):
        key = (" ", resid, " ")
        if key not in mdl_chain:
            raise SystemExit(f"model has no residue {resid} to graft")
        residue = mdl_chain[key]
        residue.id = (" ", after, string.ascii_uppercase[i])
        loop.append(residue)

    original = list(rec_chain)
    for residue in original:
        rec_chain.detach_child(residue.id)
    for residue in original:
        rec_chain.add(residue)
        if residue.id == (" ", after, " "):
            for loop_residue in loop:
                loop_residue.detach_parent()
                rec_chain.add(loop_residue)

    # Measure the junction while the insertion codes still identify the loop.
    last = ca(rec_chain, after, string.ascii_uppercase[n_loop - 1])
    nxt = rec_chain[(" ", after + 1, " ")]["N"] if (" ", after + 1, " ") in rec_chain else None
    if last is not None and nxt is not None:
        gap = float(np.linalg.norm(np.array(last.get_coord()) - np.array(nxt.get_coord())))
        print(f"junction C(last loop residue)-N({after + 1}): {gap:.2f} A "
              f"(relaxes to ~1.33 A during minimisation)")

    if opt.renumber:
        renumber_contiguously(
            rec_chain, opt.output.with_name(opt.output.stem + "_residue_map.json")
        )

    opt.output.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(receptor)
    io.save(str(opt.output))
    print(f"wrote {opt.output}")


if __name__ == "__main__":
    main()
