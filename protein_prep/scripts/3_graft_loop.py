#!/usr/bin/env python
"""Step 3 — graft the Boltz-modelled loop into the crystal receptor.

Locally superposes the Boltz-2 model onto the receptor using Cα atoms flanking
the loop on both sides, then copies only the six loop residues into chain B as
insertion codes 395A-395F. Crystal coordinates, waters and residue numbering are
preserved; because Boltz builds the loop against both anchors, the C(395F)-N(396)
peptide bond closes (relaxes to ~1.33 A during the MD minimization).

Requires biopython + numpy (present in the boltz environment).

    python scripts/3_graft_loop.py
"""

import glob
import pathlib
import numpy as np
from Bio.PDB import PDBParser, PDBIO, Superimposer

REPO = pathlib.Path(__file__).resolve().parent.parent
RECFINAL = REPO / "inputs" / "recfinal_7dli_water.pdb"
OUT = REPO / "outputs" / "recfinal_7dli_water_loopmodelled.pdb"

# locate the Boltz prediction produced by run_all.sh
hits = glob.glob(str(REPO / "outputs" / "boltz_out" / "**" / "*_model_0.pdb"),
                 recursive=True)
if not hits:
    raise SystemExit("No Boltz model found under outputs/boltz_out/ — run step 2b first.")
BOLTZ = hits[0]

LOOP_AFTER = 395
N_LOOP = 6
INS_CODES = "ABCDEF"
ESM_LOOP_IDS = list(range(LOOP_AFTER + 1, LOOP_AFTER + 1 + N_LOOP))        # 396..401
PRE = [(r, r) for r in range(LOOP_AFTER - 15, LOOP_AFTER + 1)]             # 380..395
POST = [(r, r + N_LOOP) for r in range(LOOP_AFTER + 1, LOOP_AFTER + 17)]   # rec396..411

parser = PDBParser(QUIET=True)
rec = parser.get_structure("rec", str(RECFINAL))[0]
model = parser.get_structure("m", BOLTZ)[0]
rec_b = rec["B"]
mdl = model["A"] if "A" in model else list(model.get_chains())[0]


def ca(chain, resid, icode=" "):
    key = (" ", resid, icode)
    return chain[key]["CA"] if key in chain and "CA" in chain[key] else None


fixed, moving = [], []
for r_rec, r_mdl in PRE + POST:
    a, b = ca(rec_b, r_rec), ca(mdl, r_mdl)
    if a is not None and b is not None:
        fixed.append(a); moving.append(b)
sup = Superimposer()
sup.set_atoms(fixed, moving)
sup.apply(model.get_atoms())
print(f"superposed on {len(fixed)} flanking CA atoms, RMSD {sup.rms:.2f} A")

loop_residues = []
for i, resid in enumerate(ESM_LOOP_IDS):
    res = mdl[(" ", resid, " ")]
    res.id = (" ", LOOP_AFTER, INS_CODES[i])
    loop_residues.append(res)

original = list(rec_b)
for r in original:
    rec_b.detach_child(r.id)
for r in original:
    rec_b.add(r)
    if r.id == (" ", LOOP_AFTER, " "):
        for lr in loop_residues:
            lr.detach_parent(); rec_b.add(lr)

OUT.parent.mkdir(exist_ok=True)
io = PDBIO(); io.set_structure(rec); io.save(str(OUT))
print(f"wrote {OUT.relative_to(REPO)}")

c = ca(rec_b, LOOP_AFTER, "F"); n_atom = rec_b[(" ", 396, " ")]["N"] if (" ", 396, " ") in rec_b else None
if c is not None and n_atom is not None:
    d = float(np.linalg.norm(np.array(c.get_coord()) - np.array(n_atom.get_coord())))
    print(f"C(395F)-N(396) junction in raw graft: {d:.2f} A (bonds & relaxes to ~1.33 A in MD)")
