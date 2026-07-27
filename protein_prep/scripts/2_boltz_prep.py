#!/usr/bin/env python
"""Step 2 — write the Boltz-2 input for the monomer.

Emits outputs/boltz_7dli.yaml: chain B's sequence with the FFQQFF loop inserted
after residue 395, using the deposited 7DLI mmCIF as a structural template and
single-sequence (no-MSA) mode so nothing is sent to an external server.

    python scripts/2_boltz_prep.py
"""

import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
RECFINAL = REPO / "inputs" / "recfinal_7dli_water.pdb"
YAML = REPO / "outputs" / "boltz_7dli.yaml"
TEMPLATE = "inputs/7dli.cif"           # relative to REPO (boltz is run from REPO)

LOOP_AFTER_RESID = 395
LOOP_SEQ = "FFQQFF"

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

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

YAML.parent.mkdir(exist_ok=True)
YAML.write_text(
    "version: 1\n"
    "sequences:\n"
    "  - protein:\n"
    "      id: A\n"
    f"      sequence: {target}\n"
    "      msa: empty\n"
    "templates:\n"
    f"  - cif: {TEMPLATE}\n"
)
print(f"wrote {YAML.relative_to(REPO)}: {len(target)} residues "
      f"(loop {LOOP_SEQ} after {LOOP_AFTER_RESID}), template {TEMPLATE}")
