#!/usr/bin/env python
"""Step 1 — identify the missing loop.

Aligns the modelled chain-B sequence of the receptor to the deposited 7DLI
polymer sequence (from the mmCIF) to determine the residues missing at the
Ser395/His396 chain break. Run with either environment's python.

    python scripts/1_analyze_gap.py
"""

import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
PDB = REPO / "inputs" / "recfinal_7dli_water.pdb"
CIF = REPO / "inputs" / "7dli.cif"

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def read_entity_seqs(cif):
    seqs, lines, i = [], cif.read_text().splitlines(), 0
    while i < len(lines):
        if lines[i].strip().startswith("_entity_poly.pdbx_seq_one_letter_code_can"):
            rest = lines[i].split("pdbx_seq_one_letter_code_can", 1)[1].strip()
            if rest and not rest.startswith(";"):
                seqs.append(rest.strip().strip("'\""))
            else:
                j, buf = i + 1, []
                if lines[j].startswith(";"):
                    buf.append(lines[j][1:]); j += 1
                    while not lines[j].startswith(";"):
                        buf.append(lines[j]); j += 1
                    seqs.append("".join(buf).replace("\n", "").strip())
        i += 1
    return seqs


def chain_b(pdb):
    seq, seen = [], set()
    for line in pdb.read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[21] == "B":
            n = int(line[22:26])
            if n not in seen:
                seen.add(n)
                seq.append((n, THREE_TO_ONE.get(line[17:20].strip(), "X")))
    return seq


protein_seq = max(read_entity_seqs(CIF), key=len)
cb = chain_b(PDB)
before = "".join(c for n, c in cb if 385 <= n <= 395)
after = "".join(c for n, c in cb if 396 <= n <= 406)
pb, pa = protein_seq.find(before), protein_seq.find(after)
gap = protein_seq[pb + len(before):pa]
print(f"receptor chain B: {len(cb)} residues ({cb[0][0]}..{cb[-1][0]})")
print(f"flank before break (385-395): {before}")
print(f"flank after  break (396-406): {after}")
print(f"\nMISSING loop between Ser395 and His396: {gap}  ({len(gap)} residues)")
print(f"context: ...{protein_seq[pb:pa+len(after)]}...")
