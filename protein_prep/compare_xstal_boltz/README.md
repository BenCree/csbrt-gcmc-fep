# Crystal vs Boltz-2 comparison (7DLI)

Both structures are in the **same coordinate frame** — open them together in PyMOL
and they overlay directly:

```
pymol xstal_7dli_chainB.pdb boltz_holo_aligned.pdb
```

| File | Contents |
|------|----------|
| `xstal_7dli_chainB.pdb` | Deposited 7DLI, chain B (protein + native **H8X** ligand). The 396–401 loop is disordered/absent here. |
| `boltz_holo_aligned.pdb` | Boltz-2 holo model (protein with the **modelled FFQQFF loop** + co-folded ligand `LIG`), superposed onto the crystal by sequence-aligned Cα atoms. |

Agreement (Boltz vs crystal):

- Protein backbone: 475 Cα, **1.31 Å RMSD**
- Ligand pose: co-folded ligand **1.60 Å** (centroid) from crystallographic H8X

Things to look at:
- The **modelled loop** (residues 396–401 in the Boltz model) filling the gap the
  crystal leaves between Ser395 and His396.
- The **ligand pose** — Boltz's `LIG` vs the crystal `H8X` in the pocket.
