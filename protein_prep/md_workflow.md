# A reproducible protocol for preparing and simulating the cryptochrome CRY1 (PDB 7DLI) receptor–ligand complex in explicit solvent

## Abstract

We describe an end-to-end, script-driven workflow for preparing and running
molecular dynamics (MD) of a docked small molecule in the *Drosophila*
cryptochrome CRY1 binding pocket (derived from PDB entry 7DLI), built entirely
within the OpenFF/OpenFE ecosystem so that the prepared system is directly
reusable for subsequent free-energy perturbation (FEP) calculations. The
protocol comprises: (i) automated protein preparation with PDBFixer, including
removal of a spurious free amino acid and retention of crystallographic waters;
(ii) sequence-guided identification and *de novo* modelling of a disordered
six-residue loop that was absent from the deposited coordinates; (iii) ligand
preparation and assignment of AM1-BCC charges; (iv) system assembly and
parametrization with the Amber ff14SB and OpenFF Sage force fields; and (v)
minimization, equilibration and production MD using the OpenFE `PlainMDProtocol`.
All steps are implemented as standalone Python scripts and are reproducible on a
single workstation GPU.

## 1. Introduction

Structure-based free-energy methods require a physically complete and internally
consistent starting model: missing side chains and backbone segments,
inappropriate protonation states, and unparametrized heteroatoms are common
causes of failed or unreliable simulations. The CRY1 receptor model used here
(`recfinal_7dli_water.pdb`) is a processed form of PDB 7DLI comprising a single
protein chain (chain B, 475 modelled residues) together with 170
crystallographic water molecules. Because the same chemical system is intended
to seed later FEP calculations, we prepared it with the OpenFE toolchain, whose
`PlainMDProtocol` shares its parametrization and simulation engine with the
OpenFE relative binding free-energy protocols.

During preparation we encountered two issues that are representative of
real-world structures and that motivated the additional steps documented below:
a stray, unbonded lysine residue that has no protein force-field template, and a
disordered surface loop that had been numbered over in the input file, producing
a physically discontinuous backbone.

## 2. Methods

### 2.1 Computational environment

Protein preparation, system assembly and MD were performed with OpenFE 1.10.0
and gufe 1.9.0, using OpenMM 8.4.0, the OpenFF Toolkit 0.18.0,
OpenMMForceFields 0.15.1, PDBFixer 1.12.0 and RDKit 2024.09.2. Protonation-state
assignment used PDB2PQR 3.7.1 with PROPKA 3.5.1. Loop modelling was performed in
separate environments using PyTorch 2.6.0 (CUDA 12.4 build), Boltz-2 (v2.2.1) and
Biopython 1.87. All calculations were run on a workstation
equipped with an NVIDIA RTX 4000 Ada Generation GPU (20 GB, driver 550.127.05,
CUDA 12.4).

We note that the OpenMM CUDA platform on this host failed to initialize with
`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`, because the distributed OpenMM binaries
target a newer CUDA toolkit than the installed driver supports. All MD was
therefore executed on the functionally equivalent OpenCL platform, which runs on
the same GPU; PyTorch (used for loop modelling) was pinned to a CUDA 12.4 wheel
for the same reason.

### 2.2 Protein preparation

The receptor was processed with PDBFixer to add missing heavy atoms. A single
free lysine residue present as an isolated, unbonded chain (chain C, LYS113) was
removed, since a free amino acid has no template in the Amber protein force field
and it is not part of the functional protein; chains consisting of a single
standard amino-acid residue are detected and removed automatically.
Crystallographic waters were retained.

Protonation states were then assigned at pH 7.0 with PDB2PQR (v3.7.1) using
PROPKA (v3.5.1) for pKa prediction, rather than a fixed-rule scheme: this sets
histidine tautomers (HID/HIE/HIP) and the protonation of Asp/Glu/Lys/Cys
according to their predicted pKa and local hydrogen-bonding. The resulting states
were applied as residue variants when OpenMM placed the hydrogens, so that the
added atoms remain compatible with the Amber ff14SB templates; crystallographic
waters were protonated in the same step. For 7DLI this assigned eight histidines
as HID, one as HIE and one as doubly-protonated HIP, and one buried glutamate
(Glu285) as neutral GLH — states a fixed-pH template scheme would not capture.
One cysteine flagged by PROPKA as a thiolate (CYM) was retained as neutral CYS,
as a lone deprotonated cysteine at pH 7 is unusual and warrants manual inspection.
If PDB2PQR is unavailable the pipeline falls back to OpenMM's template-based
hydrogen placement at the same pH.

### 2.3 Identification and modelling of the missing loop

Inspection of the backbone revealed a Cα–Cα distance of 14.8 Å between Ser395
and His396 (versus ~3.8 Å for a peptide bond), indicating a physical chain
break, despite continuous residue numbering in the input file. Because the input
lacked `SEQRES`/sequence records, PDBFixer's numbering-based gap detection could
not identify the break.

To determine the true content of the gap, the observed chain-B sequence was
aligned to the deposited 7DLI polymer sequence (`_entity_poly`, 498 residues)
parsed from the mmCIF file. The alignment placed six residues —
Phe-Phe-Gln-Gln-Phe-Phe (FFQQFF) — between Ser395 and His396; these residues are
disordered in the crystal structure and had been numbered over during earlier
processing.

The decision to model rather than cap the gap was based on its proximity to the
ligand: the loop lies 7.2 Å from the docked 7DLI ligand, on the rim of the
binding pocket. Capping and leaving a truncated gap is an accepted treatment for
missing loops that are remote from the region of interest, but here it would
place artificial termini and a cavity against the pocket wall; modelling the loop
is therefore the appropriate choice.

The loop was modelled by predicting the complete chain-B structure — with the
FFQQFF segment inserted at its correct sequence position — and grafting only the
six loop residues into the crystallographic receptor, so that the experimental
binding-site coordinates, crystallographic waters and original residue numbering
(His396, Tyr398, … unchanged; the loop enters as insertion codes 395A–395F) are
all preserved. The graft superposes the predicted model onto the receptor using
the Cα atoms of the residues flanking the loop on both sides (Biopython
`Superimposer`) before transferring the loop.

Two structure predictors were evaluated. A template-free language-model
prediction (ESMFold, `facebook/esmfold_v1`) reproduced the loop but, because it
is not conditioned on the input structure, its C-terminal end did not meet the
receptor after superposition: the Cys-side junction C(395F)–N(396) measured
2.75 Å, too long for a peptide bond to be inferred, leaving the loop bonded to
Ser395 but not to His396. It was therefore replaced by a template-conditioned
prediction with **Boltz-2** (v2.2.1), using the deposited 7DLI mmCIF as the
structural template and single-sequence (no-MSA) mode so that no sequence data
leave the workstation. Because Boltz-2 builds the loop in the context of both
anchors, the predicted loop is internally closed (all backbone C–N distances
1.32–1.33 Å), and after grafting the C(395F)–N(396) junction is 0.91 Å — short
enough for the peptide bond to be recognised and relaxed to 1.34 Å during energy
minimization. The Boltz-2 model was of high overall confidence (pTM 0.93,
complex pLDDT 0.90); per-residue confidence for the flexible loop itself was
pLDDT 56–87, typical of an intrinsically disordered surface loop, so its
conformation should be treated as a single plausible model to be relaxed and,
ideally, sampled.

Because the loop is only ~7 Å from the ligand, it was modelled in the
ligand-bound (holo) state: Boltz-2 co-folded the protein together with the 7dli
ligand (supplied as SMILES), conditioned on the 7DLI template and a pocket
constraint tying the ligand to the crystallographic binding residues
(auto-detected as those within 5 Å of the docked ligand). This holo model was of
higher confidence than the apo prediction (pTM 0.96, complex pLDDT 0.92, ligand
ipTM 0.96), and the co-folded ligand reproduced the crystallographic binding mode
(centroid within ~1 Å of the docked pose after pocket alignment). Critically, the
ligand shifts the loop: with flanks aligned (0.99 Å RMSD) the holo loop backbone
differs from the apo loop by a mean of 2.3 Å (up to 3.5 Å), confirming that an
apo model would misplace this pocket-adjacent loop. The holo loop was therefore
grafted into the receptor and used for MD.

### 2.4 Ligand preparation

The docked ligand corresponding to 7DLI was extracted by name from a
multi-molecule SD file (`cry1_ligands.sdf`). Explicit hydrogens were added with
coordinates where absent (RDKit); the ligand is neutral and comprises 50 atoms.
The molecule was represented as an OpenFE `SmallMoleculeComponent`.

### 2.5 System assembly and parametrization

An OpenFE `ChemicalSystem` was assembled from the prepared protein
(`ProteinComponent`), the ligand (`SmallMoleculeComponent`) and a
`SolventComponent` specifying 0.15 M NaCl. The protein and water were described
by the Amber ff14SB and TIP3P force fields; the ligand was described by the
OpenFF Sage small-molecule force field with AM1-BCC partial charges, assigned
through OpenMMForceFields. The complex was solvated in a periodic box with 1.0 nm
padding and neutralized.

### 2.6 Molecular dynamics protocol

Simulations used the OpenFE `PlainMDProtocol`. Nonbonded interactions were
treated with particle-mesh Ewald electrostatics and a real-space cutoff;
hydrogen-bond lengths were constrained. Following energy minimization, the system
was equilibrated for 50 ps in the NVT ensemble and 50 ps in the NPT ensemble
(Monte Carlo barostat, 1 bar), and a 0.5 ns production trajectory was collected
in the NPT ensemble at 300 K. Equilibration and production lengths are exposed as
parameters at the top of the driver script and should be increased for
production-quality sampling. Coordinates were written to `system.pdb`,
`minimized.pdb`, `equil_nvt.pdb`, `equil_npt.pdb` and a compressed production
trajectory (`simulation.xtc`), with a run log (`simulation.log`).

### 2.7 Choice of the simulated unit (oligomeric-state assessment)

The deposited 7DLI asymmetric unit contains three protein chains (A, B, C), all
of which are copies of a single entity (identical sequence, differing only in
which flexible residues each resolves) together with one copy of the ligand
(H8X) bound in each chain and a cryoprotectant molecule; no cofactor (e.g. FAD)
is present. The receptor used here (`recfinal_7dli_water.pdb`) is a single copy
(chain B). Two checks confirmed that a monomer is the correct simulated unit.
First, the ligand in chain B contacts only chain B (closest heavy-atom distance
2.8 Å) and is 19–21 Å from chains A and C, so the binding site is entirely
intramolecular. Second, a template-conditioned Boltz-2 prediction of the full
three-chain assembly returned high per-chain confidence but essentially zero
interface confidence (inter-chain ipTM ≈ 0.16 for all three pairs), indicating
that the three chains are crystallographic packing copies rather than a
biological oligomer. The single-chain model was also of markedly higher quality
than any chain within the assembly prediction (pTM 0.93 vs ≤ 0.70). The monomer
was therefore retained for all simulations.

## 3. Results

The complete pipeline executed successfully. Sequence alignment identified the
six missing residues (FFQQFF); template-conditioned Boltz-2 modelling and grafting
produced a continuous backbone with the crystallographic binding site, waters and
residue numbering preserved. In the raw graft the two loop junctions were
strained — the Ser395 hydroxyl overlapped the loop's first backbone nitrogen at
0.81 Å, and the C(395F)–N(396) peptide bond was compressed to 0.91 Å — but both
relaxed to physical geometries under restrained energy minimization (to 3.49 Å
and 1.34 Å respectively). In the NPT-equilibrated system the backbone runs
continuously through the loop with uniform Cα–Cα spacings of 3.8–4.0 Å and no
break, confirming that the loop is covalently closed at both ends. Because the
minimization that resolves the seam is the first step of the MD protocol,
production begins from a clean loop.

Protein preparation removed the spurious lysine and retained crystallographic
waters, yielding a receptor of 8291 atoms. Ligand parametrization and AM1-BCC
charge assignment completed and were cached, and the solvated complex minimized,
equilibrated (NVT and NPT) and produced a 0.5 ns trajectory on the GPU without
error. The prepared system provides a validated starting point for extended
sampling and for relative binding free-energy calculations within the same OpenFE
framework.

## 4. Data and code availability

The workflow is implemented as the following scripts (in `docked/water/`):

| Script | Purpose |
|---|---|
| `run_md_7dli.py` | Protein preparation, ligand loading, system assembly and MD (OpenFE) |
| `analyze_gap.py` | Sequence alignment to the deposited 7DLI sequence to identify the missing loop |
| `boltz_prep.py` | Writes the Boltz-2 input YAML (chain-B sequence with loop, 7DLI template, single-sequence) |
| `graft_loop.py` | Local superposition and grafting of the modelled loop into the receptor |
| `relax_loop_check.py` | Restrained minimization test that both graft seams relax cleanly |
| `fold_chainB_esmfold.py` | ESMFold refolding (initial attempt; superseded by Boltz-2) |
| `diagnose_protein.py` | Force-field template diagnostics for the prepared receptor |

Inputs: `recfinal_7dli_water.pdb` (receptor and crystallographic waters);
`../cry1_7dli/cry1_ligands.sdf` (docked ligands); `7dli.cif` (deposited
structure and sequence).

## References (software)

1. Eastman P. *et al.* OpenMM 8. *J. Phys. Chem. B* (2024).
2. OpenFE / gufe: Alibay I. *et al.* The Open Free Energy toolkit.
3. Wagner J. *et al.* The OpenFF Toolkit and Sage force field.
4. Passaro S., Corso G. *et al.* Boltz-2: towards accurate and efficient
   binding-affinity prediction (2025); Wohlwend J. *et al.* Boltz-1 (2024).
5. Lin Z. *et al.* Evolutionary-scale prediction of atomic-level protein
   structure with a language model (ESMFold). *Science* (2023).
6. Cock P.J.A. *et al.* Biopython. *Bioinformatics* (2009).
