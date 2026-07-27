#!/usr/bin/env python
"""Step 4 — protein preparation + MD of the loop-completed complex (OpenFE).

Fixes the loop-completed receptor (removes the stray free lysine, keeps crystal
waters, adds hydrogens), loads the 7dli ligand, assembles an OpenFE
ChemicalSystem and runs PlainMDProtocol: minimize -> NVT -> NPT -> production.

Run in the MD environment:
    python scripts/4_run_md.py
"""

import pathlib

import gufe
from openfe import (ChemicalSystem, ProteinComponent, SmallMoleculeComponent,
                    SolventComponent)
from openfe.protocols.openmm_md.plain_md_methods import PlainMDProtocol
from openff.units import unit
from openmm.app import Modeller, PDBFile
from pdbfixer import PDBFixer
from rdkit import Chem

AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HID", "HIE", "HIP", "CYX", "ASH", "GLH", "LYN",
}

REPO = pathlib.Path(__file__).resolve().parent.parent
PROTEIN_PDB = REPO / "outputs" / "recfinal_7dli_water_loopmodelled.pdb"
FIXED_PDB = REPO / "outputs" / "recfinal_7dli_water_loopmodelled_fixed.pdb"
LIGAND_SDF = REPO / "inputs" / "cry1_ligands.sdf"
LIGAND_NAME = "7dli"
WORKDIR = REPO / "outputs" / "md"

# Short-but-real defaults; increase for production sampling.
NVT_NS = 0.05
NPT_NS = 0.05
PROD_NS = 0.5
SOLVENT_PADDING_NM = 1.0
N_REPEATS = 1
# OpenCL runs on the GPU where the distributed OpenMM CUDA build is too new for
# the driver. Set 'CUDA' if your OpenMM/CUDA matches the driver, or 'CPU'.
COMPUTE_PLATFORM = "OpenCL"


def fix_protein(in_pdb, out_pdb):
    print(f"[1/4] Fixing protein: {in_pdb.name}")
    fixer = PDBFixer(filename=str(in_pdb))
    fixer.findMissingResidues()
    fixer.missingResidues = {}          # loop is already modelled in
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    # Drop stray single amino-acid chains (e.g. free lysine); keep waters.
    modeller = Modeller(fixer.topology, fixer.positions)
    stray = []
    for chain in modeller.topology.chains():
        non_water = [r for r in chain.residues() if r.name != "HOH"]
        if len(non_water) == 1 and non_water[0].name in AMINO_ACIDS:
            stray.append(non_water[0])
    for res in stray:
        print(f"      removing stray residue: chain {res.chain.id} {res.name}{res.id}")
    if stray:
        modeller.delete(stray)

    with open(out_pdb, "w") as fh:
        PDBFile.writeFile(modeller.topology, modeller.positions, fh, keepIds=True)
    print(f"      wrote {out_pdb.name} ({modeller.topology.getNumAtoms()} atoms, "
          "crystal waters kept)")
    return out_pdb


def load_ligand(sdf, name):
    print(f"[2/4] Loading ligand '{name}' from {sdf.name}")
    for mol in Chem.SDMolSupplier(str(sdf), removeHs=False):
        if mol is None or mol.GetProp("_Name").strip() != name:
            continue
        if not any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
            mol = Chem.AddHs(mol, addCoords=True)
        print(f"      {mol.GetNumAtoms()} atoms, charge {Chem.GetFormalCharge(mol):+d}")
        return SmallMoleculeComponent.from_rdkit(mol, name=name)
    raise ValueError(f"No molecule named '{name}' in {sdf}")


def main():
    WORKDIR.mkdir(parents=True, exist_ok=True)
    fixed = fix_protein(PROTEIN_PDB, FIXED_PDB)
    ligand = load_ligand(LIGAND_SDF, LIGAND_NAME)

    print("[3/4] Building ChemicalSystem (protein + ligand + solvent)")
    protein = ProteinComponent.from_pdb_file(str(fixed), name="7dli")
    solvent = SolventComponent(ion_concentration=0.15 * unit.molar)
    system = ChemicalSystem({"protein": protein, "ligand": ligand, "solvent": solvent},
                            name=f"{ligand.name}_7dli")

    settings = PlainMDProtocol.default_settings()
    settings.simulation_settings.equilibration_length_nvt = NVT_NS * unit.nanosecond
    settings.simulation_settings.equilibration_length = NPT_NS * unit.nanosecond
    settings.simulation_settings.production_length = PROD_NS * unit.nanosecond
    settings.solvation_settings.solvent_padding = SOLVENT_PADDING_NM * unit.nanometer
    settings.protocol_repeats = N_REPEATS
    if COMPUTE_PLATFORM is not None:
        settings.engine_settings.compute_platform = COMPUTE_PLATFORM

    protocol = PlainMDProtocol(settings=settings)
    print(f"[4/4] Running MD (NVT {NVT_NS} ns, NPT {NPT_NS} ns, prod {PROD_NS} ns)")
    dag = protocol.create(stateA=system, stateB=system, mapping=None)
    dagres = gufe.protocols.execute_DAG(
        dag, shared_basedir=WORKDIR, scratch_basedir=WORKDIR,
        keep_shared=True, n_retries=1,
    )
    print(f"\nDone. Success: {dagres.ok()}")
    if not dagres.ok():
        for f in dagres.protocol_unit_failures:
            print(f.traceback)


if __name__ == "__main__":
    main()
