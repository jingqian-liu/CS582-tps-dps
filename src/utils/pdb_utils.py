from pdbfixer import PDBFixer
from openmm.app import PDBFile # Import PDBFile from app for reading and writing

def add_termini_add_hydrogen(input_pdb,output_pdb_file):
    ph=7.0
    temperature=310
    forcefield = "data/protein.ff14sbonlysc.xml"
    fixer = PDBFixer(filename=input_pdb)
    
    print(f"adding hydrogens and setting protonation state for ph={ph}...")
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms() 
    fixer.addMissingHydrogens(ph) # adds h's and adjusts protonation/charges based on ph

    PDBFile.writeFile(fixer.topology, fixer.positions, open(output_pdb_file, 'w'))

    print(f"Structure written to {output_pdb_file}. It now includes all hydrogens.")
