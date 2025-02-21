import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import click
import numpy
import openmm
from openff.qcsubmit.results import TorsionDriveResultCollection
from openff.toolkit import ForceField, Molecule, ToolkitRegistry, RDKitToolkitWrapper
from openff.toolkit.utils import toolkit_registry_manager
from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper
from openff.units import unit as offunit
from openmm import app, unit
from openmm.app import ForceField as OMMForceField
from openmmforcefields.generators import GAFFTemplateGenerator
from qcportal.torsiondrive import TorsiondriveRecord
from rdkit.Chem import rdMolAlign
from rdkit.Geometry import Point3D
from tqdm import tqdm


def _parameterize_molecule(
    mapped_smiles: str,
    force_field: ForceField | OMMForceField,
    force_field_type: str,
) -> openmm.System:
    """Parameterize a particular molecules with a specified force field."""

    offmol = Molecule.from_mapped_smiles(mapped_smiles)

    if force_field_type.lower() == "smirnoff":
        return force_field.create_openmm_system(offmol.to_topology())

    elif force_field_type.lower() == "smirnoff-nagl":
        with toolkit_registry_manager(
            ToolkitRegistry([RDKitToolkitWrapper, NAGLToolkitWrapper])
        ):
            openmm_system = force_field.create_openmm_system(offmol.to_topology())

        return openmm_system

    elif force_field_type.lower() == "smirnoff-am1bcc":
        offmol.assign_partial_charges(partial_charge_method="am1bcc")
        return force_field.create_openmm_system(
            offmol.to_topology(), charge_from_molecules=[offmol]
        )

    elif force_field_type.lower() == "amber":
        # Get residue information for Amber biopolymer force field
        offmol.perceive_residues()

        return force_field.createSystem(
            offmol.to_topology().to_openmm(),
            nonbondedCutoff=0.9 * unit.nanometer,
            switchDistance=0.8 * unit.nanometer,
            constraints=None,
        )

    elif force_field_type.lower() == "gaff":
        return force_field.createSystem(
            offmol.to_topology().to_openmm(),
            nonbondedCutoff=0.9 * unit.nanometer,
            switchDistance=0.8 * unit.nanometer,
            constraints=None,
        )

    raise NotImplementedError(
        "Only SMIRNOFF, Amber, and GAFF force fields are currently supported."
    )


def _evaluate_energy(openmm_system: openmm.System, coordinates: unit.Quantity) -> float:
    """Returns the evaluated energy of the conformer in kcal / mol."""

    integrator = openmm.VerletIntegrator(0.001 * unit.femtoseconds)

    platform = openmm.Platform.getPlatformByName("Reference")
    openmm_context = openmm.Context(openmm_system, integrator, platform)

    openmm_context.setPositions(coordinates.value_in_unit(unit.nanometers))
    state = openmm_context.getState(getEnergy=True)

    potential_energy = state.getPotentialEnergy()

    return potential_energy.value_in_unit(unit.kilocalories_per_mole)


def _minimise_structure(
    openmm_system: openmm.System,
    coordinates: unit.Quantity,
    fixed_indices: Tuple[int, ...],
) -> unit.Quantity:
    """Minimizes a set of coordinates using a specified potential energy function."""

    openmm_system = copy.deepcopy(openmm_system)

    # Constrain the fixed atoms
    for index in fixed_indices:
        openmm_system.setParticleMass(index, 0.0)

    integrator = openmm.VerletIntegrator(0.001 * unit.femtoseconds)

    platform = openmm.Platform.getPlatformByName("Reference")

    openmm_context = openmm.Context(openmm_system, integrator, platform)
    openmm_context.setPositions(coordinates.m_as(offunit.nanometer))

    openmm.LocalEnergyMinimizer.minimize(openmm_context)

    state: openmm.State = openmm_context.getState(getPositions=True)
    minimised_coordinates = state.getPositions()

    return minimised_coordinates


def _compute_grid_energies(
    force_field: ForceField | OMMForceField,
    force_field_type: str,
    qc_record: TorsiondriveRecord,
    molecule: Molecule,
) -> Dict[Tuple[int, ...], Tuple[float, float, float, float]]:
    """Convert a QC record and its associated molecule into an OpenMM system
    and minimize the coordinates using an MM force field."""

    grid_energies = qc_record.final_energies

    grid_conformers = {
        grid_id: conformer
        for grid_id, conformer in zip(
            molecule.properties["grid_ids"], molecule.conformers
        )
    }

    grid_ids = sorted(grid_conformers, key=lambda x: x[0])

    # Get indices of driven torsions
    dihedral_indices = [
        indices if indices[2] >= indices[1] else indices[::-1]
        for indices in qc_record.specification.keywords.dihedrals
    ]

    # Get indices of non-driven torsions with constraints
    if "constraints" in qc_record.specification.optimization_specification.keywords:
        constraints = qc_record.specification.optimization_specification.keywords[
            "constraints"
        ]

        for constraint_type in ["freeze", "set"]:
            if constraint_type in constraints:
                for constraint in constraints[constraint_type]:
                    if constraint["type"] == "dihedral":
                        indices = constraint["indices"]
                        dihedral_indices.append(
                            indices if indices[2] >= indices[1] else indices[::-1]
                        )

    # Indices of atoms in driven torsions, which are fixed during minimization
    fixed_indices = tuple(int(i) for i in numpy.unique(dihedral_indices))

    openmm_system = _parameterize_molecule(
        molecule.to_smiles(isomeric=True, mapped=True),
        force_field,
        force_field_type,
    )

    rd_mol = molecule.to_rdkit()
    rd_conf = rd_mol.GetConformer()

    ref_rd_mol = copy.deepcopy(rd_mol)
    ref_rd_conf = ref_rd_mol.GetConformer()

    # Zero out the contribution of the driven torsion to evaluate a target for
    # a Fourier series
    target_system = copy.deepcopy(openmm_system)
    torsion_force = [
        force
        for force in target_system.getForces()
        if isinstance(force, openmm.PeriodicTorsionForce)
    ][0]

    for torsion_index in range(torsion_force.getNumTorsions()):
        i, j, k, l, periodicity, phase, _ = torsion_force.getTorsionParameters(
            torsion_index
        )

        if ((i, j, k, l) if k >= j else (l, k, j, i)) in dihedral_indices:
            torsion_force.setTorsionParameters(
                torsion_index, i, j, k, l, periodicity, phase, 0.0
            )

    # Evaluate the energy for each grid id
    energies: Dict[Tuple[int, ...], Tuple[float, float, float, float]] = dict()

    lowest_qm_energy = None
    lowest_qm_energy_grid_id = None

    for grid_id in grid_ids:
        coordinates = _minimise_structure(
            openmm_system, grid_conformers[grid_id], fixed_indices
        )

        qm_energy = (
            grid_energies[grid_id] * unit.hartree * unit.AVOGADRO_CONSTANT_NA
        ).value_in_unit(unit.kilocalories_per_mole)

        mm_energy = _evaluate_energy(openmm_system, coordinates)
        mm_target = _evaluate_energy(target_system, coordinates)

        # Get RMSD between MM and QM coordinates from RDKit
        ref_coords = grid_conformers[grid_id].m_as(offunit.nanometer)
        min_coords = coordinates.value_in_unit(unit.nanometer)

        for i in range(rd_mol.GetNumAtoms()):
            ref_rd_conf.SetAtomPosition(
                i, Point3D(ref_coords[i][0], ref_coords[i][1], ref_coords[i][2])
            )
            rd_conf.SetAtomPosition(
                i, Point3D(min_coords[i][0], min_coords[i][1], min_coords[i][2])
            )

        rmsd = rdMolAlign.AlignMol(rd_mol, ref_rd_mol)

        if lowest_qm_energy is None or qm_energy < lowest_qm_energy:
            lowest_qm_energy = qm_energy
            lowest_qm_energy_grid_id = grid_id

        energies[grid_id] = (qm_energy, mm_energy, mm_target, rmsd)

    energies = {
        json.dumps(grid_id): (
            qm_energy - energies[lowest_qm_energy_grid_id][0],
            mm_energy - energies[lowest_qm_energy_grid_id][1],
            (
                qm_energy
                - energies[lowest_qm_energy_grid_id][0]
                - (mm_target - energies[lowest_qm_energy_grid_id][2])
            ),
            rmsd,
        )
        for grid_id, (qm_energy, mm_energy, mm_target, rmsd) in energies.items()
    }

    return energies


@click.command()
@click.option(
    "-d",
    "--dataset_dir",
    default="validation-datasets",
    show_default=True,
    type=click.STRING,
    help="Directory path containing validation datasets.",
)
@click.option(
    "-f",
    "--force_field_path",
    type=click.STRING,
    default="openff_unconstrained-2.1.0.offxml",
    show_default=True,
    help="Name of force field or path to force field offxml.",
)
@click.option(
    "-n",
    "--force_field_label",
    type=click.STRING,
    default="Sage-2.1",
    show_default=True,
    help="Name of force field for use in labels.",
)
@click.option(
    "-t",
    "--force_field_type",
    type=click.STRING,
    default="smirnoff",
    show_default=True,
    help='Type of force field. Must be one of "smirnoff", "smirnoff-nagl",'
        '"smirnoff-am1bcc", "amber", or "gaff".',
)
def main(dataset_dir, force_field_path, force_field_label, force_field_type):
    force_field_type = force_field_type.lower()

    if force_field_type in {"smirnoff", "smirnoff-nagl", "smirnoff-am1bcc"}:
        force_field = ForceField(
            force_field_path, load_plugins=True, allow_cosmetic_attributes=True
        )

        if "Constraints" in force_field.registered_parameter_handlers:
            force_field.deregister_parameter_handler("Constraints")

    elif force_field_type == "amber":
        force_field = OMMForceField(force_field_path)

    elif force_field_type == "gaff":
        force_field = OMMForceField()
        gaff_generator = GAFFTemplateGenerator(
            molecules=off_molecule, forcefield=force_field_path
        )
        force_field.registerTemplateGenerator(generator.gaff_generator)

    else:
        raise NotImplementedError(
            'Argument "force_field_type" must be one of\n    smirnoff'
            "\n    smirnoff-nagl\n    smirnoff-am1bcc\n    amber\n    gaff"
        )

    torsiondrive_dataset = TorsionDriveResultCollection.parse_file(
        Path(dataset_dir, "torsiondrive-validation-dataset.json")
    )

    records_and_molecules = torsiondrive_dataset.to_records()

    qc_data = defaultdict(dict)

    for qc_record, offmol in tqdm(records_and_molecules):
        if len(offmol.properties["grid_ids"]) == 0:
            continue

        # Track which SMILES (with the driven dihedral tagged) corresponds to
        # which record id.
        dihedral_indices = numpy.unique(qc_record.specification.keywords.dihedrals)
        offmol_copy = copy.deepcopy(offmol)
        offmol_copy.properties["atom_map"] = {
            j: i + 1 for i, j in enumerate(dihedral_indices)
        }
        qc_data[qc_record.id]["smiles"] = offmol_copy.to_smiles(mapped=True)

        # Compute QM energy, minimized MM energy, MM target, and RMSD
        qc_data[qc_record.id]["energies"] = _compute_grid_energies(
            force_field,
            force_field_type,
            qc_record=qc_record,
            molecule=offmol,
        )

    output_path = Path(dataset_dir, f"torsiondrive-{force_field_label}-minimized.json")
    with open(output_path, "w") as output_file:
        json.dump(qc_data, output_file)


if __name__ == "__main__":
    main()
