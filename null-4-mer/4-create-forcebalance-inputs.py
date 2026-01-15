import copy
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple, Union
from typing_extensions import Literal

import click
from openff.bespokefit._pydantic import Field
from openff.bespokefit.optimizers.forcebalance import ForceBalanceInputFactory
from openff.bespokefit.optimizers.forcebalance.factories import (
    _TargetFactory,
    AbInitioTargetFactory,
    ForceBalanceInputFactory,
    OptGeoTargetFactory,
    TorsionProfileTargetFactory,
)
from openff.bespokefit.optimizers.forcebalance.templates import (
    AbInitioTargetTemplate,
    InputOptionsTemplate,
    TorsionProfileTargetTemplate,
    OptGeoTargetTemplate,
)
from openff.bespokefit.schema.data import BespokeQCData, LocalQCData
from openff.bespokefit.schema.fitting import OptimizationSchema, OptimizationStageSchema
from openff.bespokefit.schema.optimizers import ForceBalanceSchema
from openff.bespokefit.schema.smirnoff import (
    AngleHyperparameters,
    AngleSMIRKS,
    BondHyperparameters,
    ImproperTorsionHyperparameters,
    ProperTorsionHyperparameters,
    BondSMIRKS,
    ProperTorsionSMIRKS,
)
from openff.bespokefit.schema.targets import (
    BaseTargetSchema,
    AbInitioTargetSchema,
    OptGeoTargetSchema,
    TorsionProfileTargetSchema,
)
from openff.bespokefit.schema.tasks import OptimizationTaskSpec
from openff.bespokefit.utilities.tempcd import temporary_cd
from openff.qcsubmit.results import (
    OptimizationResultCollection,
    TorsionDriveResultCollection,
)
from openff.qcsubmit.results.filters import SMARTSFilter, SMILESFilter
from openff.toolkit import ForceField
from openff.toolkit import Molecule as OFFMolecule
from openff.units import unit
from qcelemental.models.procedures import OptimizationResult
from qcportal.optimization import OptimizationRecord


# Define a bespokefit TargetSchema for an AbInitio target from an
# OptimizationResultCollection
class OptAbInitioTargetSchema(AbInitioTargetSchema):
    type: Literal["OptAbInitio"] = "OptAbInitio"

    reference_data: Optional[
        Union[
            LocalQCData[OptimizationResult],
            BespokeQCData[OptimizationTaskSpec],
            OptimizationResultCollection,
        ]
    ] = Field(
        None,
        description="The reference QC data (either existing or to be generated on the "
        "fly) to fit against.",
    )


# Define a custom data type that includes the OptAbInitioTargetSchema
CustomTargetSchema = Union[
    TorsionProfileTargetSchema,
    AbInitioTargetSchema,
    OptGeoTargetSchema,
    OptAbInitioTargetSchema,
]

_CUSTOM_TARGET_SECTION_TEMPLATES = {
    AbInitioTargetSchema: AbInitioTargetTemplate,
    TorsionProfileTargetSchema: TorsionProfileTargetTemplate,
    OptGeoTargetSchema: OptGeoTargetTemplate,
    OptAbInitioTargetSchema: AbInitioTargetTemplate,
}

# Define a bespokefit OptimizationStageSchema that allows targets from 
# CustomTargetSchema
class CustomOptimizationStageSchema(OptimizationStageSchema):
    targets: List[CustomTargetSchema] = Field(
        [],
        description="The fittings targets to simultaneously optimize against.",
    )


# Define a bespokefit TargetFactory for the OptAbInitioTargetSchema
class OptAbInitioTargetFactory(_TargetFactory[OptAbInitioTargetSchema]):
    @classmethod
    def _target_name_prefix(cls) -> str:
        return "ab-initio"


    @classmethod
    def _generate_targets_section(cls, target_template: CustomTargetSchema, target_names: List[str]):
        """Creates the target sections which will need to be added to the main
        ForceBalance 'options.in' file."""

        target_template = target_template.copy(deep=True)
        target_template.extras = {
            key: value
            for key, value in target_template.extras.items()
            if key not in cls._section_extras_to_exclude()
        }

        template_factory = _CUSTOM_TARGET_SECTION_TEMPLATES[target_template.__class__]
        return template_factory.generate(target_template, target_names)


    @classmethod
    def _batch_qc_records(
        cls,
        target: OptAbInitioTargetSchema,
        qc_records: List[
            Tuple[Union[OptimizationRecord, OptimizationResult], OFFMolecule]
        ],
    ):
        qc_records_by_inchikey = defaultdict(list)

        for i, (qc_record, off_molecule) in enumerate(qc_records):
            inchikey = off_molecule.to_inchikey()
            qc_records_by_inchikey[inchikey].append(i)

        batch_qc_records = [
            qc_record_list
            for qc_record_list in qc_records_by_inchikey.values()
            if len(qc_record_list) > 1
        ]

        return {
            f"{cls._target_name_prefix()}-batch-{batch_index}": [
                qc_records[record_index] for record_index in record_list
            ]
            for batch_index, record_list in enumerate(batch_qc_records)
        }


    @classmethod
    def _generate_target(
        cls,
        target: CustomTargetSchema,
        qc_records: List[
            Tuple[Union[OptimizationRecord, OptimizationResult], OFFMolecule]
        ],
    ):
        from forcebalance.molecule import Molecule as FBMolecule

        if isinstance(target, OptAbInitioTargetSchema) and target.fit_force is True:
            raise NotImplementedError()

        record_names = []
        record_conformers = []
        record_energies = []

        for i, (qc_record, off_molecule) in enumerate(qc_records):
            qc_record_id = (
                qc_record.extras["id"] if "id" in qc_record.extras else qc_record.id
            )

            record_name = f"{qc_record_id}-{i}"
            record_names.append(record_name)

            record_conformer = off_molecule.conformers[0].m_as(unit.angstrom)
            record_conformers.append(record_conformer)

            record_energy = qc_record.energies[-1]
            record_energies.append(record_energy)

        # Create a FB molecule object from the last record
        fb_molecule = FBMolecule()
        fb_molecule.Data = {
            "resname": ["UNK"] * off_molecule.n_atoms,
            "resid": [0] * off_molecule.n_atoms,
            "elem": [atom.symbol for atom in off_molecule.atoms],
            "bonds": [
                (bond.atom1_index, bond.atom2_index) for bond in off_molecule.bonds
            ],
            "name": f"{qc_record_id}",
            "xyzs": record_conformers,
            "comms": record_names,
            "qm_energies": record_energies,
        }

        # Write the data
        fb_molecule.write("qdata.txt")
        fb_molecule.write("scan.xyz")

        off_molecule = copy.deepcopy(off_molecule)
        off_molecule._conformers = [off_molecule.conformers[0]]
        off_molecule.to_file("input.sdf", "SDF")

        # OpenFF does not map molecules into PDB well (it leads to files where Br is
        # confused with B for example), so we use ForceBalance instead.
        fb_molecule.Data["xyzs"] = [fb_molecule.Data["xyzs"][0]]
        del fb_molecule.Data["qm_energies"]
        del fb_molecule.Data["comms"]
        fb_molecule.write("conf.pdb")

        metadata: dict = qc_record.specification.keywords

        metadata["energy_decrease_thresh"] = None
        metadata["energy_upper_limit"] = target.energy_cutoff

        with open("metadata.json", "w") as file:
            file.write(json.dumps(metadata))


# Define a bespokefit ForceBalanceInputFactory that can use 
class CustomForceBalanceInputFactory(ForceBalanceInputFactory):

    @classmethod
    def generate(
        cls,
        root_directory: Union[str, Path],
        schema: CustomOptimizationStageSchema,
        initial_force_field: ForceField,
    ):
        if not isinstance(schema.optimizer, ForceBalanceSchema):
            raise OptimizerError(
                "Inputs can only be generated using this factory for optimizations "
                "which use ForceBalance as the optimizer."
            )

        target_factories = {
            AbInitioTargetSchema: AbInitioTargetFactory,
            TorsionProfileTargetSchema: TorsionProfileTargetFactory,
            OptGeoTargetSchema: OptGeoTargetFactory,
            OptAbInitioTargetSchema: OptAbInitioTargetFactory,
        }

        # Create the root directory.
        os.makedirs(root_directory, exist_ok=True)

        # Temporarily switch to the root directory to make setting up the folder
        # structure easier.
        with temporary_cd(str(root_directory)):
            target_sections = []

            # Create the target directories
            os.makedirs("targets", exist_ok=True)

            with temporary_cd("targets"):
                for target in schema.targets:
                    target_factory = target_factories[target.__class__]
                    target_sections.append(target_factory.generate(".", target))

            targets_section = "\n\n".join(target_sections)

            # Create the optimize.in file
            priors = {
                name: value
                for name, value in [
                    (
                        f"{hyperparameter.type}/"
                        f"{hyperparameter.offxml_tag()}/"
                        f"{attribute}",
                        prior,
                    )
                    for hyperparameter in schema.parameter_hyperparameters
                    for attribute, prior in hyperparameter.priors.items()
                ]
            }

            with open("optimize.in", "w") as file:
                file.write(
                    InputOptionsTemplate.generate(
                        schema.optimizer, targets_section=targets_section, priors=priors
                    )
                )

            # Create the force field directory
            cls._generate_force_field_directory(schema, initial_force_field)


def load_training_data(
    abinitio_dataset: str,
    optimization_dataset: str,
    torsiondrive_dataset: str,
    smarts_to_exclude: str | None = None,
    smiles_to_exclude: str | None = None,
    verbose: bool = False
):
    if smarts_to_exclude is not None:
        exclude_smarts = Path(smarts_to_exclude).read_text().splitlines()
    else:
        exclude_smarts = []
    
    if smiles_to_exclude is not None:
        exclude_smiles = Path(smiles_to_exclude).read_text().splitlines()
    else:
        exclude_smiles = []

    torsion_training_set = TorsionDriveResultCollection.parse_file(torsiondrive_dataset)
    if verbose:
        print(f"Loaded torsion training set with {torsion_training_set.n_results} entries.")
    
    torsion_training_set = torsion_training_set.filter(
        SMARTSFilter(smarts_to_exclude=exclude_smarts),
        SMILESFilter(smiles_to_exclude=exclude_smiles),
    )
    
    if verbose:
        print(f"Filtered torsion training set to {torsion_training_set.n_results} entries.")
    
    optimization_training_set = OptimizationResultCollection.parse_file(optimization_dataset)
    if verbose:
        print(f"Loaded optimization training set with {optimization_training_set.n_results} entries.")
    optimization_training_set = optimization_training_set.filter(
        SMARTSFilter(smarts_to_exclude=exclude_smarts),
        SMILESFilter(smiles_to_exclude=exclude_smiles),
    )
    if verbose:
        print(f"Filtered optimization training set to {optimization_training_set.n_results} entries.")

    abinitio_training_set = OptimizationResultCollection.parse_file(abinitio_dataset)
    if verbose:
        print(f"Loaded AbInitio training set with {abinitio_training_set.n_results} entries.")
    abinitio_training_set = abinitio_training_set.filter(
        SMARTSFilter(smarts_to_exclude=exclude_smarts),
        SMILESFilter(smiles_to_exclude=exclude_smiles),
    )
    if verbose:
        print(f"Filtered AbInitio training set to {abinitio_training_set.n_results} entries.")

    return torsion_training_set, optimization_training_set, abinitio_training_set


@click.command()
@click.option(
    "--tag",
    type=str,
    default="forcebalance",
    help="The tag to use for the fitting run.",
)
@click.option(
    "--force-field-path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    required=True,
    help="The path to the force field to use. (offxml)",
)
@click.option(
    "--abinitio-dataset-path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    required=True,
    help="The path to the AbInitio dataset to use. (JSON)",
)
@click.option(
    "--optimization-dataset-path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    required=True,
    help="The path to the optimization dataset to use. (JSON)",
)
@click.option(
    "--torsiondrive-dataset-path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    required=True,
    help="The path to the torsion dataset to use. (JSON)",
)
@click.option(
    "--valence-smirks-path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    required=True,
    help="The path to the valence parameters to optimize (JSON).",
)
@click.option(
    "--torsion-smirks-path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    required=True,
    help="The path to the torsions to optimize (JSON).",
)
@click.option(
    "--smarts-to-exclude",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    default=None,
    help=(
        "The path to a file containing a list of SMARTS patterns "
        "to exclude from the training set. "
        "The patterns should be separated by new lines."
    ),
)
@click.option(
    "--smiles-to-exclude",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    default=None,
    help=(
        "The path to a file containing a list of SMILES patterns "
        "to exclude from the training set. "
        "The patterns should be separated by new lines."
    ),
)
@click.option(
    "--protein-record-ids-path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    default=None,
    help=(
        "The path to a file containing a list of QCFractal record IDs "
        "corresponding to protein TorsionDrives. "
        "The record IDs should be separated by new lines."
    ),
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Whether to print verbose logging messages.",
)
@click.option(
    "--max-iterations",
    type=int,
    default=50,
    show_default=True,
    help="The maximum number of iterations to run the fitting for.",
)
@click.option(
    "--abinitio-weight",
    type=float,
    default=1.0,
    show_default=True,
    help="The weight of AbInitio targets in the ForceBalance objective "
        "function.",
)
@click.option(
    "--opt-geo-weight",
    type=float,
    default=0.01,
    show_default=True,
    help="The weight of optimized geometry targets in the ForceBalance "
        "objective function.",
)
@click.option(
    "--protein-torsiondrive-weight",
    type=float,
    default=1.0,
    show_default=True,
    help="The weight of protein TorsionDrive targets in the ForceBalance "
        "objective function.",
)
@click.option(
    "--port",
    type=int,
    default=55125,
    show_default=True,
    help="The port to run the server on.",
)
@click.option(
    "--torsiondrive-target-type",
    type=str,
    default="TorsionProfile",
    show_default=True,
    help="The ForceBalance target type for TorsionDrive QC data.",
)
@click.option(
    "--torsiondrive-weight",
    type=float,
    default=1.0,
    show_default=True,
    help="The weight of TorsionDrive targets in the ForceBalance objective "
        "function.",
)
def main(
    tag: str,
    force_field_path: str,
    abinitio_dataset_path: str,
    optimization_dataset_path: str,
    torsiondrive_dataset_path: str,
    valence_smirks_path: str,
    torsion_smirks_path: str,
    smarts_to_exclude: str | None,
    smiles_to_exclude: str | None,
    protein_record_ids_path: str | None,
    verbose: bool,
    max_iterations: int,
    abinitio_weight: float,
    opt_geo_weight: float,
    protein_torsiondrive_weight: float,
    port: int,
    torsiondrive_target_type: str,
    torsiondrive_weight: float,
):
    optimizer = ForceBalanceSchema(
        max_iterations=max_iterations,
        step_convergence_threshold=0.01,
        objective_convergence_threshold=0.1,
        gradient_convergence_threshold=0.1,
        n_criteria=2,
        initial_trust_radius=-1.0,
        finite_difference_h=0.01,
        extras={
            "wq_port": str(port),
            "asynchronous": "True",
            "search_tolerance": "0.1",
            "backup": "0",
            "retain_micro_outputs": "0",
        },
    )

    # Prepare QC datasets
    torsion_training_set, optimization_training_set, abinitio_training_set = load_training_data(
        abinitio_dataset=abinitio_dataset_path,
        optimization_dataset=optimization_dataset_path,
        torsiondrive_dataset=torsiondrive_dataset_path,
        smarts_to_exclude=smarts_to_exclude,
        smiles_to_exclude=smiles_to_exclude,
        verbose=verbose
    )

    # Set up options for TorsionDrive QC data
    torsiondrive_extras = {"remote": "1"}
    if torsiondrive_target_type == "TorsionProfile":
        torsiondrive_target_schema_type = TorsionProfileTargetSchema
    elif torsiondrive_target_type == "AbInitio":
        torsiondrive_target_schema_type = AbInitioTargetSchema
        torsiondrive_extras["energy_asymmetry"] = 100.0
        torsiondrive_extras["energy_mode"] = "qm_minimum"
    else:
        raise ValueError(
            "Argument torsiondrive-target-type must be one of:"
            "\n    TorsionProfile\n    AbInitio"
        )

    # Set up TorsionDrive target schemas
    if protein_record_ids_path is None:
        torsion_profile_target_schemas = [
            torsiondrive_target_schema_type(
                reference_data=torsion_training_set,
                weight=torsiondrive_weight,
                attenuate_weights=True,
                energy_denominator=1.0,
                energy_cutoff=8.0,
                extras=torsiondrive_extras,
            )
        ]

    else:
        # Split TorsionDrive dataset into protein and small molecule datasets
        protein_record_ids = {
            int(record_id)
            for record_id in Path(protein_record_ids_path).read_text().splitlines()
        }
        client_address = list(torsion_training_set.entries.keys())[0]

        protein_entries = [
            entry
            for entry in torsion_training_set.entries[client_address]
            if entry.record_id in protein_record_ids
        ]
        small_molecule_entries = [
            entry
            for entry in torsion_training_set.entries[client_address]
            if entry.record_id not in protein_record_ids
        ]

        protein_torsion_training_set = TorsionDriveResultCollection(
            entries={client_address: protein_entries}
        )
        small_molecule_torsion_training_set = TorsionDriveResultCollection(
            entries={client_address: small_molecule_entries}
        )

        torsion_profile_target_schemas = [
            torsiondrive_target_schema_type(
                reference_data=protein_torsion_training_set,
                weight=protein_torsiondrive_weight,
                attenuate_weights=True,
                energy_denominator=1.0,
                energy_cutoff=8.0,
                extras=torsiondrive_extras,
            ),
            torsiondrive_target_schema_type(
                reference_data=small_molecule_torsion_training_set,
                weight=torsiondrive_weight,
                attenuate_weights=True,
                energy_denominator=1.0,
                energy_cutoff=8.0,
                extras=torsiondrive_extras,
            )
        ]

    targets = [
        *torsion_profile_target_schemas,
        OptGeoTargetSchema(
            reference_data=optimization_training_set,
            weight=opt_geo_weight,
            extras={"batch_size": 30, "remote": "1"},
            bond_denominator=0.05,
            angle_denominator=5.0,
            dihedral_denominator=10.0,
            improper_denominator=10.0,
        ),
        OptAbInitioTargetSchema(
            reference_data=abinitio_training_set,
            weight=abinitio_weight,
            attenuate_weights=True,
            energy_denominator=1.0,
            energy_cutoff=60.0,
            extras={
                "energy_asymmetry": 100.0,
                "energy_mode": "qm_minimum",
                "remote": "1",
            },
        ),
    ]

    # a16, a17, a27, a35
    linear_angle_smirks = [
        "[*:1]~[#6X2:2]~[*:3]",  # a16
        "[*:1]~[#7X2:2]~[*:3]",  # a17
        "[*:1]~[#7X2:2]~[#7X1:3]",  # a27
        "[*:1]=[#16X2:2]=[*:3]",
    ]  # a35, this one anyways doesn't have a training target for ages

    with open(valence_smirks_path, "r") as json_file:
        valence_smirks = json.load(json_file)
    with open(torsion_smirks_path, "r") as json_file:
        torsion_smirks = json.load(json_file)

    target_parameters = []
    for smirks in valence_smirks["Angles"]:
        if smirks in linear_angle_smirks:
            parameter = AngleSMIRKS(smirks=smirks, attributes={"k"})
        else:
            parameter = AngleSMIRKS(smirks=smirks, attributes={"k", "angle"})
        target_parameters.append(parameter)
    
    for smirks in valence_smirks["Bonds"]:
        target_parameters.append(BondSMIRKS(smirks=smirks, attributes={"k", "length"}))
    
    force_field = ForceField(force_field_path)

    torsion_handler = force_field.get_parameter_handler("ProperTorsions")
    for smirks in torsion_smirks["ProperTorsions"]:
        original_k = torsion_handler.parameters[smirks].k
        attributes = {f"k{i + 1}" for i in range(len(original_k))}
        target_parameters.append(ProperTorsionSMIRKS(smirks=smirks, attributes=attributes))

    optimization_schema = OptimizationSchema(
        id=tag,
        initial_force_field=str(Path(force_field_path).resolve()),
        stages=[
            OptimizationStageSchema(
                optimizer=optimizer,
                targets=targets,
                parameters=target_parameters,
                parameter_hyperparameters=[
                    AngleHyperparameters(priors={"k": 100, "angle": 5}),
                    BondHyperparameters(priors={"k": 100, "length": 0.1}),
                    ProperTorsionHyperparameters(priors={"k": 5}),
                    ImproperTorsionHyperparameters(priors={"k": 5}),
                ],
            )
        ]
    )

    with open(f"{optimization_schema.id}.json", "w") as json_file:
        json_file.write(optimization_schema.json(indent=2))

    # Generate the ForceBalance inputs
    CustomForceBalanceInputFactory.generate(
        optimization_schema.id,
        optimization_schema.stages[0],
        ForceField(optimization_schema.initial_force_field),
    )



if __name__ == "__main__":
    main()
