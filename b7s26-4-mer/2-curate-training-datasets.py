import functools
import json
import logging
import multiprocessing
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import click
import numpy
import tqdm
from openff.qcsubmit.results import (
    OptimizationResult,
    OptimizationResultCollection,
    TorsionDriveResult,
    TorsionDriveResultCollection,
)
from openff.qcsubmit.results.filters import (
    ConformerRMSDFilter,
    ConnectivityFilter,
    ElementFilter,
    HydrogenBondFilter,
    RecordStatusFilter,
    SinglepointRecordFilter,
    UnperceivableStereoFilter,
)
from openff.toolkit import ForceField, Molecule, ToolkitRegistry, RDKitToolkitWrapper
from openff.toolkit.utils import toolkit_registry_manager
from openff.toolkit.utils.exceptions import UnassignedMoleculeChargeException
from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper
from qcportal import PortalClient
from qcportal.optimization import OptimizationRecord
from qcportal.record_models import RecordStatusEnum
from qcportal.torsiondrive import TorsiondriveRecord

# Union types for type hints
QCPortalRecord = OptimizationRecord | TorsiondriveRecord
QCSubmitResult = OptimizationResult | TorsionDriveResult
QCSubmitResultCollection = OptimizationResultCollection | TorsionDriveResultCollection


class NAGLChargeCheckFilter(SinglepointRecordFilter):
    def _filter_function(
        self,
        result: QCSubmitResult,
        record: QCPortalRecord,
        molecule: Molecule,
    ) -> bool:
        # Some of the molecules fail charging with am1bccelf10 either
        # because of no bccs or failed conformer generation, sometimes it
        # cannot be captured with just the cmiles present in the record
        # metadata, so reading from file and checking it
        can_be_charged = True
        try:
            with NamedTemporaryFile(suffix=".sdf") as tmp_file:
                molecule.to_file(tmp_file.name, "SDF")
                molecule = Molecule.from_file(tmp_file.name)
                with toolkit_registry_manager(
                    ToolkitRegistry([RDKitToolkitWrapper, NAGLToolkitWrapper])
                ):
                    molecule.assign_partial_charges(partial_charge_method="openff-gnn-am1bcc-0.1.0-rc.2.pt")

        except (UnassignedMoleculeChargeException, ValueError):
            can_be_charged = False

        return can_be_charged


def check_torsion_is_in_ring(
    molecule: Molecule,
    indices: tuple[int, int, int, int],
) -> bool:
    """
    Check if a torsion is in a ring. If a torsion I-J-K-L is given, it checks
    whether all bonds I-J, J-K, and K-L are in a ring.
    """
    i, j, k, l = indices
    return (
        molecule.get_bond_between(i, j).is_in_ring()
        and molecule.get_bond_between(j, k).is_in_ring()
        and molecule.get_bond_between(k, l).is_in_ring()
    )


def label_and_tag_ids(
    record_and_molecule: tuple[QCPortalRecord, Molecule],
    force_field: ForceField,
    parameter_types: list[str],
    explicit_ring_torsion_path: str | None = None,
) -> set[tuple[str, str, int]]:
    if explicit_ring_torsion_path is None:
        explicit_ring_torsions = []
    else:
        explicit_ring_torsions = numpy.loadtxt(explicit_ring_torsion_path, dtype=str)

    record, molecule = record_and_molecule
    mol_labels = force_field.label_molecules(molecule.to_topology())[0]
    parameter_ids = set()

    for parameter_type in parameter_types:
        parameter_labels = mol_labels[parameter_type]

        for parameter_indices, parameter in parameter_labels.items():
            # remove mismatching torsiondrives
            if isinstance(record, TorsiondriveRecord):
                # check central bond of driven torsions, i.e. middle 2 atoms
                record_indices = [
                    set(dihedral_indices[1:3])
                    for dihedral_indices in record.specification.keywords.dihedrals
                ]
                if set(parameter_indices[1:3]) not in record_indices:
                    continue

                # some general parameters overlap with in-ring torsions and
                # there are many torsion scans from Gen1 sets that have
                # in-ring torsions and we want to exclude them in training
                # as they result in higher k values unless the parameters
                # have smirks explicitly for an in-ring torsion. It is to be
                # noted that training on in-ring torsions is needed to
                # properly model puckering in rings with hetero atoms
                if parameter.id not in explicit_ring_torsions:
                    if check_torsion_is_in_ring(molecule, parameter_indices):
                        continue

            n_heavy_atoms = sum(1 for atom in molecule.atoms if atom.atomic_number != 1)
            parameter_ids.add((parameter.id, record.id, n_heavy_atoms))

    return parameter_ids


def get_parameter_distribution(
    dataset: QCSubmitResultCollection,
    parameter_types: list[str],
    force_field: ForceField,
    explicit_ring_torsion_path: str | None = None,
    n_processes: int = 8,
) -> tuple[Counter, dict[str, list[tuple[int, str]]]]:
    coverage = Counter()
    parameter_records = defaultdict(list)

    func = functools.partial(
        label_and_tag_ids,
        force_field=force_field,
        parameter_types=parameter_types,
        explicit_ring_torsion_path=explicit_ring_torsion_path,
    )
    with multiprocessing.Pool(n_processes) as pool:
        for parameter_ids in tqdm.tqdm(
            pool.imap(func, dataset.to_records()),
            total=dataset.n_results,
        ):
            for parameter_id, record_id, n_heavy_atoms in parameter_ids:
                coverage[parameter_id] += 1
                parameter_records[parameter_id].append((n_heavy_atoms, record_id))

    return coverage, dict(parameter_records)


def cap_torsions_per_parameter(
    force_field: ForceField,
    dataset: TorsionDriveResultCollection,
    cap_size: int = 5,
    explicit_ring_torsion_path: str | None = None,
    method: Literal["pick_random", "pick_heavy", "pick_light"] = "pick_random",
    verbose: bool = True,
    n_processes: int = 8,
):
    coverage, parameter_records = get_parameter_distribution(
        dataset=dataset,
        parameter_types=["ProperTorsions"],
        force_field=force_field,
        explicit_ring_torsion_path=explicit_ring_torsion_path,
        n_processes=n_processes,
    )
    records_to_keep = dict()
    for parameter_id in coverage:
        if coverage[parameter_id] <= cap_size:
            n_atom_records = parameter_records[parameter_id]
        else:
            if method == "pick_heavy":
                n_atom_records = sorted(
                    parameter_records[parameter_id], key=lambda x: x[0], reverse=True
                )[:cap_size]
            elif method == "pick_light":
                n_atom_records = sorted(
                    parameter_records[parameter_id], key=lambda x: x[0], reverse=False
                )[:cap_size]
            elif method == "pick_random":
                n_atom_records = random.sample(
                    parameter_records[parameter_id], cap_size
                )

        _, records = zip(*n_atom_records)
        records_to_keep[parameter_id] = records

    if verbose:
        print("Final coverage")
        for parameter_id, records in records_to_keep.items():
            print(
                f"{parameter_id:>6s}: {len(records):>4d} "
                f"/ {coverage[parameter_id]:>4d} records"
            )

    ids_to_keep = [
        record_id for record_ids in records_to_keep.values() for record_id in record_ids
    ]
    print(f"Total records: {dataset.n_results}")
    print(f"Total records to keep: {len(ids_to_keep)}")

    client_address = list(dataset.entries.keys())[0]
    dataset.entries[client_address] = [
        record
        for record in dataset.entries[client_address]
        if record.record_id in ids_to_keep
    ]

    return dataset


def download_and_filter_td_data(
    torsiondrive_datasets: list[str],
    torsiondrive_records_to_remove_path: str | None = None,
    include_iodine: bool = False,
    filter_incomplete_records=True,
    filter_hydrogen_bonds=True,
) -> TorsionDriveResultCollection:
    """Download and filter torsiondrive datasets."""

    if torsiondrive_records_to_remove_path is None:
        torsiondrive_records_to_remove = []
    else:
        torsiondrive_records_to_remove = numpy.loadtxt(
            torsiondrive_records_to_remove_path, dtype=int
        )

    client = PortalClient("api.qcarchive.molssi.org:443")

    if filter_incomplete_records:
        dataset = TorsionDriveResultCollection.from_server(
            client=client,
            datasets=torsiondrive_datasets,
            spec_name="default",
        )
    else:
        qcportal_datasets = [
            client.get_dataset("Torsiondrive", dataset_name)
            for dataset_name in torsiondrive_datasets
        ]
        result_records = defaultdict(dict)
        for dataset in qcportal_datasets:
            dataset.fetch_entries()
            for entry_name, spec_name, record in dataset.iterate_records(
                specification_names="default",  # status=RecordStatusEnum.complete
            ):
                entry = dataset.get_entry(entry_name)
                cmiles = entry.attributes[
                    "canonical_isomeric_explicit_hydrogen_mapped_smiles"
                ]
                inchi_key = entry.attributes.get("fixed_hydrogen_inchi_key")
                if inchi_key is None:
                    tmp_mol = Molecule.from_mapped_smiles(
                        cmiles, allow_undefined_stereo=True
                    )
                    inchi_key = tmp_mol.to_inchikey(fixed_hydrogens=True)
                td_rec = TorsionDriveResult(
                    record_id=record.id, cmiles=cmiles, inchi_key=inchi_key
                )
                result_records[dataset._client.address][record.id] = td_rec
        dataset = TorsionDriveResultCollection(
            entries={
                address: [*entries.values()]
                for address, entries in result_records.items()
            }
        )

    # filter out entries to remove
    # client.address is just the key to use to access entries
    dataset.entries[client.address] = [
        entry
        for entry in dataset.entries[client.address]
        if entry.record_id not in torsiondrive_records_to_remove
    ]

    # in a number of datasets the iodine-containing molecules
    # were tainted due to an auxiliary basis set issue
    # This has since been resolved and entries have been recomputed
    # in new datasets, but we still need to filter the old ones
    elements = ["H", "C", "N", "O", "S", "P", "F", "Cl", "Br"]
    if include_iodine:
        elements.append("I")

    # Filter out other unsuitable entries
    dataset_filters = [
        ElementFilter(allowed_elements=elements),
        UnperceivableStereoFilter(),
        ConnectivityFilter(tolerance=1.2),
    ]
    if filter_incomplete_records:
        dataset_filters.append(RecordStatusFilter(status=RecordStatusEnum.complete))
    if filter_hydrogen_bonds:
        dataset_filters.append(HydrogenBondFilter(method="baker-hubbard"))
    dataset = dataset.filter(*dataset_filters)

    return dataset


def select_parameters(
    dataset: QCSubmitResultCollection,
    parameter_types: list[str],
    force_field: ForceField,
    explicit_ring_torsion_path: str | None=None,
    n_processes: int=8,
    min_coverage: int=5,
    exempt_protein_coverage: bool=False,
):
    coverage, _ = get_parameter_distribution(
        dataset=dataset,
        parameter_types=parameter_types,
        force_field=force_field,
        explicit_ring_torsion_path=explicit_ring_torsion_path,
        n_processes=n_processes,
    )

    selected_parameters = defaultdict(list)
    for parameter_type in parameter_types:
        print(f"Coverage for {parameter_type}")
        handler = force_field.get_parameter_handler(parameter_type)

        for parameter_id, count in coverage.items():
            parameters = handler.get_parameter({"id": parameter_id})
            if not len(parameters):
                continue

            if count < min_coverage and not (
                exempt_protein_coverage and "Protein" in parameter_id
            ):
                print(f"Skipping {parameter_id} with count {count}")
                continue

            print(f"Fitting {parameter_id} with count {count}")
            selected_parameters[parameter_type].append(parameters[0].smirks)

    return selected_parameters


@click.group()
def cli():
    pass


@cli.command("download-torsiondrive")
@click.option(
    "--output-dataset-path",
    "output_dataset_path",
    required=True,
    type=click.Path(exists=False, dir_okay=False, file_okay=True),
    help="The path to write the dataset to. Should be a JSON",
)
@click.option(
    "--output-smirks-path",
    "output_smirks_path",
    required=True,
    type=click.Path(exists=False, dir_okay=False, file_okay=True),
    help="The path to write the dataset to. Should be a JSON",
)
@click.option(
    "--core-td-dataset",
    "core_td_datasets",
    multiple=True,
    required=True,
    type=str,
    help="The name of a torsiondrive dataset to download.",
)
@click.option(
    "--aux-td-dataset",
    "aux_td_datasets",
    multiple=True,
    required=True,
    type=str,
    help="The name of a torsiondrive dataset to download.",
)
@click.option(
    "--protein-td-dataset",
    "protein_td_datasets",
    multiple=True,
    required=True,
    type=str,
    help="The name of a torsiondrive dataset to download.",
)
@click.option(
    "--initial-force-field",
    "initial_force_field",
    required=True,
    type=str,
    help=(
        "The name of the initial force field to use. "
        "Alternatively, the path to a force field."
    ),
)
@click.option(
    "--explicit-ring-torsions",
    "explicit_ring_torsions",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    help=(
        "The path to a file containing a list of parameter IDs that are ring torsions. "
        "This should be a text file with one ID per line."
    ),
)
@click.option(
    "--td-records-to-remove",
    "td_records_to_remove",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    help=(
        "The path to a file containing a list of record IDs to remove. "
        "This should be a text file with one record ID per line."
    ),
)
@click.option(
    "--additional-td-records",
    "additional_td_records",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    help=(
        "The path to a file containing a TorsionDriveResultCollection "
        "containing additional torsiondrive records to include. "
        "This should be a JSON file."
    ),
)
@click.option(
    "--cap-size",
    "cap_size",
    type=int,
    default=5,
    show_default=True,
    help=(
        "The maximum number of torsions to include per parameter "
        "in the auxiliary datasets."
        "If there are more torsions than this, a subset will be selected."
    ),
)
@click.option(
    "--cap-method",
    "cap_method",
    type=click.Choice(["pick_random", "pick_heavy", "pick_light"]),
    default="pick_random",
    show_default=True,
    help=(
        "The method to use to select the torsions to include per parameter "
        "in the auxiliary datasets."
    ),
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    show_default=True,
    help="Whether to print out additional information.",
)
@click.option(
    "--select-parameters-only",
    "select_parameters_only",
    is_flag=True,
    default=False,
    show_default=True,
    help="Read an existing local dataset and select parameters.",
)
@click.option(
    "--n-processes",
    "n_processes",
    type=int,
    default=8,
    show_default=True,
    help="The number of processes to use when processing the data.",
)
@click.option(
    "--min-record-coverage",
    "min_record_coverage",
    type=int,
    default=5,
    show_default=True,
    help=(
        "The minimum number of records a parameter must have to be included in the "
        "force field optimization."
    ),
)
def download_torsiondrive(
    output_dataset_path: str,
    output_smirks_path: str,
    core_td_datasets: list[str],
    protein_td_datasets: list[str],
    aux_td_datasets: list[str],
    initial_force_field: str,
    explicit_ring_torsions: str | None=None,
    td_records_to_remove: str | None=None,
    additional_td_records: str | None=None,
    cap_size: int=5,
    cap_method: Literal[
        "pick_random", "pick_heavy", "pick_light"
    ] = "pick_random",
    verbose: bool = True,
    select_parameters_only=False,
    n_processes: int=8,
    min_record_coverage: int=5,
):
    """
    Download TorsionDrive data in stages.
    \f

    1. Download the core datasets and filter out unsuitable entries.
    2. Download the auxiliary datasets and filter out unsuitable entries.
    3. Cap the number of auxiliary torsions per parameter to a maximum of ``cap_size``.
       This can be done by picking random torsions, or selecting those
       with the least (``pick_light``) or most (``pick_heavy``) heavy atoms.
    4. Add additional torsiondrive records from a file.
    5. Filter out duplicate torsiondrive records.
    6. Filter out molecules that fail AM1-BCC ELF10 charging. This step
       is the slowest so it's done last.

    The unsuitability filters are:
        - removing incomplete entries
        - removing entries with hydrogen bonds
        - removing entries with unperceivable stereochemistry
        - removing entries with connectivity rearrangements
        - removing entries with iodine

    Parameters
    ----------
    core_td_datasets
        The core torsiondrive datasets to download.
        These are filtered for unsuitability, but are not capped.
    aux_td_datasets
        The auxiliary torsiondrive datasets to download.
        These are filtered for unsuitability, and are capped
        to a certain number of torsions per parameter.
    protein_td_datasets
        The protein torsiondrive datasets to download.
        These are filtered for unsuitability, but incomplete and hydrogen
        bonded records are retained.
    initial_forcefield
        The initial force field to use for filtering torsiondrive entries.
    explicit_ring_torsions
        A file containing a list of parameter IDs for explicit ring torsions.
    td_records_to_remove
        A file containing a list of torsiondrive record IDs to remove.
    additional_td_records
        A file containing a list of additional torsiondrive records to add.
        This should be a JSON file of a ``TorsionDriveResultCollection``.
    cap_size
        The maximum number of torsions to keep per parameter.
    cap_method
        The method to use to cap the number of torsions per parameter.
        One of ``pick_random``, ``pick_heavy``, or ``pick_light``.
    verbose
        Whether to print out information about the number of records
        at each stage.
    n_processes
        The number of processes to use for multiprocessing.
    """

    # Suppress stereochemistry warnings
    logging.getLogger("openff").setLevel(logging.ERROR)

    ff = ForceField(initial_force_field, allow_cosmetic_attributes=True)

    if not select_parameters_only:
        # Core TorsionDrives
        core_dataset = download_and_filter_td_data(
            core_td_datasets,
            td_records_to_remove,
            include_iodine=False,
        )
        if verbose:
            print(f"Number of core entries: {core_dataset.n_results}")

        # Protein TorsionDrives
        protein_dataset = download_and_filter_td_data(
            protein_td_datasets,
            td_records_to_remove,
            include_iodine=False,
            filter_incomplete_records=False,
            filter_hydrogen_bonds=False,
        )
        if verbose:
            print(f"Number of protein entries: {protein_dataset.n_results}")

        # Auxiliary TorsionDrives
        aux_dataset = download_and_filter_td_data(
            aux_td_datasets,
            td_records_to_remove,
            include_iodine=False,
        )
        aux_dataset = cap_torsions_per_parameter(
            ff,
            aux_dataset,
            cap_size=cap_size,
            method=cap_method,
            explicit_ring_torsion_path=explicit_ring_torsions,
            verbose=verbose,
            n_processes=n_processes,
        )
        if verbose:
            print(f"Number of auxiliary entries: {aux_dataset.n_results}")

        if additional_td_records is not None:
            additional_records = list(
                TorsionDriveResultCollection.parse_file(
                    additional_td_records
                ).entries.values()
            )[0]
        else:
            additional_records = []

        key = list(core_dataset.entries.keys())[0]
        all_entries = (
            core_dataset.entries[key]
            + aux_dataset.entries[key]
            + protein_dataset.entries[key]
            + additional_records
        )

        if verbose:
            print(f"Number of total entries: {len(all_entries)}")

        # filter in case we have doubled up records
        unique_entries = {record.record_id: record for record in all_entries}
        new_dataset = TorsionDriveResultCollection(
            entries={key: list(unique_entries.values())}
        )
        if verbose:
            print(f"Number of entries after deduplication: {new_dataset.n_results}")

        # Filter molecules that can't be assigned partial charges
        filtered_for_charge = new_dataset.filter(NAGLChargeCheckFilter())

        if verbose:
            print(f"Number of entries after charge filter: {filtered_for_charge.n_results}")

        with open(output_dataset_path, "w") as json_file:
            json_file.write(filtered_for_charge.json(indent=2))
        if verbose:
            print(f"Saved to {output_dataset_path}")

    else:
        filtered_for_charge = TorsionDriveResultCollection.parse_file(output_dataset_path)

    selected_parameters = select_parameters(
        filtered_for_charge,
        ["ProperTorsions"],
        force_field=ff,
        explicit_ring_torsion_path=explicit_ring_torsions,
        n_processes=n_processes,
        min_coverage=min_record_coverage,
        exempt_protein_coverage=True,
    )
    with open(output_smirks_path, "w") as json_file:
        json.dump(selected_parameters, json_file, indent=2)


def download_and_filter_opt_data(
    optimization_datasets: list[str],
    optimization_records_to_remove: str | None = None,
    include_iodine: bool = False,
    max_opt_conformers: int = 12,
    verbose: bool = False,
) -> "OptimizationResultCollection":
    """Download and filter optimization datasets."""

    if optimization_records_to_remove is None:
        optimization_records_to_remove = []
    else:
        optimization_records_to_remove = numpy.loadtxt(
            optimization_records_to_remove, dtype=int
        )

    client = PortalClient("api.qcarchive.molssi.org:443")
    dataset = OptimizationResultCollection.from_server(
        client=client,
        datasets=optimization_datasets,
        spec_name="default",
    )
    if verbose:
        print(f"Number of entries before filtering: {dataset.n_results}")

    # filter out entries to remove
    # client.address is just the key to use to access entries
    dataset.entries[client.address] = [
        entry
        for entry in dataset.entries[client.address]
        if entry.record_id not in optimization_records_to_remove
    ]

    # in a number of datasets the iodine-containing molecules
    # were tainted due to an auxiliary basis set issue
    # This has since been resolved and entries have been recomputed
    # in new datasets, but we still need to filter the old ones
    elements = ["H", "C", "N", "O", "S", "P", "F", "Cl", "Br"]
    if include_iodine:
        elements.append("I")

    # filter out other unsuitable entries
    dataset = dataset.filter(
        RecordStatusFilter(status=RecordStatusEnum.complete),
        ConnectivityFilter(tolerance=1.2),
        UnperceivableStereoFilter(),
        ElementFilter(allowed_elements=elements),
        #ConformerRMSDFilter(max_conformers=max_opt_conformers),
    )
    return dataset


@cli.command("download-optimization")
@click.option(
    "--output-dataset-path",
    "output_dataset_path",
    required=True,
    type=click.Path(exists=False, dir_okay=False, file_okay=True),
    help="The path to write the dataset to. Should be a JSON",
)
@click.option(
    "--output-smirks-path",
    "output_smirks_path",
    required=True,
    type=click.Path(exists=False, dir_okay=False, file_okay=True),
    help="The path to write the dataset to. Should be a JSON",
)
@click.option(
    "--initial-force-field",
    "initial_force_field",
    required=True,
    type=str,
    help=(
        "The name of the initial force field to use. "
        "Alternatively, the path to a force field"
    ),
)
@click.option(
    "--core-opt-dataset",
    "core_opt_datasets",
    multiple=True,
    required=True,
    type=str,
    help=(
        "The name of an optimization dataset to download. "
        "These will have iodine molecules filtered out."
    ),
)
@click.option(
    "--opt-records-to-remove",
    "opt_records_to_remove",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    help=(
        "The path to a file containing a list of record IDs to remove. "
        "This should be a text file with one record ID per line."
    ),
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    show_default=True,
    help="Whether to print out additional information.",
)
@click.option(
    "--select-parameters-only",
    "select_parameters_only",
    is_flag=True,
    default=False,
    show_default=True,
    help="Read an existing local dataset and select parameters.",
)
@click.option(
    "--max-opt-conformers",
    "max_opt_conformers",
    default=12,
    show_default=True,
    type=int,
    help="The maximum number of conformers to keep per molecule.",
)
@click.option(
    "--n-processes",
    "n_processes",
    type=int,
    default=4,
    show_default=True,
    help="The number of processes to use when processing the data.",
)
@click.option(
    "--min-record-coverage",
    "min_record_coverage",
    type=int,
    default=5,
    show_default=True,
    help=(
        "The minimum number of records a parameter must have to be included in the "
        "force field optimization."
    ),
)
def download_optimization(
    output_dataset_path: str,
    output_smirks_path: str,
    initial_force_field: str,
    core_opt_datasets: list[str],
    opt_records_to_remove: str | None=None,
    max_opt_conformers: int=12,
    verbose: bool=True,
    select_parameters_only=False,
    n_processes: int=8,
    min_record_coverage: int=5,
):
    """Download and filter optimization datasets.

    \f
    Parameters
    ----------
    core_opt_datasets
        The core optimization datasets to download.
    opt_records_to_remove
        A file containing a list of optimization record IDs to remove.
    max_opt_conformers
        The maximum number of conformers to keep per molecule.
        Conformers are filled using a greedy RMSD filter
    """

    # suppress stereochemistry warnings
    logging.getLogger("openff").setLevel(logging.ERROR)

    ff = ForceField(initial_force_field, allow_cosmetic_attributes=True)

    if not select_parameters_only:
        core_dataset = download_and_filter_opt_data(
            core_opt_datasets,
            opt_records_to_remove,
            include_iodine=False,
            max_opt_conformers=max_opt_conformers,
        )
        if verbose:
            print(f"Number of filtered core entries: {core_dataset.n_results}")

        key = list(core_dataset.entries.keys())[0]
        all_entries = core_dataset.entries[key]

        # filter in case we have doubled up records
        unique_entries = {record.record_id: record for record in all_entries}
        new_dataset = OptimizationResultCollection(
            entries={key: list(unique_entries.values())}
        )

        if verbose:
            print(f"Number of entries after deduplication: {new_dataset.n_results}")

        # Filter molecules that can't be assigned partial charges
        filtered_for_charge = new_dataset.filter(NAGLChargeCheckFilter())

        if verbose:
            print(f"Number of entries after charge filter: {filtered_for_charge.n_results}")

        with open(output_dataset_path, "w") as json_file:
            json_file.write(filtered_for_charge.json(indent=2))
        if verbose:
            print(f"Saved to {output_dataset_path}")

    else:
        filtered_for_charge = OptimizationResultCollection.parse_file(output_dataset_path)

    selected_parameters = select_parameters(
        filtered_for_charge,
        ["Bonds", "Angles"],
        force_field=ff,
        n_processes=n_processes,
        min_coverage=min_record_coverage,
    )
    with open(output_smirks_path, "w") as json_file:
        json.dump(selected_parameters, json_file, indent=2)


if __name__ == "__main__":
    cli()
