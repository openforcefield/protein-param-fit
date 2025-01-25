import json
from collections import defaultdict
from pathlib import Path

import click
from openff.qcsubmit.results import TorsionDriveResultCollection
from openff.qcsubmit.results.filters import RecordStatusFilter
from openff.toolkit import Molecule
from qcportal import PortalClient
from qcportal.record_models import RecordStatusEnum


@click.command()
@click.option(
    "-d",
    "--dataset_dir",
    default="validation-datasets",
    show_default=True,
    type=click.STRING,
    help="Directory path to which validation datasets will be written.",
)
def main(dataset_dir):
    client = PortalClient("api.qcarchive.molssi.org:443")

    protein_datasets = [
        client.get_dataset("Torsiondrive", dataset_name)
        for dataset_name in [
            "OpenFF Protein Capped 3-mer Backbones v1.0",
        ]
    ]

    torsiondrive_dataset = TorsionDriveResultCollection.from_datasets(
        datasets=protein_datasets,
        spec_name="default",
    )

    # Hack to avoid filtering incomplete datasets from
    # TorsionDriveResultCollection.from_datasets()
    # from openff.qcsubmit.results import TorsionDriveResult
    # result_records = defaultdict(dict)
    # for dataset in protein_datasets:
    #    dataset.fetch_entries()
    #    for entry_name, spec_name, record in dataset.iterate_records(
    #        specification_names="default",  # status=RecordStatusEnum.complete
    #    ):
    #        entry = dataset.get_entry(entry_name)
    #        cmiles = entry.attributes[
    #            "canonical_isomeric_explicit_hydrogen_mapped_smiles"
    #        ]
    #        inchi_key = entry.attributes.get("fixed_hydrogen_inchi_key")
    #        if inchi_key is None:
    #            tmp_mol = Molecule.from_mapped_smiles(
    #                cmiles, allow_undefined_stereo=True
    #            )
    #            inchi_key = tmp_mol.to_inchikey(fixed_hydrogens=True)
    #        td_rec = TorsionDriveResult(
    #            record_id=record.id, cmiles=cmiles, inchi_key=inchi_key
    #        )
    #        result_records[dataset._client.address][record.id] = td_rec
    # torsiondrive_dataset = TorsionDriveResultCollection(
    #    entries={
    #        address: [*entries.values()]
    #        for address, entries in result_records.items()
    #    }
    # )

    # Filter incomplete TorsionDrive results
    torsiondrive_dataset = torsiondrive_dataset.filter(
        RecordStatusFilter(status=RecordStatusEnum.complete)
    )

    # Write dict of IDs to names to file
    dataset_names = dict()
    for dataset in protein_datasets:
        dataset.fetch_entries()
        for entry_name, spec_name, record in dataset.iterate_records(
            specification_names="default",
        ):
            dataset_names[record.id] = entry_name

    with open(
        Path(dataset_dir, "torsiondrive-validation-names.json"), "w"
    ) as json_file:
        json.dump(dataset_names, json_file)

    # Write validation dataset to file
    with open(
        Path(dataset_dir, "torsiondrive-validation-dataset.json"), "w"
    ) as json_file:
        json_file.write(torsiondrive_dataset.json())


if __name__ == "__main__":
    main()
