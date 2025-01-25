import json
from collections import defaultdict
from pathlib import Path

import click
import numpy
import pandas
import seaborn
from matplotlib import patches, pyplot
from openff.toolkit import Molecule
from openmm import unit

R = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilocalorie_per_mole / unit.kelvin)
PHI_SMILES = "[#6X4]-[#6X3:1](=[#8])-[#7X3:2]-[#6X4:3]-[#6X3:4](=[#8])-[#7X3]-[#6X4]"
PSI_SMILES = "[#6X4]-[#6X3](=[#8])-[#7X3:1]-[#6X4:2]-[#6X3:3](=[#8])-[#7X3:4]-[#6X4]"


# Clusters on Ramachandran map from
# Hollingsworth SA, Karplus PA. (2010). BioMol Concepts 1, 271-283.
HOLLINGSWORTH_RAMACHANDRAN_CLUSTERS = [
    {"cluster": r"$\beta$", "phi": [180, 270], "psi": [105, 195]},
    {"cluster": r"$\gamma$", "phi": [60, 105], "psi": [-90, -30]},
    {"cluster": r"$\delta$", "phi": [225, 315], "psi": [-60, 45]},
    {"cluster": r"$\alpha$", "phi": [285, 315], "psi": [-60, -30]},
    {"cluster": r"$\varepsilon$", "phi": [45, 180], "psi": [120, 240]},
    {"cluster": r"$\zeta$", "phi": [195, 255], "psi": [45, 105]},
    {"cluster": r"$P_{II}$", "phi": [270, 315], "psi": [120, 195]},
    {"cluster": r"$\gamma'$", "phi": [255, 300], "psi": [45, 105]},
    {"cluster": r"$\delta'$", "phi": [30, 120], "psi": [-15, 75]},
    {"cluster": r"$P_{II}'$", "phi": [45, 120], "psi": [165, 240]},
]


def _compute_profile_rmse(
    ref_energies: numpy.ndarray, energies: numpy.ndarray, normalize: bool, shift: bool
) -> float:
    """Compute the RMSE between two torsion profiles after optimally
    superimposing the two."""

    if shift:
        energies = energies + (ref_energies - energies).mean()

    if normalize:
        ref_energies = ref_energies / (ref_energies.max() - ref_energies.min())
        energies = energies / (energies.max() - energies.min())

    return float(numpy.sqrt(numpy.mean(numpy.square(ref_energies - energies))))


def _plot_profile(
    plot_data: pandas.DataFrame,
    output_path: str,
    figure_size: tuple[float, float],
    y_label: str,
    y_interval: int,
    y_range: tuple[int, int],
    x_label: str,
    x_interval: int,
    x_lower: float,
    legend: bool = False,
):
    # Plot profile
    figure = pyplot.figure(figsize=figure_size)

    if legend:
        pyplot.plot(plot_data, label=plot_data.columns)
    else:
        pyplot.plot(plot_data)

    pyplot.xlim(x_lower, x_lower + 360)
    pyplot.xticks(numpy.arange(x_lower, x_lower + 361, x_interval))
    pyplot.ylim(y_range)
    pyplot.yticks(numpy.arange(y_range[0], y_range[1] + 1, y_interval))
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)

    if legend:
        figure.legend(loc="outside upper center", ncol=3)

    pyplot.savefig(output_path, bbox_inches="tight")
    pyplot.close(figure)


def _plot_heatmap(
    plot_data: pandas.DataFrame,
    output_path: str,
    figure_size: tuple[float, float],
    colorbar_label: str,
    colorbar_interval: int,
    colorbar_range: tuple[int, int],
    x_label: str,
    y_label: str,
    xy_interval: int,
    x_lower: float,
    y_lower: float,
    label_clusters: bool,
    minima_dihedrals: list[list[float]] | None = None,
):
    # Plot heatmap
    figure = pyplot.figure(figsize=figure_size)
    pyplot.imshow(
        plot_data, origin="lower", vmin=colorbar_range[0], vmax=colorbar_range[1]
    )

    pyplot.xlim(-0.5, 23.5)
    pyplot.ylim(-0.5, 23.5)
    pyplot.xticks(
        ticks=numpy.arange(0, 25, xy_interval / 15),
        labels=numpy.arange(x_lower, x_lower + 361, xy_interval),
    )
    pyplot.yticks(
        ticks=numpy.arange(0, 25, xy_interval / 15),
        labels=numpy.arange(y_lower, y_lower + 361, xy_interval),
    )
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)

    # Include colorbar
    pyplot.colorbar(
        label=colorbar_label,
        ticks=numpy.arange(colorbar_range[0], colorbar_range[1] + 1, colorbar_interval),
    )

    # Label Ramachandran clusters
    if label_clusters:
        axis = pyplot.gca()

        for cluster in HOLLINGSWORTH_RAMACHANDRAN_CLUSTERS:
            # Draw a rectange around the cluster limits
            cluster_x_lower = (cluster["phi"][0] - x_lower) / 15
            cluster_y_lower = (cluster["psi"][0] - y_lower) / 15
            cluster_width = (cluster["phi"][1] - x_lower) / 15 - cluster_x_lower
            cluster_height = (cluster["psi"][1] - y_lower) / 15 - cluster_y_lower

            axis.add_patch(
                patches.Rectangle(
                    xy=(cluster_x_lower, cluster_y_lower),
                    width=cluster_width,
                    height=cluster_height,
                    fill=False,
                )
            )

            # Label cluster name in center
            pyplot.text(
                cluster_x_lower + cluster_width / 2,
                cluster_y_lower + cluster_height / 2,
                cluster["cluster"],
                ha="center",
                va="center",
            )

    if minima_dihedrals is not None:
        pyplot.scatter(
            [(x - x_lower) / 15 for x in minima_dihedrals[0]],
            [(y - y_lower) / 15 for y in minima_dihedrals[1]],
            s=16,
            c="white",
            marker="x",
            linewidths=1.0,
        )

    pyplot.savefig(output_path, bbox_inches="tight")
    pyplot.close(figure)


def _plot_energy(
    record_name: str,
    energy_type: str,
    energy_df: pandas.DataFrame,
    output_dir: str,
    figure_size: tuple[float, float],
    ff_label: str,
    x_label: str,
    y_label: str | None = None,
    x_lower: float = -180,
    y_lower: float = -180,
    energy_interval: int = 4,
    energy_range: tuple[int, int] | None = None,
    angle_interval: int = 60,
    label_clusters: bool = False,
    minima_dihedrals: list[list[float]] | None = None,
):
    record_name = record_name.replace("-rotamer-1", "")

    if label_clusters:
        out_prefix = "ramachandran-labeled-"
    else:
        out_prefix = "ramachandran-"

    output_path = Path(
        output_dir,
        f"{out_prefix}{ff_label}-{record_name}-"
        f'{energy_type.replace(" ", "-").lower()}.pdf',
    )

    if energy_type == "QM Energy":
        energy_label = f"{record_name}\n{energy_type} " "(kcal mol$^{{-1}}$)"

    else:
        energy_label = f"{ff_label} {record_name}\n{energy_type} " "(kcal mol$^{{-1}}$)"

    if energy_range is None:
        energy_range = (
            4 * numpy.floor(energy_df[energy_type].min() / 4),
            4 * numpy.ceil(energy_df[energy_type].max() / 4),
        )

        if (energy_range[1] - energy_range[0]) < 16:
            energy_range = (
                2 * numpy.floor(energy_df[energy_type].min() / 2),
                2 * numpy.ceil(energy_df[energy_type].max() / 2),
            )

            energy_interval = 2

    # Check number of dimensions and convert DataFrame to 2-D array for plotting
    if "Y" in energy_df.columns:
        plot_data = energy_df.pivot(index="Y", columns="X", values=energy_type)
        _plot_heatmap(
            plot_data,
            output_path,
            figure_size,
            energy_label,
            energy_interval,
            energy_range,
            x_label,
            y_label,
            angle_interval,
            x_lower,
            y_lower,
            label_clusters,
            minima_dihedrals,
        )

        if label_clusters:
            beta = 1.0 / (R * 274.0)
            p_cluster = dict()

            for cluster in HOLLINGSWORTH_RAMACHANDRAN_CLUSTERS:
                if cluster["cluster"] in {r"$\alpha$", r"$\beta$"}:
                    cluster_df = energy_df[
                        (energy_df["X"] >= cluster["phi"][0])
                        & (energy_df["X"] <= cluster["phi"][1])
                        & (energy_df["Y"] >= cluster["psi"][0])
                        & (energy_df["Y"] <= cluster["psi"][1])
                    ]
                    p_cluster[cluster["cluster"]] = numpy.sum(
                        numpy.exp(-beta * cluster_df[energy_type])
                    )

            p_alpha, p_beta = p_cluster[r"$\alpha$"], p_cluster[r"$\beta$"]
            delta_f = -numpy.log(p_alpha / p_beta) / beta

            print(
                f"{ff_label:16s} {record_name:19s} {energy_type:9s} "
                f"{p_alpha:13.8f} {p_beta:13.8f} {delta_f:13.8f}"
            )

    else:
        plot_data = energy_df.set_index("X")[energy_type]
        _plot_profile(
            plot_data,
            output_path,
            figure_size,
            energy_label,
            energy_interval,
            energy_range,
            x_label,
            angle_interval,
            x_lower,
        )


def _plot_difference(
    record_name_1: str,
    energy_type_1: str,
    record_name_2: str,
    energy_type_2: str,
    qc_data,
    output_dir: str,
    figure_size: tuple[float, float],
    ff_label: str,
    x_label: str,
    y_label: str | None = None,
    x_lower: float = -180,
    y_lower: float = -180,
    energy_interval: int = 4,
    energy_range: tuple[int, int] | None = None,
    angle_interval: int = 60,
    label_clusters: bool = False,
    minima_dihedrals: list[list[float]] | None = None,
):
    energies_1 = qc_data[record_name_1]["energies"]
    energies_2 = qc_data[record_name_2]["energies"]

    record_name_1 = record_name_1.replace("-rotamer-1", "")
    record_name_2 = record_name_2.replace("-rotamer-1", "")

    if label_clusters:
        out_prefix = "ramachandran-labeled-"
    else:
        out_prefix = "ramachandran-"

    if energy_type_1 == energy_type_2:
        output_path = Path(
            output_dir,
            f"{out_prefix}{ff_label}-{record_name_1}-{record_name_2}-"
            f'{energy_type_1.replace(" ", "-").lower()}.pdf',
        )

        if energy_type_1 == "QM Energy":
            energy_label = (
                f"{record_name_1} $-$ {record_name_2}\n{energy_type_1} "
                "(kcal mol$^{{-1}}$)"
            )

        else:
            energy_label = (
                f"{ff_label}\n{record_name_1} $-$ {record_name_2}\n"
                f"{energy_type_1} "
                "(kcal mol$^{{-1}}$)"
            )

    elif record_name_1 == record_name_2:
        output_path = Path(
            output_dir,
            f"{out_prefix}{ff_label}-{record_name_1}-"
            f'{energy_type_1.replace(" ", "-").lower()}-'
            f'{energy_type_2.replace(" ", "-").lower()}.pdf',
        )

        energy_label = (
            f"{ff_label} {record_name_1}\n{energy_type_1} $-$ {energy_type_2}"
            "\n(kcal mol$^{{-1}}$)"
        )

    else:
        output_path = Path(
            output_dir,
            f"{out_prefix}{ff_label}-{record_name_1}-"
            f'{energy_type_1.replace(" ", "-").lower()}-'
            f'{record_name_2}-{energy_type_2.replace(" ", "-").lower()}.pdf',
        )

        energy_label = (
            f"{ff_label}\n{record_name_1} {energy_type_1} $-$ {record_name_2} "
            f"{energy_type_2}"
            "\n(kcal mol$^{{-1}}$)"
        )

    if energy_range is None:
        energy_range = (
            4
            * numpy.floor(
                min(
                    energies_1[energy_type_1].min(),
                    energies_2[energy_type_2].min(),
                )
                / 4
            ),
            4
            * numpy.ceil(
                max(
                    energies_1[energy_type_1].max(),
                    energies_2[energy_type_2].max(),
                )
                / 4
            ),
        )

        if (energy_range[1] - energy_range[0]) < 16:
            energy_range = (
                2
                * numpy.floor(
                    min(
                        energies_1[energy_type_1].min(),
                        energies_2[energy_type_2].min(),
                    )
                    / 2
                ),
                2
                * numpy.ceil(
                    max(
                        energies_1[energy_type_1].max(),
                        energies_2[energy_type_2].max(),
                    )
                    / 2
                ),
            )

            energy_interval = 2

    # Check number of dimensions, convert DataFrame to 2-D array and compute
    # difference for plotting
    if "Y" in energies_1.columns and "Y" in energies_2.columns:
        energy_series_1 = energies_1.set_index(["X", "Y"])[energy_type_1]
        energy_series_2 = energies_2.set_index(["X", "Y"])[energy_type_2]

        plot_data = energies_1.pivot(
            index="Y", columns="X", values=energy_type_1
        ) - energies_2.pivot(index="Y", columns="X", values=energy_type_2)
        _plot_heatmap(
            plot_data,
            output_path,
            figure_size,
            energy_label,
            energy_interval,
            energy_range,
            x_label,
            y_label,
            angle_interval,
            x_lower,
            y_lower,
            label_clusters,
            minima_dihedrals,
        )

    else:
        energy_series_1 = energies_1.set_index("X")[energy_type_1]
        energy_series_2 = energies_2.set_index("X")[energy_type_2]

        plot_data = energy_series_1 - energy_series_2
        _plot_profile(
            plot_data,
            output_path,
            figure_size,
            energy_label,
            energy_interval,
            energy_range,
            x_label,
            angle_interval,
            x_lower,
        )

    # Compute RMSE and normalized RMSE
    rmse = _compute_profile_rmse(
        energy_series_1, energy_series_2, normalize=False, shift=True
    )
    norm_rmse = _compute_profile_rmse(
        energy_series_1, energy_series_2, normalize=True, shift=True
    )

    return rmse, norm_rmse


def _plot_projection(
    record_name: str,
    energy_df: pandas.DataFrame,
    output_dir: str,
    figure_size: tuple[float, float],
    projection_label: str,
    projection_index=None,
    energy_interval: int = 4,
    energy_range: tuple[int, int] = None,
    angle_interval: int = 60,
    temperature: float = 310,
):
    beta = 1.0 / (R * temperature)

    output_path = Path(output_dir, f"{record_name}-{projection_label.lower()}.pdf")
    x_label = f'{record_name.replace("-rotamer-1", "")} {projection_label} (deg)'

    if "Y" in energy_df.columns:
        energy_label = "PMF (kcal mol$^{{-1}}$)"

        # Project profile onto one dihedral and compute potential of mean force
        dropped_index = "Y" if projection_index == "X" else "X"
        energy_columns = [s for s in energy_df.columns if s != dropped_index]
        probability_distribution = numpy.exp(-beta * energy_df[energy_columns])
        projection_partition_function = probability_distribution.groupby(
            projection_index, as_index=False
        ).sum()
        pmf = -numpy.log(projection_partition_function) / beta
        plot_data = pmf.set_index(projection_index)

    else:
        energy_label = "Energy (kcal mol$^{{-1}}$)"

        # Profile has only one dimension, so plot as is
        plot_data = energy_df.set_index("X").sort_index()

    if energy_range is None:
        energy_range = (
            4 * numpy.floor(plot_data.min().min() / 4),
            4 * numpy.ceil(plot_data.max().max() / 4),
        )

        if (energy_range[1] - energy_range[0]) < 16:
            energy_range = (
                2 * numpy.floor(plot_data.min().min() / 2),
                2 * numpy.ceil(plot_data.max().max() / 2),
            )

            energy_interval = 2

    _plot_profile(
        plot_data,
        output_path,
        figure_size,
        energy_label,
        energy_interval,
        energy_range,
        x_label,
        angle_interval,
        legend=True,
    )


@click.command()
@click.option(
    "-d/-l",
    "--dark_background/--light_background",
    default=True,
    help="Use the pyplot `dark_background` style.",
)
@click.option(
    "-f",
    "--figure_width",
    type=click.FLOAT,
    default=4.25,
    show_default=True,
    help="Width of plots in inches.",
)
@click.option(
    "-h",
    "--figure_height",
    type=click.FLOAT,
    default=None,
    show_default=True,
    help="Height of plots in inches. Default is 0.75 times figure_width.",
)
@click.option(
    "-i",
    "--input_dir",
    type=click.STRING,
    default="training-datasets",
    show_default=True,
    help="Directory path containing training TorsionDrives to be plotted.",
)
@click.option(
    "-m/-x",
    "--plot_mm_minima/--no_mm_minima",
    default=False,
    help="Plot MM minima on heatmap",
)
@click.option(
    "-n",
    "--names_path",
    type=click.STRING,
    default=Path("training-datasets", "torsiondrive-training-names.json"),
    show_default=True,
    help="File path for names of dataset records.",
)
@click.option(
    "-o",
    "--output_dir",
    type=click.STRING,
    default="plots",
    show_default=True,
    help="Directory path to which plots should be written.",
)
@click.option(
    "-s",
    "--font_size",
    type=click.INT,
    default=None,
    show_default=True,
    help="Font size in pt. Default is matplotlib rcParams.",
)
def main(
    dark_background,
    figure_width,
    figure_height,
    input_dir,
    plot_mm_minima,
    names_path,
    output_dir,
    font_size,
):
    if dark_background:
        pyplot.style.use("dark_background")

    # Reorder seaborn colorblind palette to avoid similar orange and red hues
    seaborn.set_palette(
        seaborn.color_palette(
            [
                seaborn.color_palette("colorblind")[i]
                for i in [0, 1, 2, 4, 8, 9, 7, 5, 6, 3]
            ]
        )
    )

    if figure_height is None:
        figure_size = tuple(figure_width * x for x in (1, 0.75))
    else:
        figure_size = (figure_width, figure_height)

    if font_size is not None:
        pyplot.rcParams.update({"font.size": font_size})

    with open(names_path, "r") as json_file:
        dataset_names = json.load(json_file)

    ff_labels = {
        "ff14SB": "ff14sb",
#        "ff14SB-onlysc": "ff14sbonlysc",
#        "OpenFF-2.1.0": "Sage-2.1.0",
#        "Sage-CC": "Sage-CC-0.0.3",
#        "Sage-2.1.0-NAGL": "Sage-2.1.0-NAGL",
        "Null-0.0.2": "Null-0.0.2",
        "Null-0.0.3": "Null-0.0.3",
#        "Null-0.0.3-DW": "Null-0.0.3-default-weights",
        "Null-0.0.3-SP": "Null-0.0.3-abinitio",
#        "Null-0.0.3-SP-DW": "Null-0.0.3-abinitio-default-weights",
#        "Null-0.0.3-QAmber": "Null-0.0.3-QAmber",
#        "Null-0.0.3-NBAmber": "Null-0.0.3-NBAmber",
#        "Null-0.0.3-NAGL": "Null-0.0.3-NAGL",
        "Null-0.0.3-Pair": "Null-0.0.3-Pair",
        "Specific-0.0.2": "Specific-0.0.2",
        "Specific-0.0.3": "Specific-0.0.3",
        "Specific-0.0.3-Pair": "Specific-0.0.3-Pair",
        "Specific-0.0.3-SPair": "Specific-0.0.3-Sage-Pair",
    }

    minima_dir = {
        "Null-min-pt": Path("..", "..", "null-model", "protein-mm-minima"),
        "Specific-min-pt": Path("..", "..", "b7s26-model", "protein-mm-minima"),
    }

    record_names = [
        "ala-rotamer-1",
        "arg-rotamer-1",
        "ash-rotamer-1",
        "asn-rotamer-1",
        "asp-rotamer-1",
        "cys-rotamer-1",
        "cyx-rotamer-1",
        "glh-rotamer-1",
        "gln-rotamer-1",
        "glu-rotamer-1",
        "gly-rotamer-1",
        "hid-rotamer-1",
        "hie-rotamer-1",
        "hip-rotamer-1",
        "ile-rotamer-1",
        "leu-rotamer-1",
        "lyn-rotamer-1",
        "lys-rotamer-1",
        "met-rotamer-1",
        "phe-rotamer-1",
        "pro-rotamer-1",
        "ser-rotamer-1",
        "thr-rotamer-1",
        "trp-rotamer-1",
        "tyr-rotamer-1",
        "val-rotamer-1",
    ]

    qc_data = dict()
    rmse_values = dict()
    norm_rmse_values = dict()

    for ff_label in ff_labels:
        qc_data[ff_label] = defaultdict(dict)
        rmse_values[ff_label] = defaultdict(dict)
        norm_rmse_values[ff_label] = defaultdict(dict)

        with open(
            Path(input_dir, f"torsiondrive-{ff_labels[ff_label]}-minimized.json"), "r"
        ) as input_file:
            qc_data_by_id = json.load(input_file)

        for record_id in qc_data_by_id:
            record_name = dataset_names[record_id]

            if record_name not in record_names:
                continue

            if len(record_name.split("-")) == 3:
                split_name = record_name.split("-", 1)
                record_name = f"Ace-{split_name[0].capitalize()}-Nme-{split_name[1]}"

            record_smiles = qc_data_by_id[record_id]["smiles"]
            record_energies = {
                tuple(json.loads(grid_id)): energy
                for grid_id, energy in qc_data_by_id[record_id]["energies"].items()
            }

            # Get number of dimensions in TorsionDrive
            num_dimensions = len(next(iter(record_energies.keys())))

            # Store grid energies in a pandas frame for easier manipulation
            # Wrap phi to [0, 360) and psi to [-120, 240)
            if "rotamer" in record_name:
                x_lower, y_lower = 0, -120
            else:
                x_lower, y_lower = -180, -180

            if num_dimensions == 1:
                energy_df = pandas.DataFrame(
                    [
                        {
                            "X": (grid_id[0] - x_lower) % 360 + x_lower,
                            "QM Energy": record_energies[grid_id][0],
                            "MM Energy": record_energies[grid_id][1],
                            "MM Target": record_energies[grid_id][2],
                            "MM RMSD": record_energies[grid_id][3],
                        }
                        for grid_id in record_energies
                    ]
                )

            else:
                energy_df = pandas.DataFrame(
                    [
                        {
                            "X": (grid_id[0] - x_lower) % 360 + x_lower,
                            "Y": (grid_id[1] - y_lower) % 360 + y_lower,
                            "QM Energy": record_energies[grid_id][0],
                            "MM Energy": record_energies[grid_id][1],
                            "MM Target": record_energies[grid_id][2],
                            "MM RMSD": record_energies[grid_id][3],
                        }
                        for grid_id in record_energies
                    ]
                )

            qc_data[ff_label][record_name]["record_id"] = record_id
            qc_data[ff_label][record_name]["smiles"] = record_smiles
            qc_data[ff_label][record_name]["energies"] = energy_df

            energy_range = {}
            energy_interval = {}

            if "pro" in record_name.lower():
                energy_range["QM Energy"] = (0, 32)
                energy_range["MM Energy"] = (-4, 32)
                energy_range["MM Target"] = (-12, 12)

            else:
                for energy_type in ["QM Energy", "MM Energy", "MM Target"]:
                    # energy_range[energy_type] = None
                    energy_range[energy_type] = (-4, 28)

            # Get axis labels
            if "rotamer" in record_name:
                x_label = "Phi (deg)"
                y_label = "Psi (deg)"

            else:
                x_label = "Chi1 (deg)"
                y_label = "Chi2 (deg)"

            # Get MM minima
            if plot_mm_minima:
                minima_dihedrals = [list(), list()]
                residue_name = record_name.split("-")[1].upper()
                residue_dir = Path(minima_dir[ff_label], residue_name)
                with open(Path(residue_dir, f"{residue_name}-medoids")) as medoid_file:
                    for line in medoid_file:
                        medoid = Path(residue_dir, line.strip())
                        offmol = Molecule.from_file(str(medoid))
                        qcmol = offmol.to_qcschema()
                        phi = qcmol.measure(
                            offmol.chemical_environment_matches(PHI_SMILES)[0]
                        )
                        psi = qcmol.measure(
                            offmol.chemical_environment_matches(PSI_SMILES)[0]
                        )
                        minima_dihedrals[0].append((phi - x_lower) % 360 + x_lower)
                        minima_dihedrals[1].append((psi - y_lower) % 360 + y_lower)

            else:
                minima_dihedrals = None

            # Plot torsion profiles for this record
            for energy_type in ["QM Energy", "MM Energy"]:
                for label_clusters in [True, False]:
                    _plot_energy(
                        record_name,
                        energy_type,
                        energy_df,
                        output_dir,
                        figure_size,
                        ff_label,
                        x_label,
                        y_label,
                        x_lower=x_lower,
                        y_lower=y_lower,
                        energy_range=energy_range[energy_type],
                        label_clusters=label_clusters,
                        minima_dihedrals=minima_dihedrals,
                    )

            # Plot difference heatmap between QM and MM energies
            for label_clusters in [True, False]:
                rmse, norm_rmse = _plot_difference(
                    record_name,
                    "QM Energy",
                    record_name,
                    "MM Energy",
                    qc_data[ff_label],
                    output_dir,
                    figure_size,
                    ff_label,
                    x_label,
                    y_label,
                    x_lower=x_lower,
                    y_lower=y_lower,
                    energy_interval=3,
                    energy_range=(-12, 12),
                    label_clusters=label_clusters,
                    minima_dihedrals=minima_dihedrals,
                )


if __name__ == "__main__":
    main()
