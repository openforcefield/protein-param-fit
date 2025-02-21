import json
from collections import defaultdict
from pathlib import Path

import click
import numpy
import pandas
import seaborn
from matplotlib import pyplot
from openmm import unit


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
    legend: bool = False,
):
    # Plot profile
    figure = pyplot.figure(figsize=figure_size)

    if legend:
        pyplot.plot(plot_data, label=plot_data.columns)
    else:
        pyplot.plot(plot_data)

    pyplot.xlim(-180, 180)
    pyplot.xticks(numpy.arange(-180, 181, x_interval))
    pyplot.ylim(y_range)
    pyplot.yticks(numpy.arange(y_range[0], y_range[1] + 1, y_interval))
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)

    if legend:
        figure.legend(loc="outside upper center", ncol=3)

    pyplot.savefig(output_path)
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
        labels=numpy.arange(-180, 181, xy_interval),
    )
    pyplot.yticks(
        ticks=numpy.arange(0, 25, xy_interval / 15),
        labels=numpy.arange(-180, 181, xy_interval),
    )
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)

    # Include colorbar
    pyplot.colorbar(
        label=colorbar_label,
        ticks=numpy.arange(colorbar_range[0], colorbar_range[1] + 1, colorbar_interval),
    )

    pyplot.savefig(output_path)
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
    energy_interval: int = 4,
    energy_range: tuple[int, int] | None = None,
    angle_interval: int = 60,
):
    output_path = Path(
        output_dir,
        f'{ff_label}-{record_name}-{energy_type.replace(" ", "-").lower()}.pdf',
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
    energy_interval: int = 4,
    energy_range: tuple[int, int] | None = None,
    angle_interval: int = 60,
):
    if energy_type_1 == energy_type_2:
        output_path = Path(
            output_dir,
            f"{ff_label}-{record_name_1}-{record_name_2}-"
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
            f"{ff_label}-{record_name_1}-"
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
            f"{ff_label}-{record_name_1}-"
            f'{energy_type_1.replace(" ", "-").lower()}-'
            f'{record_name_2}-{energy_type_2.replace(" ", "-").lower()}.pdf',
        )

        energy_label = (
            f"{ff_label}\n{record_name_1} {energy_type_1} $-$ {record_name_2} "
            f"{energy_type_2}"
            "\n(kcal mol$^{{-1}}$)"
        )

    energies_1 = qc_data[record_name_1]["energies"]
    energies_2 = qc_data[record_name_2]["energies"]

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
        )

    # Compute RMSE and normalized RMSE
    rmse = _compute_profile_rmse(
        energy_series_1, energy_series_2, normalize=False, shift=True
    )
    norm_rmse = _compute_profile_rmse(
        energy_series_1, energy_series_2, normalize=True, shift=True
    )

    return rmse, norm_rmse


def _plot_rmse(
    rmse: dict[str, dict[str, float]],
    output_path: str,
    figure_size: tuple[float, float],
    y_label: str,
    rotate_x_labels: bool = False,
):
    ff_labels = [*rmse]
    x_labels = [*rmse[ff_labels[0]]]

    tick_locations = numpy.arange(len(x_labels))
    bar_width = 0.8 / len(ff_labels)
    bar_location_offset = (len(ff_labels) - 1) / 2
    figure = pyplot.figure(figsize=figure_size)

    for i, ff_label in enumerate(ff_labels):
        bar_locations = tick_locations + (i - bar_location_offset) * bar_width
        bar_heights = [rmse[ff_label][x_label] for x_label in x_labels]

        pyplot.bar(bar_locations, bar_heights, width=bar_width, label=ff_label)

    x_label_rotation = 90.0 if rotate_x_labels else 0.0

    pyplot.xticks(
        tick_locations,
        labels=x_labels,
        fontsize=10,
        rotation=x_label_rotation,
    )
    pyplot.ylim(bottom=0)
    pyplot.ylabel(y_label)
    figure.legend(loc="outside upper center", ncol=2)

    pyplot.savefig(output_path)
    pyplot.close(figure)


def _plot_force_field_rmse(
    rmse: dict[str, dict[str, float]],
    output_path: str,
    figure_size: tuple[float, float],
    y_label: str,
    rotate_x_labels: bool = False,
    dark_background: bool = False,
    bootstrap_iterations: int = 2000,
    bootstrap_percentile: float = 0.95,
):
    ff_labels = [*rmse]

    # Bootstrap RMSE values to compute error bars.
    plot_data = [
        numpy.array([rmse[ff_label][rmse_index] for rmse_index in rmse[ff_label]])
        for ff_label in ff_labels
    ]

    bootstrap_samples = {
        ff_label: numpy.zeros(bootstrap_iterations) for ff_label in ff_labels
    }

    bootstrap_sample_count = len(plot_data[0])

    for bootstrap_index in range(bootstrap_iterations):
        sample_indices = numpy.random.randint(
            low=0,
            high=bootstrap_sample_count,
            size=bootstrap_sample_count,
        )

        for ff_label, ff_rmse in zip(ff_labels, plot_data):
            bootstrap_samples[ff_label][bootstrap_index] = ff_rmse[
                sample_indices
            ].mean()

    lower_percentile_index = int(
        numpy.round(bootstrap_iterations * (1 - bootstrap_percentile) / 2)
    )
    upper_percentile_index = int(
        numpy.round(bootstrap_iterations * (1 + bootstrap_percentile) / 2)
    )

    # Plot RMSEs with error bars
    bar_locations = numpy.arange(len(ff_labels))
    bar_width = 0.8
    bar_heights = numpy.zeros(len(ff_labels))
    bar_confidence_intervals = numpy.zeros((2, len(ff_labels)))

    for i, (ff_label, ff_rmse) in enumerate(zip(ff_labels, plot_data)):
        sorted_samples = numpy.sort(bootstrap_samples[ff_label])
        bar_heights[i] = ff_rmse.mean()
        bar_confidence_intervals[0][i] = float(
            numpy.abs(bar_heights[i] - sorted_samples[lower_percentile_index])
        )
        bar_confidence_intervals[1][i] = float(
            numpy.abs(bar_heights[i] - sorted_samples[upper_percentile_index])
        )

    # Print RMSEs and bootstrapped confidence intervals
    print(y_label.replace("$", "").replace("{", "").replace("}", ""))
    for i, ff_label in enumerate(ff_labels):
        print(
            f"    {ff_label:14s} {bar_heights[i]:6.4f} "
            f"({bar_heights[i] - bar_confidence_intervals[0][i]:6.4f} to "
            f"{bar_heights[i] + bar_confidence_intervals[1][i]:6.4f})"
        )

    figure = pyplot.figure(figsize=figure_size)
    pyplot.bar(
        bar_locations,
        bar_heights,
        width=bar_width,
        yerr=bar_confidence_intervals,
        ecolor="white" if dark_background else "black",
        # error_kw = {'linewidth': 2.75},
    )

    x_label_rotation = 90.0 if rotate_x_labels else 0.0
    x_labels = [
        x_label.replace("-", "\n", 1).replace("-", " ")
        for x_label in ff_labels
    ]

    pyplot.xticks(bar_locations, labels=x_labels, rotation=x_label_rotation)
    pyplot.ylim(bottom=0)
    pyplot.ylabel(y_label)

    pyplot.savefig(output_path)
    pyplot.close(figure)


def _plot_projection(
    record_name: str,
    energy_df: pandas.DataFrame,
    output_dir: str,
    figure_size: tuple[float, float],
    projection_label: str,
    projection_index: str | None = None,
    energy_interval: int = 4,
    energy_range: tuple[int, int] = None,
    angle_interval: int = 60,
    temperature: float = 310,
):
    R = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilocalorie_per_mole / unit.kelvin)
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
    default="validation-datasets",
    show_default=True,
    help="Directory path containing validation TorsionDrives to be plotted.",
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
    output_dir,
    font_size,
):
    if dark_background:
        pyplot.style.use("dark_background")

    # Reorder seaborn colorblind palette to avoid similar orange and red hues
    seaborn.set_palette(
        seaborn.color_palette(
            [
                seaborn.color_palette("tab10")[i]
                for i in [0, 1, 2, 4, 3, 9, 7, 5, 6, 8]
            ]
        )
    )

    if figure_height is None:
        figure_size = tuple(figure_width * x for x in (1, 0.75))
    else:
        figure_size = (figure_width, figure_height)

    if font_size is not None:
        pyplot.rcParams.update({"font.size": font_size})

    with open(Path(input_dir, "torsiondrive-validation-names.json"), "r") as json_file:
        dataset_names = json.load(json_file)

    ff_labels = {
        "ff14SB": "ff14sb",
        "ff14SB-onlysc": "ff14sbonlysc",
        #"OpenFF-2.1.0": "Sage-2.1.0",
        #"Sage-CC": "Sage-CC-0.0.3",
        "Sage-2.1-NAGL": "Sage-2.1.0-NAGL",
        #"Null-0.0.2": "Null-0.0.2",
        #"Null-QM": "Null-0.0.3",
        #"Null-0.0.3-DW": "Null-0.0.3-default-weights",
        #"Null-0.0.3-SP": "Null-0.0.3-abinitio",
        #"Null-0.0.3-SP-DW": "Null-0.0.3-abinitio-default-weights",
        #"Null-0.0.3-QAmber": "Null-0.0.3-QAmber",
        #"Null-0.0.3-NBAmber": "Null-0.0.3-NBAmber",
        #"Null-0.0.3-NAGL": "Null-0.0.3-NAGL",
        "Null-QM": "Null-0.0.3-Pair",
        #"Specific-0.0.2": "Specific-0.0.2",
        #"Specific-QM": "Specific-0.0.3",
        #"Specific-QM-Pair": "Specific-0.0.3-Pair",
        "Specific-QM": "Specific-0.0.3-Sage-Pair",
    }

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
            record_smiles = qc_data_by_id[record_id]["smiles"]
            record_energies = {
                tuple(json.loads(grid_id)): energy
                for grid_id, energy in qc_data_by_id[record_id]["energies"].items()
            }

            # Get number of dimensions in TorsionDrive
            num_dimensions = len(next(iter(record_energies.keys())))

            # Store grid energies in a pandas frame for easier manipulation
            # Wrap +180 to -180 for plotting
            if num_dimensions == 1:
                energy_df = pandas.DataFrame(
                    [
                        {
                            "X": (grid_id[0] if grid_id[0] < 180 else grid_id[0] - 360),
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
                            "X": (grid_id[0] if grid_id[0] < 180 else grid_id[0] - 360),
                            "Y": (grid_id[1] if grid_id[1] < 180 else grid_id[1] - 360),
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

            # Plot torsion profiles for this record
            for energy_type in ["QM Energy", "MM Energy", "MM Target"]:
                _plot_energy(
                    record_name,
                    energy_type,
                    energy_df,
                    output_dir,
                    figure_size,
                    ff_label,
                    x_label,
                    y_label,
                    energy_range=energy_range[energy_type],
                )

            # Plot difference heatmap between QM and MM energies
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
                energy_interval=3,
                energy_range=(-12, 12),
            )

            rmse_index = record_name.replace("-rotamer-1", "").lower()
            rmse_index = rmse_index.replace("ace-", "").replace("-nme", "")
            rmse_values[ff_label][rmse_index] = rmse
            norm_rmse_values[ff_label][rmse_index] = norm_rmse

    # Plot QM-MM RMSEs for each validation target by force field
    _plot_rmse(
        rmse_values,
        Path(output_dir, "torsiondrive-target-qm-mm-rmse.pdf"),
        figure_size,
        "Capped 3-mer backbone\nRMSE (kcal mol$^{-1}$)",
        rotate_x_labels=True,
    )

    _plot_rmse(
        norm_rmse_values,
        Path(output_dir, "torsiondrive-target-qm-mm-norm-rmse.pdf"),
        figure_size,
        "Capped 3-mer backbone\nNormalized RMSE",
        rotate_x_labels=True,
    )

    # Plot average RMSE by force field with bootstrapping for confidence
    # intervals
    _plot_force_field_rmse(
        rmse_values,
        Path(output_dir, "force-field-qm-mm-rmse.pdf"),
        figure_size,
        "Capped 3-mer backbone\nRMSE (kcal mol$^{-1}$)",
        dark_background=dark_background,
        #rotate_x_labels=True,
    )

    _plot_force_field_rmse(
        norm_rmse_values,
        Path(output_dir, "force-field-qm-mm-norm-rmse.pdf"),
        figure_size,
        "Capped 3-mer backbone\nNormalized RMSE",
        dark_background=dark_background,
        rotate_x_labels=True,
    )

    # Reorder seaborn colorblind palette for projections so that force fields
    # retain their color from previous plots
    seaborn.set_palette(
        seaborn.color_palette(
            [
                seaborn.color_palette("tab10")[i]
                for i in [7, 0, 1, 2, 4, 3, 9, 5, 6, 8]
                # for i in [7, 0, 5, 1, 2, 9, 4, 8, 6, 3]
            ]
        )
    )

    # Plot slices of torsion profiles
    slice_ff_labels = [*qc_data]
    for record_name in [*qc_data[slice_ff_labels[0]]]:
        # Merge DataFrames on X, Y, and QM Energy to ensure grid points are the
        # same between force fields
        energy_df = qc_data[slice_ff_labels[0]][record_name]["energies"]

        if "Y" in energy_df.columns:
            select_columns = ["X", "Y", "QM Energy", "MM Energy"]
            merge_columns = ["X", "Y", "QM Energy"]

        else:
            select_columns = ["X", "QM Energy", "MM Energy"]
            merge_columns = ["X", "QM Energy"]

        energy_df = energy_df[select_columns].rename(
            columns={"MM Energy": slice_ff_labels[0]}
        )

        for ff_label in slice_ff_labels[1:]:
            energy_df_2 = qc_data[ff_label][record_name]["energies"]

            energy_df_2 = energy_df_2[select_columns].rename(
                columns={"MM Energy": ff_label}
            )

            energy_df = pandas.merge(energy_df, energy_df_2, on=merge_columns)

        if "rotamer" in record_name:
            _plot_projection(
                record_name=record_name,
                energy_df=energy_df,
                output_dir=output_dir,
                figure_size=figure_size,
                projection_label="Phi",
                projection_index="X",
                #energy_range = (
                #(-4, 32) if 'pro' in record_name.lower() else None
                #),
            )

            _plot_projection(
                record_name=record_name,
                energy_df=energy_df,
                output_dir=output_dir,
                figure_size=figure_size,
                projection_label="Psi",
                projection_index="Y",
            )

        else:
            _plot_projection(
                record_name=record_name,
                energy_df=energy_df,
                output_dir=output_dir,
                figure_size=figure_size,
                projection_label="Chi1",
                projection_index="X",
            )

            if "Y" in energy_df.columns:
                _plot_projection(
                    record_name=record_name,
                    energy_df=energy_df,
                    output_dir=output_dir,
                    figure_size=figure_size,
                    projection_label="Chi2",
                    projection_index="Y",
                )


if __name__ == "__main__":
    main()
