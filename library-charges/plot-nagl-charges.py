from pathlib import Path

import click
from matplotlib import pyplot
import numpy
import pandas
import seaborn


@click.command()
@click.option(
    "-i",
    "--input-path",
    default="nagl-charges.dat",
    show_default=True,
    type=click.STRING,
    help="The path to the file containing library and NAGL charges.",
)
@click.option(
    "-o",
    "--output-dir",
    default="plots",
    show_default=True,
    type=click.STRING,
    help="The path to the file containing the library charges.",
)
def main(input_path, output_dir):

    #pyplot.style.use('dark_background')
    pyplot.rcParams.update({"font.size": 10})
    figure_size = tuple(4.25 * x for x in (1, 0.75))
    seaborn.set_palette(
        seaborn.color_palette(
            [
                seaborn.color_palette("colorblind")[i]
                for i in [0, 1, 2, 4, 8, 9, 7, 5, 6, 3]
            ]
        )
    )

    charge_df = pandas.read_csv(input_path, index_col=0)
    charge_df = charge_df.drop_duplicates()
    charge_df = charge_df.rename(columns={"NAGL 0.1.0rc1": "NAGL"})

    charge_labels = [c for c in charge_df.columns if c not in {"Residue", "Atom", "OpenEye ELF10"}]
    bar_width = 0.8 / len(charge_labels)
    bar_location_offset = (len(charge_labels) - 1) / 2

    #for res_name in charge_df["Residue"].unique():
    for res_name in ["ALA"]:

        residue_df = charge_df[charge_df["Residue"] == res_name]
        atoms = residue_df["Atom"].values
        tick_locations = numpy.arange(len(atoms))
        figure = pyplot.figure(figsize=figure_size)

        for i, charge_label in enumerate(charge_labels):

            bar_locations = (
                tick_locations + (i - bar_location_offset) * bar_width
            )
            bar_heights = [
                residue_df[residue_df["Atom"] == atom][charge_label].values[0]
                if charge_label == "Amber ff99"
                else float(
                    residue_df[residue_df["Atom"] == atom][charge_label].values[0].split()[0]
                )
                for atom in atoms
            ]

            pyplot.bar(
                bar_locations, bar_heights, width=bar_width, label=charge_label
            )

        #pyplot.xticks(tick_locations, labels=atoms, fontsize=8)
        pyplot.xticks(tick_locations, labels=atoms)
        pyplot.ylim(-1, 1)
        pyplot.yticks(numpy.arange(-1, 1.1, 0.2))
        #pyplot.ylabel("Charge (e)")
        pyplot.ylabel("ALA charge (e)")
        figure.legend(loc = "outside upper center", ncol=3)

        #pyplot.savefig(Path(output_dir, f"{res_name.lower()}-charges.pdf"))
        pyplot.savefig(Path(output_dir, f"{res_name.lower()}-charges.png"))
        pyplot.close(figure)


if __name__ == "__main__":
    main()

