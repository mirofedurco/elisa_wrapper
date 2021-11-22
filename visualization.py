import os
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from benchmark import COLUMNS


def analyze_speed(dataframe):
    circ_mask = dataframe["eccentricity"] == 0.0

    fig = plt.figure(constrained_layout=True)
    spec = gridspec.GridSpec(ncols=1, nrows=2, figure=fig)
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[1, 0])

    ax1.hist(dataframe["t_phoebe"][circ_mask]/dataframe["t_elisa"][circ_mask], bins=100,
             density=True, label="circular")
    ax1.set_ylabel("Prob. density")
    ax1.legend()

    ax2.hist(dataframe["t_phoebe"][~circ_mask] / dataframe["t_elisa"][~circ_mask], bins=100, density=True,
             label="eccentric")
    ax2.set_xlabel(r"$t_{phoebe}/t_{elisa}$")
    ax2.set_ylabel("Prob. density")
    ax2.legend()
    plt.show()


def params_vs_speed(dataframe, param):
    circ_mask = dataframe["eccentricity"] == 0.0

    fig = plt.figure(constrained_layout=True)
    spec = gridspec.GridSpec(ncols=1, nrows=2, figure=fig)
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[1, 0])

    ax1.scatter(dataframe[param][circ_mask], dataframe["t_phoebe"][circ_mask]/dataframe["t_elisa"][circ_mask],
                label="circular", s=2)
    ax1.set_ylabel(r"$t_{phoebe}/t_{elisa}$")
    ax1.legend()

    ax2.scatter(dataframe[param][~circ_mask], dataframe["t_phoebe"][~circ_mask]/dataframe["t_elisa"][~circ_mask],
                label="eccentric", s=2)
    ax2.set_xlabel(param)
    ax2.set_ylabel(r"$t_{phoebe}/t_{elisa}$")
    ax2.legend()
    plt.show()


def params_vs_precision(dataframe, param):
    circ_mask = dataframe["eccentricity"] == 0.0

    fig = plt.figure(constrained_layout=True)
    spec = gridspec.GridSpec(ncols=1, nrows=2, figure=fig)
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[1, 0])

    ax1.scatter(dataframe[param][circ_mask], dataframe["std dev"][circ_mask],
                label="circular", s=2)
    ax1.set_ylabel("Standard deviation")
    ax1.legend()

    ax2.scatter(dataframe[param][~circ_mask], dataframe["std dev"][~circ_mask],
                label="eccentric", s=2)
    ax2.set_xlabel(param)
    ax2.set_ylabel("Standard deviation")
    ax2.legend()
    plt.show()


if __name__ == "__main__":
    home_dir = os.getcwd()
    datafile = os.path.join(home_dir, "results", "samples.csv")

    dtfrm = pd.read_csv(datafile)
    mean_max, maxmax = dtfrm["max_dev"].mean(), dtfrm["max_dev"].max()
    print(f"Mean max deviation: {mean_max}")
    print(f"Maximum max deviation: {maxmax}")

    mean_std, max_std = dtfrm["std dev"].mean(), dtfrm["std dev"].max()
    print(f"Mean std deviation: {mean_std}")
    print(f"Maximum std deviation: {max_std}")

    circ_mask = dtfrm["eccentricity"] == 0.0

    analyze_speed(dtfrm)
    for column in COLUMNS:
        params_vs_speed(dtfrm, column)
        params_vs_precision(dtfrm, column)
