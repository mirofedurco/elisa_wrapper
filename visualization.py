import os
import pandas as pd
import numpy as np

from matplotlib.image import NonUniformImage
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import elisa_vs_phb as vs

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


def analyze_precision(dataframe):
    circ_mask = dataframe["eccentricity"] == 0.0

    fig = plt.figure(constrained_layout=True)
    spec = gridspec.GridSpec(ncols=1, nrows=2, figure=fig)
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[1, 0])

    ax1.hist(dataframe["std dev"][circ_mask], bins=100,
             density=True, label="circular")
    ax1.set_ylabel("Prob. density")
    ax1.legend()

    ax2.hist(dataframe["std dev"][~circ_mask], bins=100, density=True,
             label="eccentric")
    ax2.set_xlabel("Standard deviation")
    ax2.set_ylabel("Prob. density")
    ax2.legend()
    plt.show()


def gen_hist(x, y, ax, fig, y_sigma=None):
    y_range = y.max() if y_sigma == None else y_sigma * y.std()
    xedges = np.linspace(x.min(), x.max(), 100, endpoint=True)
    yedges = np.linspace(y.min(), y_range, 50, endpoint=True)

    h, xe, ye = np.histogram2d(x, y, bins=(xedges, yedges), density=True)
    h = h.T

    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(ye[0], ye[-1])

    # ax1 = fig.add_subplot(133, xlim=xedges[[0, -1]], ylim=yedges[[0, -1]])
    im = NonUniformImage(ax, interpolation='bilinear')
    xcenters = (xe[:-1] + xe[1:]) / 2
    ycenters = (ye[:-1] + ye[1:]) / 2
    im.set_data(xcenters, ycenters, np.log(h))
    ax.images.append(im)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('log(Prob. density)', rotation=270)


def param_vs_speed(dataframe, param):
    circ_mask = dataframe["eccentricity"] == 0.0

    fig = plt.figure(figsize=(8, 6))
    spec = gridspec.GridSpec(ncols=1, nrows=2, figure=fig)
    ax1 = fig.add_subplot(spec[0, 0], title="circular orbits")
    ax2 = fig.add_subplot(spec[1, 0], title="eccentric orbits")

    gen_hist(dataframe[param][circ_mask], dataframe["t_phoebe"][circ_mask]/dataframe["t_elisa"][circ_mask],
             ax1, fig, y_sigma=None)
    ax1.set_ylabel(r"$t_{phoebe}/t_{elisa}$")

    gen_hist(dataframe[param][~circ_mask],
             dataframe["t_phoebe"][~circ_mask] / dataframe["t_elisa"][~circ_mask], ax2, fig, y_sigma=None)
    ax2.set_xlabel(param)
    ax2.set_ylabel(r"$t_{phoebe}/t_{elisa}$")

    plt.subplots_adjust(right=1.0, top=0.95, hspace=0.35)
    plt.show()


def params_vs_speed(dataframe, params, circular):
    circ_mask = dataframe["eccentricity"] == 0.0 if circular else dataframe["eccentricity"] != 0.0
    name = "circulal orbits" if circular else "eccentric orbits"

    length = len(params)
    fig = plt.figure(figsize=(7, 2.4 * length))
    spec = gridspec.GridSpec(ncols=1, nrows=length, figure=fig)

    ax = []
    for ii, param in enumerate(params):
        ax.append(fig.add_subplot(spec[ii, 0]))
        gen_hist(dataframe[param][circ_mask], dataframe["t_phoebe"][circ_mask] / dataframe["t_elisa"][circ_mask],
                 ax[ii], fig)
        ax[ii].set_xlabel(param)
        ax[ii].set_ylabel(r"$t_{phoebe}/t_{elisa}$")

    ax[0].set_title(name)

    plt.subplots_adjust(right=1.0, top=0.98, hspace=0.35)
    plt.show()


def param_vs_precision(dataframe, param):
    circ_mask = dataframe["eccentricity"] == 0.0

    fig = plt.figure(figsize=(8, 6))
    spec = gridspec.GridSpec(ncols=1, nrows=2, figure=fig)
    ax1 = fig.add_subplot(spec[0, 0], title="circular orbits")
    ax2 = fig.add_subplot(spec[1, 0], title="eccentric orbits")

    gen_hist(dataframe[param][circ_mask], dataframe["std dev"][circ_mask], ax1, fig)
    ax1.set_ylabel("Standard deviation")

    gen_hist(dataframe[param][~circ_mask], dataframe["std dev"][~circ_mask], ax2, fig)
    ax2.set_ylabel("Standard deviation")
    ax2.set_xlabel(param)

    plt.subplots_adjust(right=1.0, top=0.95, hspace=0.35)
    plt.show()


def params_vs_precision(dataframe, params, circular):
    circ_mask = dataframe["eccentricity"] == 0.0 if circular else dataframe["eccentricity"] != 0.0
    name = "circulal orbits" if circular else "eccentric orbits"

    length = len(params)
    fig = plt.figure(figsize=(7, 2.4 * length))
    spec = gridspec.GridSpec(ncols=1, nrows=length, figure=fig)

    ax = []
    for ii, param in enumerate(params):
        ax.append(fig.add_subplot(spec[ii, 0]))
        gen_hist(dataframe[param][circ_mask], dataframe["std dev"][circ_mask], ax[ii], fig, y_sigma=5)
        ax[ii].set_xlabel(param)
        ax[ii].set_ylabel("Standard deviation")


    ax[0].set_title(name)

    plt.subplots_adjust(right=1.0, top=0.98, hspace=0.35)
    plt.show()


def produce_params_from_row(row):
    params = vs.get_params(os.path.join(os.getcwd(), 'models', 'init_binary.json'))
    params['system']['inclination'] = row['inclination']
    params['system']['period'] = row['period']
    params['system']['argument_of_periastron'] = row['argument_of_periastron']
    params['system']['mass_ratio'] = row['mass_ratio']
    params['system']['semi_major_axis'] = row['semi_major_axis']
    params['system']['eccentricity'] = row['eccentricity']

    params['primary']['surface_potential'] = row['p__surface_potential']
    params['secondary']['surface_potential'] = row['s__surface_potential']
    params['primary']['t_eff'] = row['p__t_eff']
    params['secondary']['t_eff'] = row['s__t_eff']

    return params


def compare_models(dtfrm, row_id, passband, alpha=7, n_phs=300):
    row = dtfrm.iloc[row_id]

    params = produce_params_from_row(row)
    vs.compare_lc(params=params, alpha=alpha, passband=passband, nphs=n_phs)


if __name__ == "__main__":
    home_dir = os.getcwd()
    datafile = os.path.join(home_dir, "results", "samples_bak2.csv")

    dtfrm = pd.read_csv(datafile)
    mean_max, maxmax = dtfrm["max_dev"].mean(), dtfrm["max_dev"].max()
    print(f"Mean max deviation: {mean_max}")
    print(f"Maximum max deviation: {maxmax}")

    mean_std, max_std = dtfrm["std dev"].mean(), dtfrm["std dev"].max()
    print(f"Mean std deviation: {mean_std}")
    print(f"Maximum std deviation: {max_std}")

    circ_mask = dtfrm["eccentricity"] == 0.0

    print(len(circ_mask), np.sum(circ_mask))

    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    # id = dtfrm['std dev'].idxmax()
    id = np.random.randint(0, dtfrm.shape[1])
    print(dtfrm.iloc[id])
    compare_models(dtfrm, id, 'TESS', n_phs=100)

    # analyze_speed(dtfrm)
    # analyze_precision(dtfrm)

    # params_vs_speed(dtfrm, params=['inclination', 'mass_ratio', 'p__t_eff', 'std dev', 'max_dev'], circular=True)
    # params_vs_speed(dtfrm, params=['eccentricity', 'p__surface_potential', 'p__t_eff', 'std dev'], circular=False)
    # params_vs_precision(dtfrm, params=['inclination', 'p__t_eff', 'max_dev'], circular=True)
    # params_vs_precision(dtfrm, params=['p__t_eff', 'max_dev'], circular=False)
    # for column in COLUMNS:
    #     param_vs_speed(dtfrm, column)
    #     param_vs_precision(dtfrm, column)
