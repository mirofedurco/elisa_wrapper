import os
import csv
import numpy as np

from copy import copy
from time import time
from multiprocessing.pool import Pool

import elisa_vs_phb as vs

from elisa import BinarySystem, settings
from elisa.base.error import (
    MorphologyError, AtmosphereError,
    LimbDarkeningError, InitialParamsError,
    TemperatureError, GravityError, SpotError
)

I_RANGE = (75, 90)
P_RANGE = (1, 10)
ARG0_RANGE = (0, 360)
E_RANGE = (0, 0.9)
Q_RANGE = (0.1, 1)
A_RANGE = (1, 10)
OMEGA_RANGE = (2, 10)
TEFF_RANGE = (3500, 50000)
TEFF_RANGE_OVERCONTACT = (3500, 8000)
N_PHS_RANGE = (100, 800)
# N_PHS_RANGE = (10, 100)

ALPHA = 7
N_SAMPLES = 6000
NUMBER_OF_PROCESSES = 7

ERRORS = (
    MorphologyError, AtmosphereError,
    LimbDarkeningError, InitialParamsError,
    TemperatureError, GravityError, SpotError
)

COLUMNS = ["inclination", "period", "argument_of_periastron", "eccentricity", "mass_ratio", "semi_major_axis",
           "p__surface_potential", "p__t_eff", "s__surface_potential", "s__t_eff", "std dev", "max_dev", "t_phoebe",
           "t_elisa", "N_phases"]


def draw_params(params, circular=False):
    params['system']['inclination'] = np.random.uniform(I_RANGE[0], I_RANGE[1])
    params['system']['period'] = np.random.uniform(P_RANGE[0], P_RANGE[1])
    params['system']['argument_of_periastron'] = np.random.uniform(ARG0_RANGE[0], ARG0_RANGE[1])
    params['system']['mass_ratio'] = np.random.uniform(Q_RANGE[0], Q_RANGE[1])
    params['system']['semi_major_axis'] = np.random.uniform(A_RANGE[0], A_RANGE[1])

    params['system']['eccentricity'] = 0.0 if circular else np.random.uniform(E_RANGE[0], E_RANGE[1])

    crit_pot = BinarySystem.libration_potentials_static(1.0, params['system']['mass_ratio'])
    params['primary']['surface_potential'] = np.random.uniform(OMEGA_RANGE[0], OMEGA_RANGE[1]) \
        if circular else np.random.uniform(crit_pot[1], OMEGA_RANGE[1])

    detached = params['primary']['surface_potential'] >= crit_pot[1]
    params['secondary']['surface_potential'] = np.random.uniform(crit_pot[1], OMEGA_RANGE[1]) \
        if detached else params['primary']['surface_potential']

    if detached:
        params['primary']['t_eff'] = np.random.uniform(TEFF_RANGE[0], TEFF_RANGE[1])
        params['secondary']['t_eff'] = np.random.uniform(TEFF_RANGE[0], TEFF_RANGE[1])
    else:
        params['primary']['t_eff'] = np.random.uniform(TEFF_RANGE_OVERCONTACT[0], TEFF_RANGE_OVERCONTACT[1])
        low = params['primary']['t_eff'] - 500 if params['primary']['t_eff'] - 500 < 3500 else 3500
        params['secondary']['t_eff'] = np.random.uniform(low, params['primary']['t_eff'] + 500)

    return params


def eval_node(params, pssbnds, pssbnd_to_analyse, file):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    circular = np.random.choice([True, False], p=[0.4, 0.6])
    params = draw_params(params=params, circular=circular)

    try:
        ntri, r_eq = vs.produce_aux_params(params)
    except ERRORS as e:
        print(f"Invalid system, reason: {e}")
        return False

    nphs = np.random.randint(N_PHS_RANGE[0], N_PHS_RANGE[1])
    print(f"Number of phases: {nphs}")
    phases = np.linspace(-0.5, 0.5, num=nphs)

    try:
        start_time = time()
        binary = vs.get_binary(params, triangles=ntri, r_eq=r_eq)
        binary = vs.run_observation(binary, phases, passbands=pssbnds)
        elapsed_p = np.round(time() - start_time, 2)
    except (ValueError,) as e:
        print(f"Invalid system, reason: {e}")
        return False

    print(f'PHOEBE time: {elapsed_p} s')

    phases_b = binary[f'{pssbnd_to_analyse}@times@latest'].value / binary['period@orbit'].value
    fluxes_b = binary[f'{pssbnd_to_analyse}@fluxes@latest'].value

    try:
        start_time = time()
        obs, binary_e = vs.get_data_elisa(params, phases, passbands=pssbnds)
        elapsed_e = np.round(time() - start_time, 2)
        print(f'ELISA time: {elapsed_e} s')
    except ERRORS as e:
        print(f"Invalid system, reason: {e}")
        return False

    phases_e, fluxes_e = obs.phases, obs.fluxes
    fluxes_e = fluxes_e[passband_to_analyse]

    fluxes_e /= np.max(fluxes_e)
    fluxes_b /= np.max(fluxes_b)

    print(f'Mean flux elisa: {fluxes_e.mean()}, phoebe: {fluxes_b.mean()}')

    res = np.abs(fluxes_e - fluxes_b)
    stdev = res.std()
    maxdev = res.max()

    write_csv_row(file, params, stdev, maxdev, elapsed_p, elapsed_e, nphs)
    # vs.display_comparison(phases_e, fluxes_e, phases_b, fluxes_b)

    return True


def create_file_header(file):
    with open(file, "w") as fl:
        writer = csv.writer(fl)
        writer.writerow(COLUMNS)


def write_csv_row(file, params, stdev, maxdev, t_phoebe, t_elisa, n_phs):
    row = [params["system"][col] for col in COLUMNS[:6]]
    row += [params["primary"][col.split("__")[1]] for col in COLUMNS[6:8]]
    row += [params["secondary"][col.split("__")[1]] for col in COLUMNS[8:10]]
    row += [stdev, maxdev, t_phoebe, t_elisa, n_phs]

    with open(file, "a") as fl:
        writer = csv.writer(fl)
        writer.writerow(row)


def perform_sampling(params_orig, pssbnds, pssbnd_to_analyse, file):
    params = copy(params_orig)
    if not os.path.isfile(file):
        create_file_header(file)
    success = np.full(int(N_SAMPLES), False, dtype=bool)
    while not success.all():
        # eval_node(params, pssbnds, pssbnd_to_analyse, circular)
        fail_mask = success == False
        pool = Pool(processes=NUMBER_OF_PROCESSES)
        result = [pool.apply_async(eval_node, (params, pssbnds, pssbnd_to_analyse, file)) for _ in success[fail_mask]]
        pool.close()
        pool.join()
        success[fail_mask] = np.array([r.get() for r in result])
        print(f"Number of succesfull attempts: {np.count_nonzero(success)}/{N_SAMPLES}")


if __name__ == "__main__":
    settings.configure(
        LIMB_DARKENING_LAW='logarithmic',
        DEFAULT_DISCRETIZATION_FACTOR=ALPHA
    )
    passband_to_analyse = "TESS"
    passbands = ["bolometric", "TESS", "Kepler"]

    home_dir = os.getcwd()

    data_orig = vs.get_params(home_dir + '/models/init_binary.json')
    result_file = home_dir + '/results/samples.csv'
    perform_sampling(data_orig, passbands, passband_to_analyse, file=result_file)
