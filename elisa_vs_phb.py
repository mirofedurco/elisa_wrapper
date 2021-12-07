import os
import json
import phoebe
import numpy as np
from time import time

from elisa.binary_system import system
from elisa.observer.observer import Observer
from elisa import units
from elisa import settings
from elisa.const import Position
from elisa.binary_system.container import OrbitalPositionContainer
import astropy.units as u

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib


matplotlib.use('TkAgg')


PASSBAND_DICT = {"Kepler": "Kepler:mean", "TESS": "TESS:T", "bolometric": "Bolometric:900-40000"}


def get_params(filename):
    with open(filename, "r") as f:
        return json.loads(f.read())


def get_phoebe_binary(params, r_eq):
    """
    Initializing a Phoebe bundle from binary system JSON.

    :param params: Dict; system JSON with parameters
    :param r_eq: Tuple; equivalent radii
    :return: phoebe bundle
    """
    b = phoebe.Bundle.default_binary()
    # b = phoebe.Bundle.default_binary(contact_binary=True)

    b.set_value("requiv@primary@component", r_eq[0])
    b.set_value("requiv@secondary@component", r_eq[1])
    b.set_value("teff@primary", params["primary"]["t_eff"] * u.K)
    b.set_value("teff@secondary", params["secondary"]["t_eff"] * u.K)
    b.set_value("abun@primary", params["primary"]["metallicity"])
    b.set_value("abun@secondary", params["secondary"]["metallicity"])
    b.set_value("syncpar@primary@component", params['primary']['synchronicity'])
    b.set_value("syncpar@secondary@component", params['secondary']['synchronicity'])
    b.set_value("gravb_bol@primary", params['primary']['gravity_darkening'])
    b.set_value('gravb_bol@secondary', params['secondary']['gravity_darkening'])
    b.set_value('irrad_frac_refl_bol@primary', params['primary']['albedo'])
    b.set_value('irrad_frac_refl_bol@secondary', params['secondary']['albedo'])

    b.set_value_all('ld_func_bol', value=settings.LIMB_DARKENING_LAW)
    # b.set_value_all('ld_mode_bol', value='manual')
    b.set_value_all('ld_coeffs_source_bol', value='ck2004')

    b.set_value('period@binary@component', params['system']['period'] * u.d)
    b.set_value("per0@binary@component", params['system']['argument_of_periastron'] * u.deg)
    b.set_value('ecc@binary@component', params['system']['eccentricity'])
    b.set_value('incl@binary@component', params['system']['inclination'] * u.deg)
    b.set_value("q@binary@component", params['system']['mass_ratio'])
    b.set_value('sma@binary@component', params['system']['semi_major_axis'] * u.solRad)

    b.set_value("irrad_method@phoebe01@compute", "wilson")

    b.set_value('atm@primary', "ck2004")
    b.set_value('atm@secondary', "ck2004")

    return b


def run_phoebe_observation(binary, phases, passbands):
    """
    Producing observations using Phoebe.

    :param binary: phoebe bundle
    :param phases: numpy.array; photometric phases
    :param passbands: List;
    :return: phoebe bundle
    """
    times = phases * binary['period@orbit'].value
    passbands = [passbands] if type(passbands) == str else passbands
    for psbnd in passbands:
        binary.add_dataset('lc', times=times, passband=PASSBAND_DICT[psbnd], dataset=psbnd)
    binary.run_compute(irrad_method='wilson')
    # binary.run_compute()
    return binary


def get_data_phb(params, req):
    get_phoebe_binary(params, req)


def prepare_elisa_for_obs(params, passbands):
    """
    Initialize BinarySystem and Observer instance.

    :param params: dict; JSON with system parameters
    :param passbands: List;
    :return: elisa.Observer;
    """
    passbands = [passbands] if type(passbands) == str else passbands
    binary = system.BinarySystem.from_json(params)
    o = Observer(passband=passbands, system=binary)  # specifying the binary system to use in light curve synthesis
    return o


def get_elisa_observations(observer, phases):
    """
    Produce LC in elisa.

    :param observer: elisa.Observer;
    :param phases: numpy.array; photometric phases
    :return: elisa.Observer
    """
    observer.lc(
        phases=phases,
        # normalize=True
    )
    return observer


def display_comparison(phases_e, fluxes_e, phases_b, fluxes_b):
    res = (fluxes_e - fluxes_b)
    res -= res.mean()
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    ax1.plot(phases_b, fluxes_b, label='PHOEBE', c='blue')
    # ax1.plot(phases_b+0.5, fluxes_b, label='phoebe', c='red', linestyle='dashed')
    ax1.plot(phases_e, fluxes_e, label='ELISa', c='red')
    # ax1.plot(phases_e-0.5, fluxes_e, label='elisa', c='blue', linestyle='dashed')
    ax2.plot(phases_b, res, label='ELISa-PHOEBE')
    ax1.legend()
    ax2.legend(loc=0)
    # ax2.ticklabel_format(scilimits=(-3, 5))
    ax1.set_ylabel('Flux')
    ax2.set_xlabel('Phase')
    ax2.set_ylabel('Residual flux')
    plt.subplots_adjust(hspace=0, right=0.98, top=0.98, left=0.15)
    plt.show()


def produce_aux_params(params):
    """
    Generating parameters for the use in PHOEBE such as SMA, albedos, number of triangles and equivalent radii from
    elisa BinarySystem.

    :param params: dict; system JSON
    :return: Tuple; number of triangles, eqiovalent radii
    """
    binary = system.BinarySystem.from_json(params)
    sma = binary.semi_major_axis * units.DISTANCE_UNIT
    params["primary"]["albedo"] = binary.primary.albedo
    params["secondary"]["albedo"] = binary.secondary.albedo
    params["primary"]["gravity_darkening"] = binary.primary.gravity_darkening
    params["secondary"]["gravity_darkening"] = binary.secondary.gravity_darkening

    alphas = (np.degrees(binary.primary.discretization_factor), np.degrees(binary.secondary.discretization_factor))
    print(f"Discretization factors: {alphas}")

    # binary.plot.surface(colormap='temperature')

    position = Position(idx=0, distance=1.0, azimuth=90, true_anomaly=0, phase=0)
    container = OrbitalPositionContainer.from_binary_system(binary, position)
    container.build()

    ntri = (container.primary.faces.shape[0], container.secondary.faces.shape[0])
    print(f"Number of triangles: {ntri}")
    r_eq = binary.calculate_equivalent_radius(component='both')
    r_eq1 = r_eq['primary'] * sma
    r_eq2 = r_eq['secondary'] * sma

    print(f"Radii: {r_eq1.to(u.solRad):.2f}, {r_eq2.to(u.solRad):.2f} solRad")

    return ntri, (r_eq1, r_eq2)


def compare_lc(params, alpha, passband, nphs, normalize=True):
    settings.configure(
        LIMB_DARKENING_LAW='logarithmic',
        DEFAULT_DISCRETIZATION_FACTOR=alpha
    )

    phases = np.linspace(-0.6, 0.6, num=nphs)

    ntri, r_eq = produce_aux_params(params)

    start_time = time()
    binary = get_phoebe_binary(params, r_eq=r_eq)
    binary = run_phoebe_observation(binary, phases, passbands=passband)
    elapsed = np.round(time() - start_time, 2)
    print(f'PHOEBE time: {elapsed} s')
    phases_b = binary[f'{passband}@times@latest'].value / binary['period@orbit'].value
    fluxes_b = binary[f'{passband}@fluxes@latest'].value

    binary.get_parameter(context='dataset', qualifier='fluxes', dataset=passband)

    start_time = time()
    obs = prepare_elisa_for_obs(params, passband)
    obs = get_elisa_observations(obs, phases)
    elapsed = np.round(time() - start_time, 2)
    print(f'ELISA time: {elapsed} s')

    phases_e, fluxes_e = obs.phases, obs.fluxes
    fluxes_e = fluxes_e[passband]

    if normalize:
        fluxes_e /= np.max(fluxes_e)
        fluxes_b /= np.max(fluxes_b)

    print(f'Mean flux elisa: {fluxes_e.mean()}, phoebe: {fluxes_b.mean()}')

    display_comparison(phases_e, fluxes_e, phases_b, fluxes_b)


if __name__ == "__main__":
    matplotlib.use('TkAgg')
    logger = phoebe.logger(clevel='WARNING')
    home_dir = os.getcwd()

    settings.configure(
        LIMB_DARKENING_LAW='logarithmic',
        # REFLECTION_EFFECT=False
    )

    orbit = 'circular'
    alpha = 7
    N_phs = 100
    # passband = "bolometric"
    passband = "TESS"
    normalize = True

    data_circ = get_params(home_dir + '/models/test_binary_circ.json')
    data_ecc = get_params(home_dir + '/models/test_binary_ecc.json')

    data = data_ecc if orbit == 'eccentric' else data_circ

    compare_lc(params=data, alpha=alpha, passband=passband, nphs=N_phs, normalize=normalize)
