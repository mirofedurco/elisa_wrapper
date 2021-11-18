import os
import sys
import json
import phoebe
import numpy as np
from time import time

from elisa.binary_system import system
from elisa.observer.observer import Observer
from elisa import units
from elisa import settings
from elisa import ld
from elisa.const import Position
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.binary_system.surface.gravity import calculate_polar_gravity_acceleration
import astropy.units as u

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib


matplotlib.use('TkAgg')


PASSBAND_DICT = {"Kepler": "Kepler:mean", "TESS": "TESS:T", "bolometric": "Bolometric:900-40000"}


def get_params(filename):
    with open(filename, "r") as f:
        return json.loads(f.read())


def get_binary(params, triangles, r_eq):
    b = phoebe.Bundle.default_binary()
    # b = phoebe.Bundle.default_binary(contact_binary=True)

    b.set_value("requiv@primary@component", r_eq[0])
    b.set_value("requiv@secondary@component", r_eq[1])
    b.set_value('teff@primary', params['primary']['t_eff']*u.K)
    b.set_value('teff@secondary', params['secondary']['t_eff']*u.K)
    b.set_value('abun@primary', params['primary']['metallicity'])
    b.set_value('abun@secondary', params['secondary']['metallicity'])
    b.set_value('syncpar@primary', params['primary']['synchronicity'])
    b.set_value('syncpar@secondary', params['secondary']['synchronicity'])
    b.set_value('gravb_bol@primary', params['primary']['gravity_darkening'])
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

    b.set_value("irrad_method@phoebe01@phoebe@compute", "wilson")
    # b.set_value('ntriangles@primary', triangles[0])
    # b.set_value('ntriangles@secondary', triangles[1])

    b.set_value("irrad_method@phoebe01@compute", "wilson")

    b.set_value('atm@primary', "ck2004")
    b.set_value('atm@secondary', "ck2004")

    b.set_value("syncpar@primary@component", params['primary']['synchronicity'])
    b.set_value("syncpar@secondary@component", params['secondary']['synchronicity'])

    # b.set_value("ld_coeffs_bol@primary", ld_bol_coeff[orbit]['primary'])
    # b.set_value("ld_coeffs_bol@secondary", ld_bol_coeff[orbit]['secondary'])
    # b['ld_mode@secondary'] = 'manual'

    # print(b.filter(context=['compute']))
    # print(b.filter(context=['component', 'system']))
    # print(b.filter(context=['constraint']))
    # b.set_value_all('ld_mode_bol', value='manual')
    # print(b.get_parameter(context=['component'], component='primary', qualifier='ld_coeffs_source_bol'))
    # print(b.get_parameter(context=['component'], component='primary', qualifier='ld_func_bol'))
    # print(b.get_parameter(context='compute', qualifier='irrad_method'))
    # print(b['compute'].twigs)
    # print(b['component'].twigs)
    # sys.exit()

    return b


def run_observation(binary, phases, passbands):
    times = phases * binary['period@orbit'].value
    passbands = [passbands] if type(passbands) == str else passbands
    for psbnd in passbands:
        binary.add_dataset('lc', times=times, passband=PASSBAND_DICT[psbnd], dataset=psbnd)
    binary.run_compute(irrad_method='wilson')
    # binary.run_compute()
    return binary


def get_data_phb(params, req):
    get_binary(params, req)


def get_data_elisa(params, phs, passbands):
    passbands = [passbands] if type(passbands) == str else passbands
    binary = system.BinarySystem.from_json(params)
    o = Observer(passband=passbands, system=binary)  # specifying the binary system to use in light curve synthesis

    o.lc(
        phases=phs,
        # normalize=True
    )
    return o, binary


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


if __name__ == "__main__":
    logger = phoebe.logger(clevel='WARNING')
    home_dir = os.getcwd()

    settings.configure(
        LIMB_DARKENING_LAW='logarithmic',
    )

    orbit = 'circular'
    # orbit = 'eccentric'

    # alpha = 2
    # alpha = 3
    # alpha = 5
    alpha = 7
    # alpha = 10
    # N_phs = 10
    # N_phs = 400
    N_phs = 100
    surface_discredizations = [10, 7, 5, 3]
    n_phases = np.arange(10, 300, 20)
    passband_to_disp = "TESS"
    passbands = ["bolometric", "TESS", "Kepler"]
    normalize = True
    # normalize = False

    data_circ = get_params(home_dir + '/models/test_binary_circ.json')
    data_ecc = get_params(home_dir + '/models/test_binary_ecc.json')

    data = data_ecc if orbit == 'eccentric' else data_circ

    # config.REFLECTION_EFFECT = False
    data['primary']['discretization_factor'] = alpha
    # params['secondary']['discretization_factor'] = alpha
    binary = system.BinarySystem.from_json(data)
    sma = binary.semi_major_axis * units.DISTANCE_UNIT

    data["primary"]["albedo"] = binary.primary.albedo
    data["secondary"]["albedo"] = binary.secondary.albedo
    data["primary"]["gravity_darkening"] = binary.primary.gravity_darkening
    data["secondary"]["gravity_darkening"] = binary.secondary.gravity_darkening

    alphas = (np.degrees(binary.primary.discretization_factor), np.degrees(binary.secondary.discretization_factor))
    print(f"Discretization factors: {alphas}")

    position = Position(idx=0, distance=1.0, azimuth=90, true_anomaly=0, phase=0)
    container = OrbitalPositionContainer.from_binary_system(binary, position)
    container.build()

    ntri = (container.primary.faces.shape[0], container.secondary.faces.shape[0])
    print(f"Number of triangles: {ntri}")
    r_eq = binary.calculate_equivalent_radius(component='both')
    # K = 0.999
    K = 1.0
    # K = 1.002
    r_eq1 = K * r_eq['primary'] * sma
    r_eq2 = K * r_eq['secondary'] * sma

    print(f"Radii: {r_eq1.to(u.solRad):.2f}, {r_eq2.to(u.solRad):.2f} solRad")

    phases = np.linspace(-0.6, 0.6, num=N_phs)
    data['primary']['discretization_factor'] = alpha
    data['secondary']['discretization_factor'] = alpha

    start_time = time()
    binary = get_binary(data, triangles=ntri, r_eq=(r_eq1, r_eq2))
    binary = run_observation(binary, phases, passbands=passbands)
    elapsed = np.round(time() - start_time, 2)
    print(f'PHOEBE time: {elapsed} s')
    phases_b = binary[f'{passband_to_disp}@times@latest'].value / binary['period@orbit'].value
    fluxes_b = binary[f'{passband_to_disp}@fluxes@latest'].value

    start_time = time()
    binary.get_parameter(context='dataset', qualifier='fluxes', dataset=passband_to_disp)

    obs, binary_e = get_data_elisa(data, phases, passbands=passbands)
    elapsed = np.round(time() - start_time, 2)
    print(f'ELISA time: {elapsed} s')

    phases_e, fluxes_e = obs.phases, obs.fluxes
    fluxes_e = fluxes_e[passband_to_disp]

    if normalize:
        fluxes_e /= np.max(fluxes_e)
        fluxes_b /= np.max(fluxes_b)

    print(f'Mean flux elisa: {fluxes_e.mean()}, phoebe: {fluxes_b.mean()}')

    display_comparison(phases_e, fluxes_e, phases_b, fluxes_b)
