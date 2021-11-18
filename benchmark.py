import os
import elisa_vs_phb as vs


I_RANGE = (60, 90)
P_RANGE = (1, 10)
ARG0_RANGE = (0, 360)
E_RANGE = (0, 0.9)
Q_RANGE = (0.1, 1)
A_RANGE = (1, 10)
OMEGA_RANGE = (2, 10)
TEFF_RANGE = (3500, 20000)


if __name__ == "__main__":
    home_dir = os.getcwd()

    data_circ = vs.get_params(home_dir + '/models/test_binary_circ.json')
    data_ecc = vs.get_params(home_dir + '/models/test_binary_ecc.json')

