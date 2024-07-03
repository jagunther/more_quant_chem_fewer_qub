import os
import numpy as np
import pytest
from more_quant_chem_fewer_qub.matrix_operators import (
    hamil_from_fcidump,
    make_dyall_fcidump,
)

# Mock system: Only one-electron interactions, 3 spatial orbitals, 2 electrons
#       spin = ms = 0
# Total Hamiltonian
#       H = Σ_pq=1^3 Σ_s h_pq a_ps^+ a_qs
#
# Dyall Hamiltonian
#       H_dyall = Σ_pq=1^2 Σ_s h_pq a_ps^+ a_qs + Σ_s ε a_3s^+ a_3s
#
# parameters:
#   h_11, h_12, h_22, h_13, h_23, ε
#
# orbital order: 1α 2α 3α 1β 2β 3β
# CI basis order: |100100> |100010> |100001> |010100> |010010> |010001> |001100> |001010> |001001>

e_const = 0
h11 = 0
h12 = 0.5
h22 = 1
h13 = 0.25
h23 = 0.125
h33 = 3  # does not play a role
fcidumpfile_dyall_mock = os.getcwd() + "/FCIDUMP_dyall_mock"
fcidumpfile_mock = os.getcwd() + "/FCIDUMP_mock"

ε = 2
mo_energy_mock = [0, 0, ε]

h_dyall_mock = np.array(
    [
        [2 * h11, h12, 0, h12, 0, 0, 0, 0, 0],
        [h12, h11 + h22, 0, 0, h12, 0, 0, 0, 0],
        [0, 0, h11 + ε, 0, 0, h12, 0, 0, 0],
        [h12, 0, 0, h11 + h22, h12, 0, 0, 0, 0],
        [0, h12, 0, h12, 2 * h22, 0, 0, 0, 0],
        [0, 0, h12, 0, 0, h22 + ε, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, h11 + ε, h12, 0],
        [0, 0, 0, 0, 0, 0, h12, h22 + ε, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 2 * ε],
    ]
)

make_dyall_fcidump_ref_list = [
    (fcidumpfile_mock, 2, mo_energy_mock, h_dyall_mock, "mock")
]


@pytest.mark.parametrize(
    "fcidumpfile, norb_act, mo_energy, hamil_dyall_ref, label",
    make_dyall_fcidump_ref_list,
)
def test_make_dyall_fcidump(fcidumpfile, norb_act, mo_energy, hamil_dyall_ref, label):
    print("\n\n" + 10 * "#" + "  " + label + "  " + 10 * "#")
    result_file = "FCIDUMP_dyall_temporary"
    make_dyall_fcidump(fcidumpfile, result_file, 2, mo_energy)
    result_hamil = hamil_from_fcidump(result_file)
    os.remove(result_file)
    assert np.allclose(result_hamil, hamil_dyall_ref)


hamil_from_fcidump_ref_list = [
    (fcidumpfile_dyall_mock, h_dyall_mock, "mock hamiltonian")
]


@pytest.mark.parametrize("fcidumpfile, hamil_ref, label", hamil_from_fcidump_ref_list)
def test_hamil_from_fcidump(fcidumpfile, hamil_ref, label):
    print("\n\n" + 10 * "#" + "  " + label + "  " + 10 * "#")
    result = hamil_from_fcidump(fcidumpfile)
    assert np.allclose(result, hamil_ref)
