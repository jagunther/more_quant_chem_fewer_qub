import os
import pytest
import numpy as np
import timeit
from more_quant_chem_fewer_qub.e2 import (
    e2_from_fcimatrix,
    e2_general,
    e2_from_casmatrix,
    e_nevpt2,
)
from pyscf import gto
from more_quant_chem_fewer_qub.matrix_operators import perturbation_ops_from_fcidump

# Mock system: Only one-electron interactions, 3 spatial orbitals, 2 electrons
#       spin = ms = 0
# Total Hamiltonian
#       H = Σ_pq=1^3 Σ_s h_pq a_ps^+ a_qs
#
# Dyall Hamiltonian
#       H_dyall = Σ_pq=1^2 Σ_s h_pq a_ps^+ a_qs + Σ_s ε a_3s^+ a_3s
#
# perturbation
#       V = Σ_s (h_13 a_3s^+ a_1s + h_23 a_3s^+ a_2s) + h.c.
#
# parameters:
#   h_11, h_12, h_22, h_13, h_23, ε
#
# orbital order: 1α 2α 3α 1β 2β 3β
# CI basis order: |100100> |100010> |100001> |010100> |010010> |010001> |001100> |001010> |001001>

h11 = 0
h12 = 0.5
h22 = 1
h13 = 0.25
h23 = 0.125
ε = 2
mo_energy_mock = [0, 0, ε]
fcidumpfile_mock = os.getcwd() + "/FCIDUMP_mock"

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

v_mock = np.array(
    [
        [0, 0, h13, 0, 0, 0, h13, 0, 0],
        [0, 0, h23, 0, 0, 0, 0, h13, 0],
        [h13, h23, 0, 0, 0, 0, 0, 0, h13],
        [0, 0, 0, 0, 0, h13, h23, 0, 0],
        [0, 0, 0, 0, 0, h23, 0, h23, 0],
        [0, 0, 0, h13, h23, 0, 0, 0, h23],
        [h13, 0, 0, h23, 0, 0, 0, 0, h13],
        [0, h13, 0, 0, h23, 0, 0, 0, h23],
        [0, 0, h13, 0, 0, h23, h13, h23, 0],
    ]
)


def e2_mock_exact():
    "Calculate E2 from manually constructed matrices h and v"
    eigvals, eigvecs = np.linalg.eigh(h_dyall_mock)
    gs = eigvecs[:, 0]
    id = np.identity(h_dyall_mock.shape[0])
    Pi = id - np.outer(gs, gs)
    R0 = np.linalg.inv(eigvals[0] * id - h_dyall_mock)
    return gs.T @ v_mock @ Pi @ R0 @ Pi @ v_mock @ gs


def test_e2_mock():
    h_dyall_res, v_res = perturbation_ops_from_fcidump(
        fcidumpfile_mock, 2, mo_energy_mock
    )
    res = e2_general(h_dyall_res, v_res)
    assert abs(res - e2_mock_exact()) < 1e-8


xyz_H2 = "H1 0 0 0; H2 1.2324008952 0 0"
basis_H2 = {"H1": "631g", "H2": "sto3g"}

xyz_BH = """B 0 0 0; H 1.2324008952 0 0"""

basis_BH = "sto3g"

xyz_LiH = """Li 0 0 0; H 1 0 0"""

e2_ref_list = [
    (xyz_H2, basis_H2, 2, 2, "H2, only 1 virtual"),
    (xyz_BH, "sto3g", 5, 6, "BH, only 1 virtual"),
    (xyz_BH, "sto3g", 4, 4, "BH, only 1 virtual + frozen core"),
    (xyz_BH, "sto3g", 4, 6, "BH, 2 virtuals"),
    (xyz_H2, "631g", 2, 2, "H2, 2 virtuals"),
    (xyz_LiH, "631g", 4, 4, "LiH, 7 virtuals"),
]


@pytest.mark.parametrize("xyz, basis, norb_act, nelec_act, label", e2_ref_list)
def test_e2_from_casmatrix(xyz, basis, norb_act, nelec_act, label):
    print("\n\n" + 10 * "#" + "  " + label + "  " + 10 * "#")
    mol = gto.M(atom=xyz, basis=basis)
    hf = mol.RHF.newton()
    hf.run()
    e2_fcimat = e2_from_fcimatrix(hf, norb_act, nelec_act)
    e2_casmat = e2_from_casmatrix(hf, norb_act, nelec_act)
    print(f"{e2_fcimat=}, {e2_casmat=}")
    assert abs(e2_fcimat - e2_casmat) < 1e-8


xyz_H2O = """
O          0.00000        0.00000        0.11779
H          0.00000        0.75545       -0.47116
H          0.00000       -0.75545       -0.47116
"""
