import pytest
import numpy as np
from more_quant_chem_fewer_qub.cas import hamil_cas, dim_cas
from pyscf import gto
from pyscf.mcscf import CASCI

xyz_H2 = """H 0 0 0 
            H 0.7408481486 0 0"""
basis_H2 = "sto3g"


hamil_cas_ref_list = [(xyz_H2, basis_H2, 2, 2, "H2 STO-3G")]


@pytest.mark.parametrize("xyz, basis, norb_act, nelec_act, label", hamil_cas_ref_list)
def test_hamil_cas(xyz, basis, norb_act, nelec_act, label):
    print("\n\n" + 10 * "#" + "  " + label + "  " + 10 * "#")
    mol = gto.M(xyz, basis)
    hf = mol.RHF()
    hf.run()

    h_cas, e_const = hamil_cas(hf, norb_act, nelec_act)
    eigvals = np.linalg.eigvalsh(h_cas)
    e_res = sorted(eigvals) + e_const

    casci = CASCI(hf, norb_act, nelec_act)
    dim = dim_cas(norb_act, nelec_act)
    casci.fcisolver.nroots = dim
    e_ref = sorted(casci.kernel()[0])

    np.testing.assert_allclose(e_res, e_ref)
