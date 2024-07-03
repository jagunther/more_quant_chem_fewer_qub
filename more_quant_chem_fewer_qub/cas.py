from scipy.special import binom
from pyscf.mcscf import CASCI, CASSCF
from pyscf.fci.direct_spin1 import pspace


def hamil_cas(hf, norb_act, nelec_act):
    """

    :param rhf: PySCF scf object, needs to be converged a converged RHF calculation
    :param norb_act: number of spatial orbitals in active space
    :param nelec_act: number of electrons in active space
    :return: (2D np.ndarray, float): The Hamiltonian and a constant energy offset
    """
    cas = CASCI(hf, norb_act, nelec_act)
    h1e_cas, ecore = cas.get_h1eff()
    h2e_cas = cas.get_h2eff()
    dim = dim_cas(norb_act, nelec_act)
    h_ci = pspace(h1e_cas, h2e_cas, norb_act, nelec_act, np=dim)[1]
    return h_ci, ecore


def e_cas(rhf, norb_act, nelec_act, spin=0, nroots=1, ci0=None):
    """
    CASCI calculation based on RHF calculation

    :param rhf: PySCF scf object, needs to be converged a converged RHF calculation
    :param norb_act: int, the number of active orbitals (spatial)
    :param nelec_act: tuple or int, the number of active electrons
    :param spin: int, the spin (0=singlet, 1/2=doublet, 1=triplet)
    :param nroots: int, the number of roots to solve for
    :param ci0: CIVector, initial guess
    :return: float and CIvector, the resulting total energy and its eigenvector
    """
    if not isinstance(nelec_act, tuple):
        if nelec_act % 2 == 1:
            raise ValueError("Need even number of active electrons")
        nelec_act = (nelec_act // 2, nelec_act // 2)

    casci = CASCI(rhf, norb_act, nelec_act)
    casci.fix_spin(ss=spin * (spin + 1), shift=0.01)
    casci.fcisolver.nroots = nroots
    res = casci.kernel(ci0=ci0)
    ci_vecs = [res[2]] if nroots == 1 else res[2]
    for ci_vec in ci_vecs:
        res_spin = casci.fcisolver.spin_square(ci_vec, norb_act, nelec_act)[0]
        if not abs(res_spin - spin) < 1e-5:
            raise RuntimeError(
                f"Incorrect spin state, wanted {spin=}, but got {res_spin=}"
            )
    energies = res[0]
    return energies, ci_vecs[0]


def dim_cas(norb_act, nelec_act):
    if isinstance(nelec_act, int):
        assert nelec_act % 2 == 0
        nelec_act = (nelec_act // 2, nelec_act // 2)
    dim = binom(norb_act, nelec_act[0]) * binom(norb_act, nelec_act[1])
    return int(dim)
