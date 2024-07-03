from pyscf.tools import fcidump
from pyscf.ao2mo.addons import restore
from pyscf.mcscf import CASCI
import numpy as np
from numba import jit
import copy


def norm1_from_hf(hf, norb_act=None, nelec_act=None):
    """
    Computes the qubit 1-norm after fermion-to-qubit mapping.
    Allows for active space truncation with respect to the canonical
    orbitals by specifying norb_act and nelec_act.

    :param hf: converged PySCF RHF object
    :param norb_act: int, number of active spatial orbitals
    :param nelec_act: int, number of active electrons
    :return: float, float, float, the 1-norm of the hamiltonian after fermion-to-qubit for
                the constant, quadratic and quartic terms
    """
    if norb_act is None or nelec_act is None:
        norb_act = hf.mol.nao
        nelec_act = hf.mol.nelec

    casci = CASCI(hf, norb_act, nelec_act)
    h1e, ecore = casci.get_h1eff()
    h2e = restore("s1", casci.get_h2eff(), norb_act)
    return norm1(h1e, h2e)


def norm1_from_fcidump(fcidumpfile):
    """
    Computes the qubit 1-norm after fermion-to-qubit mapping

    :param fcidumpfile: str, path to FCIDUMP file
    :return: float, float, float, the 1-norm of the hamiltonian after fermion-to-qubit for
                the constant, quadratic and quartic terms
    """
    data = fcidump.read(fcidumpfile)
    h1e = data["H1"]
    h2e = data["H2"]
    norb = data["NORB"]
    h2e = restore("s1", h2e, norb)
    return norm1(h1e, h2e)


def norm1(h1e, h2e):
    """
    Computes the qubit 1-norm after fermion-to-qubit mapping

    see eq.(19)-(22) in Koridon et al., PRR 2021, or Loaiza et al., Quantum Sci. Technol. 2023
    :param h1e: 2D numpy array, the 1-electron integral tensor
    :param h2e: 4D numpy array, the 2-electron electron repulsion tensor (chemists' notation)
    :return: float, float, float, the 1-norm of the hamiltonian after fermion-to-qubit for
                the constant, quadratic and quartic terms
    """
    norb = h1e.shape[0]
    lambda_c = 0
    for p in range(norb):
        lambda_c += h1e[p, p]
        for r in range(norb):
            lambda_c += 0.5 * h2e[p, p, r, r]
            lambda_c -= 0.25 * h2e[p, r, r, p]
    lambda_c = abs(lambda_c)

    lambda_t = 0
    for p in range(norb):
        for q in range(norb):
            curr = h1e[p, q]
            for r in range(norb):
                curr += h2e[p, q, r, r]
                curr -= 0.5 * h2e[p, r, r, q]
            lambda_t += abs(curr)

    lambda_v = 0
    for r in range(norb - 1):
        for p in range(r + 1, norb):
            for q in range(norb - 1):
                for s in range(q + 1, norb):
                    lambda_v += 0.5 * abs(h2e[p, q, r, s] - h2e[p, s, r, q])

    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    lambda_v += 0.25 * abs(h2e[p, q, r, s])

    return lambda_c, lambda_t, lambda_v


def norm1_active_space(h1e, h2e, norb_core, norb_act):
    """
    Computes the qubit 1-norm after fermion-to-qubit mapping of an active-space
    truncated Hamiltonian.

    see eq.(19)-(22) in Koridon et al., PRR 2021, or Loaiza et al., Quantum Sci. Technol. 2023
    :param h1e: 2D numpy array, the 1-electron integral tensor
    :param h2e: 4D numpy array, the 2-electron electron repulsion tensor (chemists' notation)
    :param norb_core: int, the number of core orbitals
    :param norb_act: int, the number of active orbitals
    :return: float, float, float, the 1-norm of the hamiltonian after fermion-to-qubit for
                the constant, quadratic and quartic terms
    """
    start = norb_core
    end = norb_core + norb_act
    h1e_act = h1e[start:end, start:end]
    h2e_act = h2e[start:end, start:end, start:end, start:end]

    lambda_c, lambda_t, lambda_v = norm1(h1e_act, h2e_act)
    return lambda_c, lambda_t, lambda_v


#######################################################################################

# The methods norm_mrpt_fulldifferenc, normv23_mrpt, normv23_mrpt_abs
# assume the following setup: The hamiltonian H specified by h1e and h2e
# is in the frozen-core approximation, so the effects of the core electrons is
# folded into h1e and the frozen core energy efrcore already.
# The orbital space is then partitioned into an active and a virtual part with
# the variable norb_act.
# Furthermore, e_virt is the list of orbital energies of the virtuals which
# goes into the definition of the Dyall Hamiltonian.


def norm_mrpt_fulldifference(h1e, h2e, norb_act, e_virt, efrcore):
    """
    Computes upper bounds on the 1-norm of Dyall-Hamiltonian and on the 2/3-norm
    of the difference V = H - H_Dyall. The number of orbitals (active + virtual)
    is denoted by N.

    :param h1e: NxN numpy array
    :param h2e: NxNxNxN
    :param norb_act: int
    :param e_virt: list of length N-norb_act
    :param efrcore: float
    :return: tuple (float, float)
    """

    normh1 = abs(efrcore) + np.sum(np.abs(e_virt))
    normh1 += np.sum(np.abs(h1e[:norb_act, :norb_act]))
    normh1 += np.sum(np.abs(h2e[:norb_act, :norb_act, :norb_act, :norb_act]))

    # remove all-active entries
    h1e_noact = copy.copy(h1e)
    h2e_noact = copy.copy(h2e)
    h1e_noact[:norb_act, :norb_act] = 0
    h2e_noact[:norb_act, :norb_act, :norb_act, :norb_act] = 0

    normv23 = np.sum(np.abs(h1e_noact) ** (2 / 3))
    normv23 += np.sum(np.abs(e_virt) ** (2 / 3))
    normv23 += np.sum(2 * np.abs(h2e_noact) ** (2 / 3))
    normv23 = 2 ** (5 / 2) * normv23 ** (3 / 2)
    return normh1, normv23


def normv23_mrpt_abs(h1e, h2e, norb_act):
    """
    Computes upper bounds on the mrpt2 perturbation 2/3-norm, using the absolute value
    of each coefficient.

    :param h1e: NxN numpy array
    :param h2e: NxNxNxN
    :param norb_act: int
    :return: float
    """

    def v11sum():
        return np.sum(np.abs(v11(h1e, norb_act)) ** (2 / 3))

    v11_sum_abs = v11sum()

    def v31sum():
        return np.sum(np.abs(v31(h1e, h2e, norb_act)) ** (2 / 3))

    v31_sum_abs = v31sum()

    def v22sum():
        return np.sum(np.abs(v22(h2e, norb_act) ** (2 / 3)))

    v22_sum_abs = v22sum()
    v33_vals = v33(h2e, norb_act)

    @jit(nopython=True)
    def v33sum():
        return np.sum(v33_vals ** (2 / 3))
        # return np.sum(np.abs(v33(h2e, norb_act, e_virt, efrcore) ** (2 / 3)))

    v33_sum_abs = v33sum()

    v_mrpt2_abs = (
        2 ** (5 / 3) * v11_sum_abs
        + 2 ** (16 / 3) * v31_sum_abs
        + 2 ** (14 / 3) * v22_sum_abs
        + 64 * v33_sum_abs
    ) ** (3 / 2)
    return v_mrpt2_abs


def normv23_mrpt(h1e, h2e, norb_act, e_virt, efrcore, times):
    """
    Computes upper bounds on the mrpt2 perturbation 2/3-norm as a function
    of t.

    :param h1e: NxN numpy array
    :param h2e: NxNxNxN
    :param norb_act: int
    :param e_virt: list of length N-norb_act
    :param efrcore: float
    :return: tuple (float, float)
    :param times: list of float
    :return: list of float, results for each time
    """
    res = np.empty_like(times)
    for i, t in enumerate(times):
        v11_sum = np.sum(np.abs(v11(h1e, norb_act, e_virt, efrcore, t) ** (2 / 3)))
        v31_sum = np.sum(np.abs(v31(h1e, h2e, norb_act, e_virt, efrcore, t) ** (2 / 3)))
        v22_sum = np.sum(np.abs(v22(h2e, norb_act, e_virt, efrcore, t) ** (2 / 3)))
        v33_sum = np.sum(np.abs(v33(h2e, norb_act, e_virt, efrcore, t) ** (2 / 3)))

        v_mrpt2 = (
            2 ** (5 / 3) * v11_sum
            + 2 ** (16 / 3) * v31_sum
            + 2 ** (14 / 3) * v22_sum
            + 64 * v33_sum
        ) ** (3 / 2)
        res[i] = v_mrpt2
    return res


def v11(h1e, norb_act, e_virt=None, efrcore=None, t=None):
    """
    1-electron excitation coeffs, h1e-h1e interaction, first line of eq. (24)
    Takes absolute values for t=None.
    """
    h1e_actvirt = h1e[norb_act:, :norb_act]
    if t is not None:
        exp_virt = np.exp(-1j * (e_virt + efrcore) * t)
        res = np.einsum("v,va,vb -> ab", exp_virt, h1e_actvirt, h1e_actvirt)
    else:
        res = np.einsum("va,vb -> ab", np.abs(h1e_actvirt), np.abs(h1e_actvirt))
    return res


def v31(h1e, h2e, norb_act, e_virt=None, efrcore=None, t=None):
    """
    1-electron excitation coeffs, h1e-h2e interaction, second line of eq. (24)
    Takes absolute values for t=None
    """
    h1e_actvirt = h1e[norb_act:, :norb_act]
    h2e_actvirt = h2e[norb_act:, :norb_act, :norb_act, :norb_act]
    if t is not None:
        exp_virt = np.exp(-1j * (e_virt + efrcore) * t)
        res = np.einsum("v, vd, vabc -> abcd", exp_virt, h1e_actvirt, h2e_actvirt)
    else:
        res = np.einsum("vd, vabc -> abcd", np.abs(h1e_actvirt), np.abs(h2e_actvirt))
    return res


def v22(h2e, norb_act, e_virt=None, efrcore=None, t=None):
    """
    2-electron excitation coeffs, h2e-h2e interaction, third line of eq. (24)
    Takes absolute values for t=None
    """
    h2e_actvirt = h2e[norb_act:, norb_act:, :norb_act, :norb_act]
    if t is not None:
        mesh = np.tile(e_virt, (e_virt.shape[0], 1)) + np.tile(
            np.atleast_2d(e_virt).T, (1, e_virt.shape[0])
        )
        exp_virt = np.exp(-1j * (mesh + efrcore) * t)
        res = np.einsum("vw, vwab, vwcd -> abcd", exp_virt, h2e_actvirt, h2e_actvirt)
    else:
        res = np.einsum("vwab, vwcd -> abcd", np.abs(h2e_actvirt), np.abs(h2e_actvirt))
    return res


def v33(h2e, norb_act, e_virt=None, efrcore=None, t=None):
    """
    1-electron excitation coeffs, h2e-h2e interaction, fourth line of eq. (24)
    Takes absolute values for t=None
    """
    h2e_actvirt = h2e[norb_act:, :norb_act, :norb_act, :norb_act]
    if t is not None:
        exp_virt = np.exp(-1j * (e_virt + efrcore) * t)
        res = np.einsum("v, vabc, vdef -> abcdef", exp_virt, h2e_actvirt, h2e_actvirt)
    else:
        res = np.einsum(
            "vabc, vdef -> abcdef", np.abs(h2e_actvirt), np.abs(h2e_actvirt)
        )
    return 0.5 * res
