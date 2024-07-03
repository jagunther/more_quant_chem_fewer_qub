import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from pyscf import fci, mrpt
from pyscf.mcscf import CASCI
from pyscf.ao2mo.addons import restore
from pyscf.fci.addons import cre_a, cre_b, des_a, des_b
from more_quant_chem_fewer_qub.matrix_operators import perturbation_ops
from more_quant_chem_fewer_qub.cas import hamil_cas


def e2_1actvirt(hf, norb_act, nelec_act):
    """
    Computes all second-order energy corrections of single active-virtual
    excitations.
    :param hf: a converged PySCF SCF-object
    :param norb_act: int, number of active orbitals
    :param nelec_act: int, number of active electrons
    :return: 2 2D numpy arrays, their length equals the number of virtual orbitals.
        The first contains all energy corrections of alpha excitations, the second
        all energy corrections of beta excitations (in increasing order with mo-energy
        of virtual orbital).
    """
    # constructing integrals of frozen core Hamiltonian
    nelec = sum(hf.mol.nelec)
    if isinstance(nelec_act, tuple):
        nelec_act = sum(nelec_act)
    norb = hf.mo_coeff.shape[0]
    nelec_core = nelec - nelec_act
    assert nelec_core % 2 == 0
    norb_core = int(nelec_core / 2)
    norb_virt = norb - norb_act - norb_core
    fci_frcore = CASCI(hf, norb - norb_core, nelec_act)
    h1e, _ = fci_frcore.get_h1eff()
    h2e = restore("s1", fci_frcore.get_h2eff(), norb_act + norb_virt)

    # folding 1-body term into 2-body terms
    for v in range(norb_act, norb_act + norb_virt):
        for a in range(norb_act):
            for b in range(norb_act):
                h2e[v, a, b, b] += h1e[v, a] / (nelec_act - 1)

    casci = CASCI(hf, norb_act, nelec_act)
    res = casci.kernel()
    e_cas = res[1]  # without energy of core electrons
    vec_gs = res[2]

    # CAS Hamiltonians for either alpha or beta electron removed
    assert nelec_act % 2 == 0
    neleca = nelecb = nelec_act // 2

    # sigma = alpha
    e2_valpha = np.zeros(norb_virt)
    hf.mol.nelec = (nelec // 2 - 1, nelec // 2)
    h_cas_des_a = hamil_cas(hf, norb_act, (neleca - 1, nelecb))[0]
    sh_cas_des_a = sparse.csr_array(h_cas_des_a)
    for v in range(norb_act, norb_act + norb_virt):
        e_const = e_cas - hf.mo_energy[norb_core + v]
        vec_valpha = 0

        # tau = alpha
        if neleca >= 2:
            for a in range(norb_act):
                for b in range(norb_act):
                    for c in range(norb_act):
                        tmp = des_a(vec_gs, norb_act, (neleca, nelecb), a)
                        tmp = des_a(tmp, norb_act, (neleca - 1, nelecb), c)
                        tmp = cre_a(tmp, norb_act, (neleca - 2, nelecb), b)
                        vec_valpha += h2e[v, a, b, c] * tmp

        # tau = beta
        if neleca >= 1 and nelecb >= 1:
            for a in range(norb_act):
                for b in range(norb_act):
                    for c in range(norb_act):
                        tmp = des_a(vec_gs, norb_act, (neleca, nelecb), a)
                        tmp = des_b(tmp, norb_act, (neleca - 1, nelecb), c)
                        tmp = cre_b(tmp, norb_act, (neleca - 1, nelecb - 1), b)
                        vec_valpha += h2e[v, a, b, c] * tmp
        vec_valpha = vec_valpha.flatten()
        x, fail = splinalg.minres(sh_cas_des_a, vec_valpha, shift=e_const)
        if fail:
            raise RuntimeError
        e2_valpha[v - norb_act] = -np.dot(vec_valpha, x)

    # sigma = beta
    e2_vbeta = np.zeros(norb_virt)
    hf.mol.nelec = (nelec // 2, nelec // 2 - 1)
    h_cas_des_b = hamil_cas(hf, norb_act, (neleca, nelecb - 1))[0]
    sh_cas_des_b = sparse.csr_array(h_cas_des_b)
    for v in range(norb_act, norb_act + norb_virt):
        e_const = e_cas - hf.mo_energy[norb_core + v]
        vec_vbeta = 0

        # tau = alpha
        if neleca >= 1 and nelecb >= 1:
            for a in range(norb_act):
                for b in range(norb_act):
                    for c in range(norb_act):
                        tmp = des_b(vec_gs, norb_act, (neleca, nelecb), a)
                        tmp = des_a(tmp, norb_act, (neleca, nelecb - 1), c)
                        tmp = cre_a(tmp, norb_act, (neleca - 1, nelecb - 1), b)
                        vec_vbeta += h2e[v, a, b, c] * tmp

        # tau = beta
        if nelecb >= 2:
            for a in range(norb_act):
                for b in range(norb_act):
                    for c in range(norb_act):
                        tmp = des_b(vec_gs, norb_act, (neleca, nelecb), a)
                        tmp = des_b(tmp, norb_act, (neleca, nelecb - 1), c)
                        tmp = cre_b(tmp, norb_act, (neleca, nelecb - 2), b)
                        vec_vbeta += h2e[v, a, b, c] * tmp
        vec_vbeta = vec_vbeta.flatten()
        x, fail = splinalg.minres(sh_cas_des_b, vec_vbeta, shift=e_const)
        if fail:
            raise RuntimeError
        e2_vbeta[v - norb_act] = -np.dot(vec_vbeta, x)

    # set back to original value
    hf.mol.nelec = (nelec // 2, nelec // 2)
    return e2_valpha, e2_vbeta


def e2_2actvirt(hf, norb_act, nelec_act):
    nelec = sum(hf.mol.nelec)
    if isinstance(nelec_act, tuple):
        nelec_act = sum(nelec_act)
    norb = hf.mo_coeff.shape[0]
    nelec_core = nelec - nelec_act
    assert nelec_core % 2 == 0
    norb_core = nelec_core // 2
    norb_virt = norb - norb_act - norb_core
    fci_frcore = CASCI(hf, norb - norb_core, nelec_act)
    h2e = restore("s1", fci_frcore.get_h2eff(), norb_act + norb_virt)

    casci = CASCI(hf, norb_act, nelec_act)
    res = casci.kernel()
    e_cas = res[1]  # without energy of core electrons
    vec_gs = res[2]
    neleca = nelecb = nelec_act // 2

    # sigma = alpha, tau = alpha
    e2_valpha_walpha = np.zeros((norb_virt, norb_virt))
    if neleca >= 2:
        hf.mol.nelec = (nelec // 2 - 2, nelec // 2)
        h_cas_des_aa = hamil_cas(hf, norb_act, (neleca - 2, nelecb))[0]
        sh_cas_des_aa = sparse.csr_array(h_cas_des_aa)
        for v in range(norb_act, norb_act + norb_virt):
            for w in range(norb_act, norb_act + norb_virt):
                vec_valpha_walpha = 0
                for a in range(norb_act):
                    for b in range(norb_act):
                        tmp = des_a(vec_gs, norb_act, (neleca, nelecb), a)
                        tmp = des_a(tmp, norb_act, (neleca - 1, nelecb), b)
                        vec_valpha_walpha += 0.5 * h2e[v, a, w, b] * tmp
                e_const = (
                    e_cas - hf.mo_energy[norb_core + v] - hf.mo_energy[norb_core + w]
                )
                vec_valpha_walpha = vec_valpha_walpha.flatten()
                x, fail = splinalg.minres(
                    sh_cas_des_aa, vec_valpha_walpha, shift=e_const
                )
                if fail:
                    raise RuntimeError
                e2_valpha_walpha[v - norb_act, w - norb_act] = -np.dot(
                    vec_valpha_walpha, x
                )

    # sigma = alpha, tau = beta
    e2_valpha_wbeta = np.zeros((norb_virt, norb_virt))
    if neleca >= 1 and nelecb >= 1:
        hf.mol.nelec = (nelec // 2 - 1, nelec // 2 - 1)
        h_cas_des_ab = hamil_cas(hf, norb_act, (neleca - 1, nelecb - 1))[0]
        sh_cas_des_ab = sparse.csr_array(h_cas_des_ab)
        for v in range(norb_act, norb_act + norb_virt):
            for w in range(norb_act, norb_act + norb_virt):
                vec_valpha_wbeta = 0
                for a in range(norb_act):
                    for b in range(norb_act):
                        tmp = des_a(vec_gs, norb_act, (neleca, nelecb), a)
                        tmp = des_b(tmp, norb_act, (neleca - 1, nelecb), b)
                        vec_valpha_wbeta += 0.5 * h2e[v, a, w, b] * tmp
                e_const = (
                    e_cas - hf.mo_energy[norb_core + v] - hf.mo_energy[norb_core + w]
                )
                vec_valpha_wbeta = vec_valpha_wbeta.flatten()
                x, fail = splinalg.minres(
                    sh_cas_des_ab, vec_valpha_wbeta, shift=e_const
                )
                if fail:
                    raise RuntimeError
                e2_valpha_wbeta[v - norb_act, w - norb_act] = -np.dot(
                    vec_valpha_wbeta, x
                )

    # sigma = beta, tau = alpha
    e2_vbeta_walpha = np.zeros((norb_virt, norb_virt))
    if neleca >= 1 and nelecb >= 1:
        hf.mol.nelec = (nelec // 2 - 1, nelec // 2 - 1)
        h_cas_des_ba = hamil_cas(hf, norb_act, (neleca - 1, nelecb - 1))[0]
        sh_cas_des_ba = sparse.csr_array(h_cas_des_ba)
        for v in range(norb_act, norb_act + norb_virt):
            for w in range(norb_act, norb_act + norb_virt):
                vec_vbeta_walpha = 0
                for a in range(norb_act):
                    for b in range(norb_act):
                        tmp = des_b(vec_gs, norb_act, (neleca, nelecb), a)
                        tmp = des_a(tmp, norb_act, (neleca, nelecb - 1), b)
                        vec_vbeta_walpha += 0.5 * h2e[v, a, w, b] * tmp
                e_const = (
                    e_cas - hf.mo_energy[norb_core + v] - hf.mo_energy[norb_core + w]
                )
                vec_vbeta_walpha = vec_vbeta_walpha.flatten()
                x, fail = splinalg.minres(
                    sh_cas_des_ba, vec_vbeta_walpha, shift=e_const
                )
                if fail:
                    raise RuntimeError
                e2_vbeta_walpha[v - norb_act, w - norb_act] = -np.dot(
                    vec_vbeta_walpha, x
                )

    # sigma = beta, tau = beta
    e2_vbeta_wbeta = np.zeros((norb_virt, norb_virt))
    if nelecb >= 2:
        hf.mol.nelec = (nelec // 2, nelec // 2 - 2)
        h_cas_des_bb = hamil_cas(hf, norb_act, (neleca, nelecb - 2))[0]
        sh_cas_des_bb = sparse.csr_array(h_cas_des_bb)
        for v in range(norb_act, norb_act + norb_virt):
            for w in range(norb_act, norb_act + norb_virt):
                vec_vbeta_wbeta = 0
                for a in range(norb_act):
                    for b in range(norb_act):
                        tmp = des_b(vec_gs, norb_act, (neleca, nelecb), a)
                        tmp = des_b(tmp, norb_act, (neleca, nelecb - 1), b)
                        vec_vbeta_wbeta += 0.5 * h2e[v, a, w, b] * tmp
                e_const = (
                    e_cas - hf.mo_energy[norb_core + v] - hf.mo_energy[norb_core + w]
                )
                vec_vbeta_wbeta = vec_vbeta_wbeta.flatten()
                x, fail = splinalg.minres(sh_cas_des_bb, vec_vbeta_wbeta, shift=e_const)
                if fail:
                    raise RuntimeError
                e2_vbeta_wbeta[v - norb_act, w - norb_act] = -np.dot(vec_vbeta_wbeta, x)

    # set back to original value
    hf.mol.nelec = (nelec // 2, nelec // 2)

    return e2_valpha_walpha, e2_valpha_wbeta, e2_vbeta_walpha, e2_vbeta_wbeta


def e2_from_casmatrix(hf, norb_act, nelec_act):
    res = 0
    res += np.sum(e2_1actvirt(hf, norb_act, nelec_act))
    res += 2 * np.sum(e2_2actvirt(hf, norb_act, nelec_act))
    return res


def e2_from_fcimatrix(hf, norb_act, nelec_act, casscf_actspace=None):
    h_dyall, v = perturbation_ops(hf, norb_act, nelec_act, casscf_actspace)
    return e2_general(h_dyall, v)


def e_nevpt2(rhf, norb_act: int, nelec_act: int, frcore=False) -> float:
    """
    Computes the strongly-contracted NEVPT2 energy, implemented by PySCF

    :param rhf: converged PySCF RHF object
    :param norb_act: number of active orbitals
    :param nelec_act: number of active electrons
    :param frcore: whether the core should be kept frozen
    :return: the cas energy plus nevpt2 energy correction
    """
    cas = CASCI(rhf, norb_act, nelec_act)
    e_casci = cas.kernel()[0]
    if frcore:
        nevpt = mrpt.NEVPT(cas)

        dm1, dm2, dm3 = fci.rdm.make_dm123(
            "FCI3pdm_kern_sf", nevpt.load_ci(), nevpt.load_ci(), norb_act, nelec_act
        )
        dms = {"1": dm1, "2": dm2, "3": dm3}
        eris = mrpt.nevpt2._ERIS(nevpt._mc, nevpt.mo_coeff)
        _, e_Sr = mrpt.nevpt2.Sr(nevpt._mc, nevpt.load_ci(), dms, eris)
        _, e_Srs = mrpt.nevpt2.Srs(nevpt._mc, dms, eris)
        e_nevtp2_frcore = e_Sr + e_Srs
        print(f"{e_nevtp2_frcore=}")
        return e_casci + e_nevtp2_frcore
    else:
        e_nevpt2 = mrpt.NEVPT2(cas)
        print(f"{e_nevpt2=}")
        return e_casci + e_nevpt2


def e2_general(h, v=None, vec=None, method="linear system"):
    if method == "linear system":
        e_gs, vec = splinalg.eigsh(h, k=1, which="SA")
        vec = vec[:, 0]
        id = np.identity(h.shape[0])
        Pi = id - np.outer(vec, vec)
        np.matmul(v, vec, out=vec)
        np.matmul(Pi, vec, out=vec)
        return np.dot(vec, splinalg.spsolve(e_gs * id - h, vec))
    elif method == "resolvent":
        e_gs, gs = splinalg.eigsh(h, k=1, which="SA")
        gs = gs[:, 0]
        id = np.identity(h.shape[0])
        Pi = id - np.outer(gs, gs)
        if sparse.issparse(e_gs * id - h):
            R0 = splinalg.inv(e_gs * id - h)
        else:
            R0 = np.linalg.inv(e_gs * id - h)
        return gs.T @ v @ Pi @ R0 @ Pi @ v @ gs
    elif method == "over spectrum":
        energies, eigvecs = np.linalg.eigh(h)
        numerators = (eigvecs[:, 0].T @ v @ eigvecs[:, 1:]) ** 2
        denominators = energies[0] - energies[1:]
        return np.sum(numerators / denominators)
