from more_quant_chem_fewer_qub.diatomic import equilibrium_bond_length
import numpy as np
import matplotlib.pyplot as plt
from more_quant_chem_fewer_qub.cas import e_cas
from pyscf import gto, scf
from more_quant_chem_fewer_qub.e2 import e2_from_fcimatrix, e_nevpt2
import pickle

basis_BH = "631g"
xyz = "B 0 0 0; H x 0 0"


def rhf(r, symm=False, spin=0, rdm1_guess=None):
    mol = gto.M(atom=xyz.replace("x", str(r)), basis=basis_BH, spin=spin, symmetry=symm)
    hf = scf.RHF(mol).newton()
    hf.kernel(rdm1_guess)

    mo_upd, _, stable, _ = hf.stability(return_status=True)

    if not stable and r > 7:
        hf_r7 = rhf(7, symm=symm, spin=spin)
        dm_r7 = hf_r7.make_rdm1(hf_r7.mo_coeff, hf_r7.mo_occ)
        hf.run(dm_r7)
        mo_upd, _, stable, _ = hf.stability(return_status=True)

    iter = 0
    while not stable and iter < 10:
        dm = hf.make_rdm1(mo_upd, hf.mo_occ)
        hf = hf.run(dm)
        mo_upd, _, stable, _ = hf.stability(return_status=True)
        iter += 1
    if not stable:
        raise RuntimeError("RHF calculation did not converge")

    return hf


def e_rhf(hf):
    return hf.e_tot


def e_casci54(hf, ci0=None):
    return e_cas(hf, 5, (2, 2), nroots=1, ci0=ci0)[0]


def e_fci_frcore(hf, ci0=None):
    return e_cas(hf, 10, (2, 2), nroots=1, ci0=ci0)[0]


def e_casci54_mrpt2(hf, ci0=None):
    return e_casci54(hf, ci0=ci0) + e2_from_fcimatrix(hf, 5, (2, 2))


def e_casci54_nevpt2(hf, ci0=None):
    return e_nevpt2(hf, norb_act=5, nelec_act=4, frcore=True)


def main():
    equilib_bond = []
    e_eq = []
    for method in [e_rhf, e_casci54, e_casci54_mrpt2, e_fci_frcore, e_casci54_nevpt2]:
        e_func = lambda x: method(rhf(x))
        equilib_bond.append(equilibrium_bond_length(e_func))
        e_eq.append(e_func(equilib_bond[-1]))
        print(f"\n{e_eq=}\n")
    print(f"\n{e_eq=}\n")
    print(f"\n{equilib_bond=}\n")

    e_separate = []
    e_pes = {
        "RHF": [],
        "CASCI(5,4)": [],
        "FCI frcore": [],
        "MRPT2": [],
        "SC-NEVPT2": [],
    }
    # by carrying out pes scan to r=40, the final ci-vector leads to good initial
    # guess for the far separated calculation (could perhaps be done in a smarter way)
    r_pes = np.concatenate(
        (np.arange(0.6, 1.7, 0.05), np.arange(1.8, 5, 0.1), np.arange(6, 40, 1))
    )
    rdm1 = None
    ci0_cas = None
    ci0_fci = None
    for r in r_pes:
        print(f"{r=}")
        hf = rhf(r, rdm1_guess=rdm1)
        rdm1 = hf.make_rdm1()
        e_pes["RHF"].append(e_rhf(hf))
        e, ci0_cas = e_cas(hf, 5, (2, 2), ci0=ci0_cas)
        e_pes["CASCI(5,4)"].append(e)
        e_pes["MRPT2"].append(e_casci54_mrpt2(hf, ci0=ci0_cas))
        e, ci0_fci = e_cas(hf, 10, (2, 2), ci0=ci0_fci)
        e_pes["FCI frcore"].append(e)
        e_pes["SC-NEVPT2"].append(e_casci54_nevpt2(hf, ci0=ci0_cas))

    # the last rdm1 and ci0 should be very good estimates for the separated atoms
    hf_sep = rhf(500, rdm1_guess=rdm1)
    e_separate.append(e_rhf(hf_sep))
    e_separate.append(e_casci54(hf_sep, ci0=ci0_cas))
    e_separate.append(e_casci54_mrpt2(hf_sep, ci0_cas))
    e_separate.append(e_fci_frcore(hf_sep, ci0=ci0_fci))
    e_separate.append(e_casci54_nevpt2(hf_sep, ci0=ci0_cas))
    e_diss = np.array(e_separate) - np.array(e_eq)

    print(f"{equilib_bond=}")
    print(f"{e_diss=}")
    print(f"{e_eq=}")
    print(f"{e_separate=}")
    print(f"{e_pes=}")

    for label, e in e_pes.items():
        plt.plot(r_pes, e, label=label)
    plt.legend()
    plt.show()

    data = {"e_pes_scan": e_pes, "r_pes_scan": r_pes}
    print(data)
    with open("BH_data_631G.pkl", "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()
