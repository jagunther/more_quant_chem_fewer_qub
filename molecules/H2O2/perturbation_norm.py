from pyscf import scf, gto, ao2mo
from pyscf.mcscf import CASCI
from more_quant_chem_fewer_qub.norm import (
    norm_mrpt_fulldifference,
    normv23_mrpt,
    normv23_mrpt_abs,
)
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

font = {"family": "serif", "size": 26}
matplotlib.rc("font", **font)


def main():
    hooh_xyz = "hydrogenperoxide_casci10o14e_def2tzvp.xyz"

    BASIS = "def2tzvp"
    mol_hooh = gto.M(atom=hooh_xyz, basis=BASIS)

    hf_hooh = scf.RHF(mol_hooh).newton()
    hf_hooh.kernel()
    norb_act = 10
    nelec_act = 14
    nelec = sum(hf_hooh.mol.nelec)
    norb = hf_hooh.mol.nao
    norb_core = (nelec - nelec_act) // 2
    norb_virt = norb - norb_act - norb_core
    print(f"{norb_virt=}")
    fci_frcore = CASCI(hf_hooh, norb_act + norb_virt, nelec_act)

    norb = norb_act + norb_virt  # not counting frozen core orbitals
    h1e, efrcore = fci_frcore.h1e_for_cas()
    efrcore = efrcore - fci_frcore.energy_nuc()
    h2e = fci_frcore.get_h2cas()
    h2e = ao2mo.restore(1, h2e, norb)
    e_virt = hf_hooh.mo_energy[-norb_virt:]

    normh1, normv23_diff = norm_mrpt_fulldifference(h1e, h2e, norb_act, e_virt, efrcore)
    print(f"{normh1=:.3g}")
    print(f"{normv23_diff ** 2 =:.3g}")
    normv23_abs = normv23_mrpt_abs(h1e, h2e, norb_act)
    print(f"{normv23_abs=:.3g}")
    times = [i * 10 for i in range(10)]
    normv23_time = normv23_mrpt(h1e, h2e, norb_act, e_virt, efrcore, times)
    for t, normv23 in zip(times, normv23_time):
        print(f"{t=}, {normv23=:.3g}")

    normv23_abs_incr_virt = []
    normv23_diff_incr_virt = []
    for i in range(1, norb_virt + 1):
        h1e_curr = h1e[: norb_act + i, : norb_act + i]
        h2e_curr = h2e[: norb_act + i, : norb_act + i, : norb_act + i, : norb_act + i]
        e_virt_curr = e_virt[:i]
        normv23_abs_curr = normv23_mrpt_abs(h1e_curr, h2e_curr, norb_act)
        normv23_diff_curr = norm_mrpt_fulldifference(
            h1e_curr, h2e_curr, norb_act, e_virt_curr, efrcore
        )[1]
        normv23_abs_incr_virt.append(normv23_abs_curr)
        normv23_diff_incr_virt.append(normv23_diff_curr)
    normv23_diff_incr_virt = np.array(normv23_diff_incr_virt)

    # need to multiply by 2 because of spin
    x = [2 * i for i in range(1, norb_virt + 1)]

    af = 15  # start of asymptotic fit
    poly_normv23_diff = np.polyfit(
        np.log(x)[af:], np.log(normv23_diff_incr_virt**2)[af:], deg=1
    )
    poly_normv23_abs = np.polyfit(
        np.log(x)[af:], np.log(normv23_abs_incr_virt)[af:], deg=1
    )
    print(f"{poly_normv23_diff=}")
    print(f"{poly_normv23_abs=}")

    fit_normv23_diff = np.exp(poly_normv23_diff[1]) * x ** poly_normv23_diff[0]
    fit_normv23_abs = np.exp(poly_normv23_abs[1]) * x ** poly_normv23_abs[0]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(
        x,
        [n**2 for n in normv23_diff_incr_virt],
        label=r"$\|v_{diff}\|_{2/3}^2$",
        lw=3,
    )
    ax.scatter(x, normv23_abs_incr_virt, label=r"$\|v^{MRPT2}\|_{2/3}$", lw=3)
    ax.plot(x, fit_normv23_diff, color="black", zorder=1, ls="--")
    ax.plot(x, fit_normv23_abs, color="black", zorder=1, ls="--")
    ax.set(
        yscale="log",
        xscale="log",
        xlabel="number of virtual orbitals " + r"$K-k$",
        ylim=[5e5, 1e19],
    )
    for i in range(6, 19):
        ax.axhline(10 ** (i), lw=0.5, color="lightgrey", zorder=-10)
    ax.legend(framealpha=1, loc="upper left")
    plt.show()
    fig.savefig("normv_hooh.pdf", bbox_inches="tight")

    return


if __name__ == "__main__":
    main()
