from pyscf import scf, gto, ao2mo
from pyscf.mcscf import CASCI
from more_quant_chem_fewer_qub.norm import normv23_mrpt_abs, norm_mrpt_fulldifference
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

font = {"family": "serif", "size": 26}
matplotlib.rc("font", **font)


def make_hydrogenchain_xyz(n, d=0.75):
    xyz = ""
    for i in range(n):
        xyz = xyz + f"H{i}  {i*d}  0   0\n"
    xyz.rstrip("\n")
    return xyz


def main():
    nmax = 16
    BASIS = "631g"
    normv23_abs = []
    normv23_fulldiff = []
    x = [2 * i for i in range(1, nmax)]
    for n in x:
        print(f"\n n = {n} \n")
        xyz = make_hydrogenchain_xyz(n)
        mol = gto.M(atom=xyz, basis=BASIS)
        hf = scf.RHF(mol).newton()
        hf.kernel()

        nelec = sum(hf.mol.nelec)
        norb = mol.nao
        norb_act = n
        norb_virt = norb - norb_act
        print(f"{nelec=}")
        print(f"{norb_virt=}")
        print(f"{norb_act=}")
        fci = CASCI(hf, norb, nelec)

        h1e, efrcore = fci.h1e_for_cas()
        efrcore = efrcore - fci.energy_nuc()
        h2e = fci.get_h2cas()
        h2e = ao2mo.restore(1, h2e, norb)
        e_virt = hf.mo_energy[-norb_virt:]
        normv23_abs.append(normv23_mrpt_abs(h1e, h2e, norb_act))
        normv23_fulldiff.append(
            norm_mrpt_fulldifference(h1e, h2e, norb_act, e_virt, efrcore)[1] ** 2
        )
        print(f"{normv23_fulldiff[-1]=:.3g}")
        print(f"{normv23_abs[-1]=:.3g}")

    xfit = 2 * np.array(x)  # need to multiply by 2 because of spin

    normv23_fulldiff = np.array(normv23_fulldiff)
    poly_normv23_diff = np.polyfit(np.log(xfit), np.log(normv23_fulldiff), deg=1)
    poly_normv23_abs = np.polyfit(np.log(xfit), np.log(normv23_abs), deg=1)
    xfit_fine = np.linspace(1, 4 * nmax)
    fit_normv23_diff = np.exp(poly_normv23_diff[1]) * xfit_fine ** poly_normv23_diff[0]
    fit_normv23_abs = np.exp(poly_normv23_abs[1]) * xfit_fine ** poly_normv23_abs[0]
    print(f"{poly_normv23_diff=}")
    print(f"{poly_normv23_abs=}")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(xfit, normv23_fulldiff, label=(r"$\|v_{diff}\|_{2/3}^2$"), lw=4)
    ax.scatter(xfit, normv23_abs, label=r"$\|v^{MRPT2}\|_{2/3}$", lw=4)
    ax.plot(xfit_fine, fit_normv23_diff, color="black", zorder=10, ls="--")
    ax.plot(xfit_fine, fit_normv23_abs, color="black", zorder=10, ls="--")
    ax.set(
        yscale="log",
        xscale="log",
        xlabel="number of active orbitals " + r"$k$",
        ylim=(1e1, 1e18),
    )
    for i in range(1, 19):
        ax.axhline(10 ** (i), lw=0.5, color="lightgrey", zorder=-10)
    ax.legend(framealpha=1)
    plt.show()
    fig.savefig("normv_hydrogenchain.pdf", bbox_inches="tight")

    return


if __name__ == "__main__":
    main()
