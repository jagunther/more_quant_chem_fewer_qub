import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle

font = {"family": "serif", "size": 26}
matplotlib.rc("font", **font)
data_file = "BH_data_631G.pkl"


def main():
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    print(data["e_pes_scan"].keys())

    xlim = [0.58, 4]
    r_pes = data["r_pes_scan"]
    e_RHF = np.array(data["e_pes_scan"]["RHF"])
    e_CASCI54 = np.array(data["e_pes_scan"]["CASCI(5,4)"])
    e_CASCI54_mrpt2 = np.array(data["e_pes_scan"]["MRPT2"])
    e_CASCI54_nevpt2 = np.array(data["e_pes_scan"]["SC-NEVPT2"])
    e_FCI_frozen_core = np.array(data["e_pes_scan"]["FCI frcore"])

    fig, axes = plt.subplots(
        2, 1, figsize=(14.5, 9), gridspec_kw={"height_ratios": [2, 1]}
    )
    fig.subplots_adjust(hspace=0.1)
    axes[0].plot(r_pes, e_RHF, label="RHF", lw=3)
    axes[0].plot(r_pes, e_CASCI54, label="CASCI(5o,4e)", lw=3)
    axes[0].plot(r_pes, e_CASCI54_mrpt2, label="CASCI(5o,4e) + MRPT2", lw=3)
    axes[0].plot(
        r_pes, e_CASCI54_nevpt2, label="CASCI(5o,4e) + SC-NEVPT2", lw=4, ls=(0, (3, 3))
    )
    axes[0].plot(r_pes, e_FCI_frozen_core, label="FCI frozen core", lw=3, color="black")
    axes[0].legend(loc=(0.083, 0.51), fontsize=20)
    axes[0].set_ylabel(r"Energy / H")
    axes[0].set_xticklabels([])
    axes[0].set_xlim(xlim)
    axes[0].set_ylim([-25.18, -24.8])

    # plotting differences in mHartree
    axes[1].plot(r_pes, 1e3 * (e_CASCI54 - e_FCI_frozen_core), color="C1", lw=3)
    axes[1].plot(r_pes, 1e3 * (e_CASCI54_mrpt2 - e_FCI_frozen_core), color="C2", lw=3)
    axes[1].plot(
        r_pes,
        1e3 * (e_CASCI54_nevpt2 - e_FCI_frozen_core),
        color="C3",
        lw=4,
        ls=(0, (3, 3)),
    )
    axes[1].set_ylim([0, 42])
    axes[1].set_xlim(xlim)
    axes[1].set_yticks([0, 10, 20, 30, 40])
    axes[1].set_yticklabels(["0", "", "20", "", "40"])
    for y in [10, 20, 30, 40]:
        axes[1].axhline(y, lw=0.5, color="lightgrey", zorder=-1)
    axes[1].set_xlabel(r"atomic distance in $\AA$")
    axes[1].set_ylabel(r"$E - E_{FCI}$ / mH")
    plt.show()
    fig.savefig("pes_scan_BH_6-31G.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
