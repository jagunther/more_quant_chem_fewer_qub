from pyscf import gto, scf, mrpt
from pyscf.mcscf import CASCI
import numpy as np
from more_quant_chem_fewer_qub.e2 import e2_from_casmatrix

hooh_xyz = "hydrogenperoxide_casci10o14e_def2tzvp.xyz"
ts_xyz = "transitionstate_casci10o14e_def2tzvp.xyz"
h2oo_xyz = "oxywater_casci10o14e_def2tzvp.xyz"

BASIS = "def2tzvp"

KJMOL_PER_HARTREE = 2625.5002


def hf(xyz):
    mol = gto.M(atom=xyz, basis=BASIS, spin=0)
    hf = scf.RHF(mol).newton()
    hf.run()
    return hf


def e_cas_mrpt2():
    hf_hooh = hf(hooh_xyz)
    hf_ts = hf(ts_xyz)
    hf_h2oo = hf(h2oo_xyz)
    e_rhf = np.array([hf.kernel() for hf in [hf_hooh, hf_ts, hf_h2oo]])
    e_mrpt2 = np.array(
        [e2_from_casmatrix(hf, 10, 14) for hf in [hf_hooh, hf_ts, hf_h2oo]]
    )
    casci_hooh = CASCI(hf_hooh, 10, 14)
    casci_ts = CASCI(hf_ts, 10, 14)
    casci_h2oo = CASCI(hf_h2oo, 10, 14)

    e_casci = np.array(
        [casci.kernel()[0] for casci in [casci_hooh, casci_ts, casci_h2oo]]
    )
    e_casci_mrpt2 = np.array([e0 + e2 for (e0, e2) in zip(e_casci, e_mrpt2)])
    return e_rhf, e_casci, e_casci_mrpt2


def e_cas_nevpt():
    # strongly contracted NEVPT2
    hf_hooh = hf(hooh_xyz)
    hf_ts = hf(ts_xyz)
    hf_h2oo = hf(h2oo_xyz)
    casci_hooh = CASCI(hf_hooh, 10, 14)
    casci_ts = CASCI(hf_ts, 10, 14)
    casci_h2oo = CASCI(hf_h2oo, 10, 14)
    e_casci = np.array(
        [casci.kernel()[0] for casci in [casci_hooh, casci_ts, casci_h2oo]]
    )

    # we want the NEVPT2 energy in the frozen core approximation, but PySCF only computes the total
    # one. However, the 8 contributions are printed when the method is run, and Sr(-1) + Srs(-2) are
    # the NEVPT2 energy correction in the frozen core approximation.
    e_nevpt2 = [mrpt.NEVPT2(cas) for cas in [casci_hooh, casci_ts, casci_h2oo]]
    e_nevpt2_fc = [
        -0.09815626499877 - 0.30714701720375,
        -0.10117664520385 - 0.30270898012845,
        -0.08943903384320 - 0.30492520555448,
    ]
    e_casci_nevpt2 = np.array(e_casci) + np.array(e_nevpt2)
    e_casci_nevpt2_fc = np.array(e_casci) + np.array(e_nevpt2_fc)
    return e_casci, e_casci_nevpt2, e_casci_nevpt2_fc


if __name__ == "__main__":
    e_rhf, _, e_casci_mrpt2 = e_cas_mrpt2()
    e_casci, __, e_casci_nevpt2_fc = e_cas_nevpt()

    e_rhf *= KJMOL_PER_HARTREE
    e_casci_mrpt2 *= KJMOL_PER_HARTREE
    e_casci *= KJMOL_PER_HARTREE
    e_casci_nevpt2_fc *= KJMOL_PER_HARTREE

    e_rhf_barrier = e_rhf[1] - e_rhf[2]
    e_casci_barrier = e_casci[1] - e_casci[2]
    e_casci_nevpt2_fc_barrier = e_casci_nevpt2_fc[1] - e_casci_nevpt2_fc[2]
    e_casci_mrpt2_barrier = e_casci_mrpt2[1] - e_casci_mrpt2[2]

    print("\nHOOH     TS      H2OO    (in kJ/mol)")
    print(f"{e_rhf=}")
    print(f"{e_casci=}")
    print(f"{e_casci_nevpt2_fc=}")
    print(f"{e_casci_mrpt2=}")
    print(f"{e_rhf_barrier=}")
    print(f"{e_casci_barrier=}")
    print(f"{e_casci_nevpt2_fc_barrier=}")
    print(f"{e_casci_mrpt2_barrier=}")

# output:
#
# HOOH     TS      H2OO    (in kJ/mol)
# e_rhf=array([-396015.58144204, -395792.72433514, -395869.61453619])
# e_casci=array([-396117.08288595, -395891.00189045, -395935.89362369])
# e_casci_nevpt2_fc=array([-397181.20673444, -396951.40368054, -396971.29701311])
# e_casci_mrpt2=array([-397201.6317235 , -396975.35169536, -396991.45706543])
# e_rhf_barrier=76.89020105224336
# e_casci_barrier=44.89173324115109
# e_casci_nevpt2_fc_barrier=19.893332565552555
# e_casci_mrpt2_barrier=16.105370064789895
