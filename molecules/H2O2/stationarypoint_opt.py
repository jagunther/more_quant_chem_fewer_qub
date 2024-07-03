from pyscf import gto, scf
from pyscf.mcscf import CASCI
from pyscf.geomopt.geometric_solver import optimize

BASIS = "def2tzvp"


def opt_casci():
    mol = gto.M(atom="hydrogenperoxide_rhf_def2tzvp.xyz", basis=BASIS, spin=0)
    hf = scf.RHF(mol).newton()
    hf.kernel()
    casci = CASCI(hf, 10, 14)
    mol_eq = optimize(casci)
    mol_eq.tofile("hydrogenperoxide_casci10o14e_def2tzvp.xyz", format="xyz")

    mol = gto.M(atom="transitionstate_rhf_def2tzvp.xyz", basis=BASIS, spin=0)
    hf = scf.RHF(mol).newton()
    casci = CASCI(hf, 10, 14)
    params = {"transition": True}
    mol_ts = casci.Gradients().optimizer(solver="geomeTRIC").kernel(params)
    mol_ts.tofile("transitionstate_casci10o14e_def2tzvp.xyz", format="xyz")

    mol = gto.M(atom="oxywater_rhf_def2tzvp.xyz", basis=BASIS, spin=0)
    hf = scf.RHF(mol).newton()
    casci = CASCI(hf, 10, 14)
    mol_eq = optimize(casci)
    mol_eq.tofile("oxywater_casci10o14e_def2tzvp.xyz")


def main():
    opt_casci()


if __name__ == "__main__":
    main()
