import numpy as np
from pyscf import fci
from pyscf.tools import fcidump
from pyscf.mcscf import CASCI, CASSCF
import os
import scipy


def hamil_from_fcidump(fcidumpfile):
    """
    Computes the Hamiltonian in matrix form from an FCIDUMP file

    :param fcidumpfile: str, name of FCIDUMP file
    :return: 2d-np.array, the Hamiltonian
    """
    data = fcidump.read(fcidumpfile)
    assert data["NELEC"] % 2 == 0
    assert data["MS2"] == 0

    nelec = data["NELEC"]
    norb = data["NORB"]
    h1e = data["H1"]
    h2e = data["H2"]
    dim = int(scipy.special.binom(norb, nelec / 2) ** 2)
    nelec = (int(nelec / 2), int(nelec / 2))
    print(f"construction of CI Hamiltonian, {dim=}")
    return fci.direct_spin1.pspace(h1e, h2e, norb, nelec, np=dim)[1]


def make_dyall_fcidump(fcidumpfile_in, fcidumpfile_out, norb_act, mo_energy):
    """
    Reads an FCIDUMP file and constructs the Dyall Hamiltonian as a new FCIDUMP
    file for the given number of active orbitals and MO energies.
    The active space goes from 0 to norb_act (so either there are no core orbitals,
    or the core orbitals are already folded into the Hamiltonian as an effective 1-body
    interaction).

    :param fcidumpfile_in: str, the file to read
    :param fcidumpfile_out: str, the file to output to
    :param norb_act: number of active orbitals
    :param mo_energy: list of float
    :return: None
    """
    assert os.path.exists(fcidumpfile_out) == False

    with open(fcidumpfile_in, "r") as f:
        file = f.read()

    # extracting header
    header_end = None
    for idx, line in enumerate(file.splitlines()):
        if "&END" in line:
            header_end = idx
            break
    fcidump_dyall = file.splitlines()[:header_end]
    assert "NORB" in fcidump_dyall[0]
    first_line = fcidump_dyall[0].replace(" ", "")
    norb = int(first_line.split("NORB=")[1].split(",")[0])
    assert len(mo_energy) == norb

    cas_idx = np.arange(0, norb_act) + 1  # fcidump starts with index 1
    virt_idx = np.arange(norb_act, norb) + 1  # fcidump starts with index 1

    # extracting CAS part
    for line in file.splitlines()[header_end:]:
        indices = set(map(int, line.split()[1:]))
        # 0 is not normal index, denotes 1-electron and constant term
        if indices.issuperset([0]):
            indices.remove(0)
        # only take into account if all indices are in active space
        if indices.issubset(cas_idx):
            fcidump_dyall.append(line)

    # adding diagonal virtual part
    for idx in virt_idx:
        e_virt = mo_energy[idx - 1]  # 0 indexed
        line = f"{e_virt}    {idx}    {idx}    0    0"
        fcidump_dyall.append(line)
    fcidump_dyall = "\n".join(fcidump_dyall)

    # write to FCIDUMP
    with open(fcidumpfile_out, "w") as f:
        f.write(fcidump_dyall)


def perturbation_ops_from_fcidump(fcidumpfile, norb_act, mo_energy):
    """
    Given a full Hamiltonian H encoded in a FCIDUMP file, this method constructs the
    MRPT2 Dyall Hamiltonian H0 for the specified active space and the perturbation
    operator V = H - H0 and returns them in matrix form.
    The active space goes from 0 to norb_act (so either there are no core orbitals,
    or the core orbitals are already folded into the Hamiltonian as an effective 1-body
    interaction).

    :param fcidumpfile: str, name of FCIDUMP file. It should contain the full Hamiltonian (in the
                        frozen core approximation).
    :param norb_act: number of orbitals to include in the active space
    :param mo_energy: energies of Molecular orbitals
    :return: two 2d-np.arrays, the Dyall Hamiltonian and the perturbation
    """
    fcidumpfile_dyall = fcidumpfile + "_dyall"
    make_dyall_fcidump(fcidumpfile, fcidumpfile_dyall, norb_act, mo_energy)
    h_fci = hamil_from_fcidump(fcidumpfile)
    h_dyall = hamil_from_fcidump(fcidumpfile_dyall)
    os.remove(fcidumpfile_dyall)
    return h_dyall, h_fci - h_dyall


def perturbation_ops(hf, norb_act, nelec_act, casscf_actspace=None):
    """
    Constructs the MRPT2 Dyall Hamiltonian and the perturbation operator in frozen core
    approximation and returns them in matrix form.
    By default, canonical RHF orbitals are used for the active space, their energies give the
    virtual diagonal part of the Dyall-Hamiltonian.
    If casscf_actspace parameter is passed, orbitals of a CASSCF wavefunction are used for the CASCI active
    calculation, and the canonicalized virtual orbital energies give the diagonal part of the
    Dyall Hamiltonian.

    :param xyz: str, the nuclear geometry
    :param basis: str, basis set for quantum chemical calculation
    :param norb_act: int, number of active orbitals
    :param nelec_act: int, number of active electrons.
    :param casscf_actspace=None: tuple of int, the active space for the CASSCF calculation.
                        If None, only a RHF calculation will be done.
    :return: two 2d-np.arrays, the Dyall Hamiltonian and the perturbation
    """

    if casscf_actspace is not None:
        casscf = CASSCF(hf, *casscf_actspace)
        casscf.fix_spin(ss=0)
        casscf.kernel()
        hf.mo_coeff = casscf.mo_coeff
        mo_energy = casscf.mo_energy
    else:
        mo_energy = hf.mo_energy

    # constructing frozen core Hamiltonian
    nelec = sum(hf.mol.nelec)
    if isinstance(nelec_act, tuple):
        nelec_act = sum(nelec_act)
    norb = hf.mo_coeff.shape[0]
    nelec_core = nelec - nelec_act
    assert nelec_core % 2 == 0
    norb_core = int(nelec_core / 2)
    casci = CASCI(hf, norb - norb_core, nelec_act)
    fcidump_name = "FCIDUMP_frozencore_temporary"
    fcidump.from_mcscf(casci, fcidump_name)

    # get operators
    mo_energy = mo_energy[norb_core:]
    h_dyall, v = perturbation_ops_from_fcidump(fcidump_name, norb_act, mo_energy)
    os.remove(fcidump_name)
    return h_dyall, v
