from scipy.optimize import minimize_scalar


def dissociation_energy(e_func, r_eq=None):
    """
    parameters:
        e_func: function that reads bond length and returns energy
        r_eq: float, the equilibrium bond distance in Angstrom. Will
                be optimized if not known.
    returns:
        dissociation energy
    """
    e_dissoc = e_func(1000)
    if r_eq is None:
        r_eq = equilibrium_bond_length(e_func)
    e_eq = e_func(r_eq)
    return e_dissoc - e_eq


def equilibrium_bond_length(e_func):
    """
    parameters:
        e_func: function that reads bond-length and returns energy

    returns:
        equilibrium bond length
    """

    print("\n\n" + 10 * "#" + " START diatomic minimization " + 10 * "#")
    r_eq = minimize_scalar(
        e_func, bracket=(0.7, 1.7), method="Brent", tol=1e-5, options={"disp": 3}
    ).x
    print("\n\n" + 10 * "#" + " END diatomic minimization " + 10 * "#")
    return r_eq
