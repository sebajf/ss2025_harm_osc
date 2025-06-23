import jax
import harm_osc_system_sympy  # for recycling the plotting function

name = "Harmonic Oscillator (JAX version)"

def L(q, v, phys_params):
    m, k = phys_params["m"], phys_params["k"]
    return m / 2 * v**2 - k / 2 * q**2


# midpoint discretization for any L
def L_d(q0, q1, h, phys_params):
    return h * L((q1 + q0) / 2, (q1 - q0) / h, phys_params)


D1_Ld = jax.jacfwd(L_d, argnums=0)  # The default, actually
D2_Ld = jax.jacfwd(L_d, argnums=1)


def DEL_equations_harmonic_osc(q0, q1, q2, h, phys_params):
    return D1_Ld(q1, q2, h, phys_params) + D2_Ld(q0, q1, h, phys_params)

@jax.jit
def f_DEL_q2_first(q2, q0, q1, h, phys_params):
    """"
    This function is used to solve for q2 in the DEL equations.
    It is structured to match the expected input for `scipy.optimize.root`.
    """
    return DEL_equations_harmonic_osc(q0, q1, q2, h, phys_params)

# Our plotting function is the same as in the sympy version, so we might as well
# use it directly.
plot_results = harm_osc_system_sympy.plot_results
