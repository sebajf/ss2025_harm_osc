from sympy import symbols, diff, simplify, lambdify
import matplotlib.pyplot as plt
import jax.numpy as jnp

name = "Harmonic Oscillator (sympy version)"

def DEL_equations_harmonic_osc():
    # Define symbols (internal name and display name)
    q0, q1, q2, h, m, k = symbols("q0 q1 q2 h m k")

    # Discrete Lagrangian
    L_d = h * (m / 2 * ((q1 - q0) / h) ** 2 - k / 2 * ((q1 + q0) / 2) ** 2)

    # Compute derivatives
    D1_Ld = simplify(diff(L_d, q0))
    D2_Ld = simplify(diff(L_d, q1))
    D1_Ld = D1_Ld.subs({q0: q1, q1: q2}, simultaneous=True)
    DEL_equations = simplify(D1_Ld + D2_Ld)
    DEL_lambdified = lambdify([q0, q1, q2, h, m, k], DEL_equations, "numpy")

    def f_DEL_q2_first(q2, q0, q1, h, phys_params):
        # (Could check if phys_params contains the required keys)
        m, k = phys_params["m"], phys_params["k"]
        return DEL_lambdified(q0, q1, q2, h, m, k)

    return f_DEL_q2_first

f_DEL_q2_first = DEL_equations_harmonic_osc()

def plot_results(trajectory, h, N, filename="harmonic_oscillator_trajectory.png"):
    plt.figure(figsize=(10, 6))
    time_points = h * jnp.arange(N + 1)
    plt.plot(time_points, trajectory, "bo-", label="Oscillator Position")
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.title("Discrete Harmonic Oscillator Trajectory")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
