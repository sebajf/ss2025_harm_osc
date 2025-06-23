import harm_osc_system_sympy
import integrator
import numpy as np


def parameters_example_1():
    # Return physical parameters as a dictionary
    phys_params = {
        "m": 1.0,  # Mass
        "k": 1.0,  # Spring constant
    }
    return phys_params


def simulation_parameters_short_time():
    h = 0.1  # Time step
    N = 100  # Number of steps
    q0_q1 = np.array([1.0, 1.1])
    return h, N, q0_q1


def main():
    f_DEL_q2_first = harm_osc_system_sympy.DEL_equations_harmonic_osc()
    phys_params = parameters_example_1()
    h, N, q0_q1 = simulation_parameters_short_time()
    trajectory = integrator.perform_simulation(f_DEL_q2_first, phys_params, h, N, q0_q1)
    harm_osc_system_sympy.plot_results(
        trajectory,
        h,
        N,
        filename="harm_osc_example1_short_time.png",
    )


if __name__ == "__main__":
    main()
