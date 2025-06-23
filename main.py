import harm_osc_system_jax as the_system  # Switch sympy and jax versions here
import integrator
import jax.numpy as jnp


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
    q0_q1 = jnp.array([1.0, 1.1])
    return h, N, q0_q1


def main():
    f_DEL_q2_first = the_system.f_DEL_q2_first
    phys_params = parameters_example_1()
    h, N, q0_q1 = simulation_parameters_short_time()
    trajectory = integrator.perform_simulation(f_DEL_q2_first, phys_params, h, N, q0_q1)
    the_system.plot_results(
        trajectory,
        h,
        N,
        filename="harm_osc_example1_short_time.png",
    )


if __name__ == "__main__":
    print(f"Simulating {the_system.name}...")
    main()
    print("Done.")
