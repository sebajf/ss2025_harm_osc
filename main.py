import harm_osc_system_jax as the_system  # Switch sympy and jax versions here
import integrator
import jax.numpy as jnp
import jax


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


def show_off_some_derivatives():
    # Let's show off some of JAX's automatic differentiation capabilities!
    f_DEL_q2_first = the_system.f_DEL_q2_first
    phys_params = parameters_example_1()
    h, N, q0_q1 = simulation_parameters_short_time()

    # Sensitivity of qN wrt initial conditions
    def final_point(f_DEL_q2_first, phys_params, h, N, q0, q1):
        trajectory = integrator.perform_simulation_with_split_initial_conditions(
            f_DEL_q2_first, phys_params, h, N, q0, q1
        )
        return trajectory[-1]  # The -1 "wraps around" to the last element

    # Notice that q0 and q1 are the last two arguments of the final_point function.
    partial_qN_partial_q0 = jax.jacfwd(final_point, argnums=4)
    partial_qN_partial_q1 = jax.jacfwd(final_point, argnums=5)
    # The two objects above are functions, not numbers. We still need to evaluate them.

    q0, q1 = q0_q1
    print(
        "Sensitivity of final point wrt q0:",
        partial_qN_partial_q0(f_DEL_q2_first, phys_params, h, N, q0, q1),
    )
    print(
        "Sensitivity of final point wrt q1:",
        partial_qN_partial_q1(f_DEL_q2_first, phys_params, h, N, q0, q1),
    )

    partial_trajectory_partial_q0 = jax.jacfwd(
        integrator.perform_simulation_with_split_initial_conditions, argnums=4
    )
    partial_trajectory_partial_q1 = jax.jacfwd(
        integrator.perform_simulation_with_split_initial_conditions, argnums=5
    )
    trajectory = integrator.perform_simulation_with_split_initial_conditions(
        f_DEL_q2_first, phys_params, h, N, q0, q1
    )
    vectorfield_q0 = partial_trajectory_partial_q0(
        f_DEL_q2_first, phys_params, h, N, q0, q1
    )
    the_system.plot_results(
        trajectory,
        h,
        N,
        filename="harm_osc_q0_sensitivity.png",
        vectorfield=vectorfield_q0,
        vectorfieldname="sensitivity wrt q0 (scaled)",
    )

    vectorfield_q1 = partial_trajectory_partial_q1(
        f_DEL_q2_first, phys_params, h, N, q0, q1
    )
    the_system.plot_results(
        trajectory,
        h,
        N,
        filename="harm_osc_q1_sensitivity.png",
        vectorfield=vectorfield_q1,
        vectorfieldname="sensitivity wrt q1 (scaled)",
    )


if __name__ == "__main__":
    print(f"Simulating {the_system.name}...")
    main()
    print("Showing off some derivatives...")
    show_off_some_derivatives()
    print("Done.")
