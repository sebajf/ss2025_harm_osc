import harm_osc_system_jax as the_system
import integrator
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import time


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

    # Let's vectorize wrt initial conditions
    perform_sim_with_batched_initial_conditions = jax.vmap(
        integrator.perform_simulation,
        in_axes=(None, None, None, None, 0),
    )

    exponent = 5
    number_of_initial_conditions = 10**exponent

    batched_q0_q1 = generate_batched_initial_conditions(number_of_initial_conditions)

    # Call the function once to trigger JIT compilation, so that we can time it afterwards without
    # that overhead.
    print("Warming up JIT compilation...")
    _ = perform_sim_with_batched_initial_conditions(
        f_DEL_q2_first, phys_params, h, N, batched_q0_q1
    )

    # Perform the simulation in parallel for all initial conditions
    print("Performing simulation with batched initial conditions and timing it...")
    start_time = time.perf_counter()
    batched_trajectories = perform_sim_with_batched_initial_conditions(
        f_DEL_q2_first, phys_params, h, N, batched_q0_q1
    )
    # jax has asynchronous execution. The following line makes the program wait
    # until all values are ready, so that we can compute the actual execution time.
    jax.block_until_ready(batched_trajectories)
    end_time = time.perf_counter()

    print(
        f"Time taken for 10**{exponent} initial conditions: {end_time - start_time:.3f} seconds"
    )

    # Plot the results, but with a maximum on the number of trajectories for practical reasons.
    # This is to avoid making matplotlib too slow
    max_trajectories_to_plot = min(1000, batched_trajectories.shape[0])

    plt.figure(figsize=(10, 6))
    time_points = h * jnp.arange(N + 1)
    for i in range(max_trajectories_to_plot):
        plt.plot(time_points, batched_trajectories[i], "b-", alpha=0.1)
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.title(
        f"Harm. Osc. Trajectories: 10**{exponent} Batched Initial Conditions. Time taken: {end_time - start_time:.3f} seconds."
    )
    plt.tight_layout()
    plt.savefig(f"harm_osc_batched_trajectories{exponent}.png")


def generate_batched_initial_conditions(number_of_initial_conditions=10):
    q0batch = jax.random.uniform(
        jax.random.PRNGKey(number_of_initial_conditions),
        shape=(number_of_initial_conditions, 1),
        minval=-1.2,
        maxval=1.2,
    )
    differencebatch = jax.random.uniform(
        jax.random.PRNGKey(number_of_initial_conditions + 1),
        shape=(number_of_initial_conditions, 1),
        minval=-0.2,
        maxval=0.2,
    )
    # Stack the random values to create the initial conditions
    batched_q0_q1 = jnp.hstack((q0batch, q0batch + differencebatch))
    return batched_q0_q1


if __name__ == "__main__":
    print("Running some simulations in parallel with JAX...")
    main()
    print("Done.")
