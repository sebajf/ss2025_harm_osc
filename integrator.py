import jax.numpy as jnp
from customNewton import newton_raphson


def perform_simulation(f_DEL_q2_first, phys_params, h, N, init_cond):
    trajectory = jnp.zeros(N + 1)  # from 0 to N
    # Set initial conditions
    trajectory = trajectory.at[0:2].set(init_cond)  # Set first two values

    for i in range(1, N):  # i goes from 1 to N-1
        q0 = trajectory[i - 1]
        q1 = trajectory[i]

        guess = 2 * q1 - q0  # guess for q2
        q2 = newton_raphson(f_DEL_q2_first, x0=guess, args=(q0, q1, h, phys_params))
        trajectory = trajectory.at[i + 1].set(q2)

    return trajectory

def perform_simulation_with_split_initial_conditions(f_DEL_q2_first, phys_params, h, N, q0, q1):
    # This function is a wrapper to handle the case where initial conditions are given separately
    # as q0 and q1, rather than as a single array.
    # It will be used to take derivatives with respect to q0 and q1 in the jax version.
    init_cond = jnp.array([q0, q1])
    return perform_simulation(f_DEL_q2_first, phys_params, h, N, init_cond)
