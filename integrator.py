import jax.numpy as jnp
from jax import lax, jit
from customNewton import newton_raphson

def perform_simulation(f_DEL_q2_first, phys_params, h, N, init_cond):
    trajectory = jnp.zeros(N + 1)  # from 0 to N
    trajectory = trajectory.at[0:2].set(init_cond)  # Set initial conditions

    # We'll use jax.lax.scan, since the number of iterations is static
    def step_fn(carry, _):
        q0, q1 = carry
        guess = 2 * q1 - q0  # Initial guess for q2
        q2 = newton_raphson(f_DEL_q2_first, x0=guess, args=(q0, q1, h, phys_params))
        return (q1, q2), q2  # We'll need (q1, q2) for the next step (it's the "carry")

    # Use lax.scan to iterate over the steps
    init_carry = (trajectory[0], trajectory[1])
    _, q2_values = lax.scan(step_fn, init_carry, jnp.arange(1, N))
    # Now q_values contains *all* the computed q2 values: q2, q3, ..., qN

    # Store those computed values (the initial conditions were already set)
    trajectory = trajectory.at[2:].set(q2_values)
    return trajectory

# Apply JIT with static_argnames
perform_simulation = jit(perform_simulation, static_argnames=["f_DEL_q2_first", "N"])

def perform_simulation_with_split_initial_conditions(f_DEL_q2_first, phys_params, h, N, q0, q1):
    # This function is a wrapper to handle the case where initial conditions are given separately
    # as q0 and q1, rather than as a single array.
    # It will be used to take derivatives with respect to q0 and q1 in the jax version.
    init_cond = jnp.array([q0, q1])
    return perform_simulation(f_DEL_q2_first, phys_params, h, N, init_cond)
