import jax.numpy as jnp
from jax import jacfwd, lax


def newton_raphson(f, x0, args=(), tol=1e-6, max_iter=100):
    # JAX-compatible Newton-Raphson method for root finding, modified for vmap compatibility.
    #
    # The previous implementation was not compatible with JAX's vmap, as it used
    # "regular" for loops and conditionals. For compatibility, we change to lax.while_loop.
    # This uses a "condition function" and a "body function", which we define below.

    # The body function does the actual work in the loop.
    # It takes the current state (x, dx) and returns the next state.
    # Here x is the current approximation to the root and dx is the change in x.
    def body_fun(state):
        x, _ = state
        fx = f(x, *args)
        dfx = jacfwd(f)(x, *args)
        dx = -fx / dfx
        x = x + dx
        return x, dx

    # The condition function checks if we should continue iterating.
    def cond_fun(state):
        _, dx = state
        return jnp.any(jnp.abs(dx) >= tol)

    x = x0
    dx = (
        jnp.inf
    )  # Initialize dx to infinity, so it's definitely larger than tol at the start.
    state = (x, dx)

    # Start! And get the final state.
    state = lax.while_loop(cond_fun, body_fun, state)
    x, _ = state

    return x
