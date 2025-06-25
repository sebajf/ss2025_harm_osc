import jax.numpy as jnp
from jax import jacfwd

def newton_raphson(f, x0, args=(), tol=1e-6, max_iter=100):
    """
    JAX-compatible Newton-Raphson method for root finding.

    Parameters:
    - f: Function for which the root is sought.
    - x0: Initial guess for the root.
    - args: Additional arguments to pass to the function.
    - tol: Tolerance for convergence.
    - max_iter: Maximum number of iterations.

    Returns:
    - x: The root found.
    """
    x = x0
    for _ in range(max_iter):
        fx = f(x, *args)
        dfx = jacfwd(f)(x, *args)  # Compute the derivative using JAX
        dx = -fx / dfx
        x = x + dx

        if jnp.abs(dx) < tol:
            return x

    raise RuntimeError("Newton-Raphson method did not converge")