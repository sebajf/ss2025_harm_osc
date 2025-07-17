import harm_osc_system_jax as the_system  # Switch sympy and jax versions here
import integrator
import pytest


# You can "simulate" replacing the sympy version with the jax version by
# changing the import statement above.
# If the tests were passing with the old sympy version, they should also
# pass with the jax version.


@pytest.mark.parametrize(
    "phys_params",
    [
        {"m": 1.0, "k": 1.0},
        {"m": 2.0, "k": 1.0},
    ],
)
@pytest.mark.parametrize(
    "h, N, init_cond",
    [
        (0.1, 100, [0.0, 0.0]),
        (0.1, 20, [1.0, 1.1]),
        (0.01, 100, [1.0, 1.1]),
    ],
)
def test_harmonic_oscillator(phys_params, h, N, init_cond, ndarrays_regression):
    trajectory = integrator.perform_simulation(
        the_system.f_DEL_q2_first, phys_params, h, N, init_cond
    )
    ndarrays_regression.check({"values": trajectory})
