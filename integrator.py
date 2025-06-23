import numpy as np
from scipy.optimize import root


def perform_simulation(f_DEL_q2_first, phys_params, h, N, init_cond):
    trajectory = np.zeros(N + 1)  # from 0 to N
    # Set initial conditions
    trajectory[0:2] = init_cond  # Set first two values

    for i in range(1, N):  # i goes from 1 to N-1
        q0 = trajectory[i - 1]
        q1 = trajectory[i]

        guess = q1 + h * (q1 - q0)  # guess for q2
        q2 = root(f_DEL_q2_first, x0=guess, args=(q0, q1, h, phys_params))
        # (If you want, you can check if the root finding was successful)
        trajectory[i + 1] = q2.x[0]

    return trajectory
