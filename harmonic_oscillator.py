from sympy import symbols, diff, simplify, lambdify
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

# Define symbols (internal name and display name)
q0, q1, q2, h, m, k = symbols('q0 q1 q2 h m k')

# Discrete Lagrangian
L_d = h * (m / 2 * ((q1 - q0) / h)**2 - k / 2 * ((q1 + q0) / 2)**2)

# Compute derivatives
D1_Ld = simplify(diff(L_d, q0))
D2_Ld = simplify(diff(L_d, q1))
D1_Ld = D1_Ld.subs({q0: q1, q1: q2}, simultaneous=True)
DEL_equations = simplify(D1_Ld + D2_Ld)

f_DEL = lambdify([q0, q1, q2, h, m, k], DEL_equations, "numpy")


# "Prepare the numbers"

# This overwrites the old meaning of m, k, and h
# (but the expressions and functions do not change)
m = 1.0  # Mass
k = 1.0  # Spring constant
h = 0.1  # Time step

N = 100  # Number of steps
trajectory = np.zeros(N + 1)  # from 0 to N

# Set initial conditions
trajectory[0] = 1.0
trajectory[1] = 1.1


# Root finding in a loop

for i in range(1, N):  # i goes from 1 to N-1
    q0 = trajectory[i - 1]
    q1 = trajectory[i]

    def solve_for_q2(q2):
        return f_DEL(q0, q1, q2, h, m, k)

    guess = q1 + h * (q1 - q0)  # guess for q2
    q2 = root(solve_for_q2, x0=guess)
    # (If you want, you can check if the root finding was successful)
    trajectory[i + 1] = q2.x[0]

# Visualization (saved to a file)

plt.figure(figsize=(10, 6))
time_points = h * np.arange(N+1)
plt.plot(time_points, trajectory, 'bo-', label='Oscillator Position')
# Add a line at y=0 for reference
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Discrete Harmonic Oscillator Trajectory')
plt.legend()
plt.tight_layout()
plt.savefig('harmonic_oscillator_trajectory.png')