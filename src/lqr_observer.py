"""
lqr_observer.py

This file will:
import A,B,C,D for my inverted pendulum (cart pole) setup
define weighiting matrices Q and R
design the LQR gain K
choose observer poles
design the observer gain L

State vector:
    x = [theta, theta_dot, cart_position, cart_velocity]^T

Input:
    u = horizontal force applied to cart

Output:
    y = [theta, cart_position]^T

"""

import numpy as np
import control as ct

from pendulum_ss import get_ss_matrices

#fetch matrices globally
A, B, C, D = get_ss_matrices()

def design_lqr():
    
    Q = np.diag([1000.0, 100.0, 10.0, 1.0])
    R = np.array([[0.01]])

    K, S, E = ct.lqr(A, B, Q, R)

    return K, S, E


def design_observer():
    """Return observer gain and requested/actual observer poles"""
    observer_poles = np.array([-4.0, -5.0, -6.0, -7.0])

    L = ct.place(A.T, C.T, observer_poles).T
    observer_eigenvalues = np.linalg.eigvals(A - L @ C)

    return L, observer_poles, observer_eigenvalues


def control_input(x_hat, K):
    """Compute observer based state-feedback control imput"""

    return -K @ x_hat

def observer_dynamics(x_hat, y, u, L):
    """Compute x_hat_dot from the model, known input, and measurements"""

    y_hat = C @ x_hat
    innovation = y - y_hat

    x_hat_dot = A @ x_hat + B @ u + L @ innovation

    return x_hat_dot


if __name__ == "__main__":
    K, S, E = design_lqr()
    L, requested_poles, actual_poles = design_observer()

    print("LQR gain K:")
    print(K)

    print("\nRiccati solution S:")
    print(S)

    print("\nClosed-loop eigenvalues E:")
    print(E)

    print("\nObserver gain L:")
    print(L)

    print("\nRequested observer poles:")
    print(requested_poles)

    print("\nActual observer poles:")
    print(actual_poles)