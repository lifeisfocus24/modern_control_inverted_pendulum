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
    
    Q = np.diag([1000, 100, 10, 1])
    R = np.array([[0.01]])

    K, S, E = ct.lqr(A, B, Q, R)

    return K, S, E

def design_observer():
    observer_poles = np.array([-4, -5, -6, -7])

    L = ct.place(A.T, C.T, observer_poles).T

    return L, observer_poles


def observer_dynamics(x_hat, x, K, L):
    y = C @ x
    y_hat = C @ x_hat

    u = -K @ x_hat

    x_hat_dot = A @ x_hat + B @ u + L @ (y - y_hat)

    return x_hat_dot, u


if __name__ == "__main__":
    K, S, E = design_lqr()
    L, observer_poles = design_observer()

    print("LQR gain K:")
    print(K)

    print("\nRiccati solution S:")
    print(S)

    print("\nClosed-loop eigenvalues E:")
    print(E)

    print("\nObserver gain L:")
    print(L)

    print("\nObserver poles:")
    print(observer_poles)