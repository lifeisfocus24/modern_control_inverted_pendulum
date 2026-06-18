"""
pendulum_ss.py

Linearized cart-pole inverted pendulum state-space model.

State vector:
    x = [theta, theta_dot, cart_position, cart_velocity]^T

Input:
    u = horizontal force applied to cart

Output:
    y = [theta, cart_position]^T
"""

import numpy as np

def get_default_parameters():
    """
    These are the default physical parameters for the cart-pole system
    """

    return {
        "M": 1.0,     # cart mass (kg)
        "m": 0.1,     # pendulum mass (kg)
        "l": 0.5,     # pendulum COM length (m)
        "g": 9.81,    # gravity (m/s^2) 
    }

def get_ss_matrices(M=1.0, m=0.1, l=0.5, g=9.81):
    """
    Return linearized state-space matrices A, B, C, D
    The system is linearized arund the upright equilibrium theta = 0

    State vector:
    x = [theta, theta_dot, cart_position, cart_velocity]^T

    Input:
    u = horizontal force applied to cart

    Output:
    y = [theta, cart_position]^T
    """

    A = np.array([[0.0, 1.0, 0.0, 0.0], 
                  [((M + m) * g) / (M * l), 0.0, 0.0, 0.0], 
                  [0.0, 0.0, 0.0, 1.0], 
                  [-(m * g) / M, 0.0, 0.0, 0.0]])

    B = np.array([[0.0],
                [-1.0 / (M * l)], 
                [0.0], 
                [1.0 / M]])
    
    C = np.array([
        [1.0, 0.0, 0.0, 0.0],  # measure theta
        [0.0, 0.0, 1.0, 0.0]   # measure cart position
    ])

    D = np.array([
        [0.0],
        [0.0]
    ])

    return A, B, C, D


def state_derivative(x,u, A=None, B=None):
    """
    Copmute x_dot = Ax + Bu 

    Parameters:
    x: State vector with shape (4,)
    u: scalar input force
    A: Optional A matrix
    B: optional B matrix

    This code is setup that we can input new A and B values otherwise we use the default values for them
    """

    if A is None or B is None:
       A, B, _, _ = get_ss_matrices()
    
    x = np.asarray(x).reshape(4, 1)
    u = np.array([[u]])

    x_dot = A @ x + B @ u

    return x_dot.flatten()


# run this file directly to print and verify the state-space matrices
# python src/pendulum_ss.py

if __name__ == "__main__":
    A, B, C, D = get_ss_matrices()

    print("A matrix:")
    print(A)

    print("\nB matrix:")
    print(B)

    print("\nC matrix:")
    print(C)

    print("\nD matrix:")
    print(D)

