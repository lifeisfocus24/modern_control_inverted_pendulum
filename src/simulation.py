"""
simulation.py

Simulation for the Linearized cart-pole inverted pendulum model.
This file will contain the time loop for our simulation, executes the state-space and observer math, computes the distrubance
The outputs of this file will updat the plant, update observer, store results.

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
from lqr_observer import design_lqr, design_observer, control_input, observer_dynamics

def main():
    #load the matrices and design gains
    A, B, C, D = get_ss_matrices()

    K, _, _ = design_lqr()

    L, _, _ = design_observer()

    # Simulation parameters

    dt = 0.001          # Integration timestep (s)
    t_start = 0.0
    t_final = 10.0

    num_steps = int(round((t_final - t_start) / dt)) + 1
    time = np.linspace(t_start, t_final, num_steps)

    #initial conditions for inverted pole cart pendulum system
    x = np.array([0.05, 0, 0, 0])
    x_hat = np.zeros(4)

    #initialize storage arrays
    x_history = np.zeros((num_steps, 4))
    x_hat_history = np.zeros((num_steps, 4))
    y_history = np.zeros((num_steps, 4))
    u_history = np.zeros((num_steps, 4))
    error_history = np.zeros((num_steps, 4))
    innovation_history = np.zeros((num_steps, 4))

    #time loop
    for i in range(num_steps):

        #calculate the control force using current estimate
        u = float(control_input(x_hat, K))
        
        #generate the measurment from the current actual state
        y = C @ x + D @ u 

        #instantaneous derivatives for plant and observer
        x_dot = A @ x + B @ u
        x_hat_dot = observer_dynamics(x_hat, y, u, L)

        #store values
        x_history[i] = x
        x_hat_history[i] = x_hat
        y_history[i] = y
        u_history[i] = float(np.asanyarray(u).squeze())
        error_history[i] = error
        innovation_history[i] = innovation 


        x = x + x_dot * dt
        x_hat = x_hat + x_hat * dt



if __name__ == "__main__":
    main()