"""
simulation.py

Simulation for the Linearized cart-pole inverted pendulum model.
This file contains the simulation time loop, executes the plant and observer
dynamics, and stores the resulting signals.
The outputs of this file will update the plant, update observer, store results.

State vector:
    x = [theta, theta_dot, cart_position, cart_velocity]^T

Input:
    u = horizontal force applied to cart

Output:
    y = [theta, cart_position]^T
"""

import numpy as np

from pendulum_ss import get_ss_matrices
from lqr_observer import design_lqr, design_observer, control_input, observer_dynamics

def run_simulation():
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
    y_history = np.zeros((num_steps, 2))
    u_history = np.zeros(num_steps)
    error_history = np.zeros((num_steps, 4))
    innovation_history = np.zeros((num_steps, 2))

    #time loop
    for i in range(num_steps):

        #calculate the control force using current estimate
        u = np.asarray(control_input(x_hat, K)).item()

        #generate the measurement from the current actual state
        y = C @ x + D[:, 0] * u 

        #instantaneous derivatives for plant and observer
        x_dot = A @ x + B[:, 0] * u
        x_hat_dot, innovation = observer_dynamics(x_hat, y, u, L)

        # calculate error
        error = x - x_hat

        #store values
        x_history[i] = x
        x_hat_history[i] = x_hat
        y_history[i] = y
        u_history[i] = u
        error_history[i] = error
        innovation_history[i] = innovation 


        #Euler integration to update the States
        if i < num_steps - 1:
            x = x + x_dot * dt
            x_hat = x_hat + x_hat_dot * dt

        


    return {
        "time": time,
        "states": x_history,
        "estimated_states": x_hat_history,
        "outputs": y_history,
        "control_input": u_history,
        "estimation_error": error_history,
        "innovation": innovation_history,
    }

if __name__ == "__main__":
    results = run_simulation()
    #temporary print out to confirm the results are within expectations
    print(np.isfinite(results["states"]).all())
    print(np.isfinite(results["estimated_states"]).all())
    print(results["states"][-1])
    print(results["estimated_states"][-1])
    print(results["estimation_error"][-1])
    print(np.max(np.abs(results["control_input"])))
    