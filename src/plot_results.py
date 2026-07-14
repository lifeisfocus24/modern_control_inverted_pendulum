"""
plot_results.py

This file takes the simulation data and uses these stored results to generate figures of the states, estimation error, control input, and innovation.

State vector for the Linearized cart-pole inverted pendulum model:
    x = [theta, theta_dot, cart_position, cart_velocity]^T

Input:
    u = horizontal force applied to cart

Output:
    y = [theta, cart_position]^T
"""
import numpy as np
import matplotlib.pyplot as plt

from simulation import run_simulation

#State labels
STATE_LABELS = [
    r"$\theta$ (Angle, rad)", 
    r"$\dot{\theta}$ (Angular Velocity, rad/s)", 
    "Cart Position (m)", 
    "Cart Velocity (m/s)"
]

INNOVATION_LABELS = [
    r"Angle Innovation (rad)",
    "Cart Position Innovation (m)",
]


def plot_states(time, x_history, x_hat_history):
    """Plot true and estimated state trajectories."""
    
    # Determine the number of state variables
    num_states = x_history.shape[1]

    fig, ax = plt.subplots(num_states, 1, figsize=(10,3 * num_states), sharex=True, squeeze=False)
    
    ax = ax.flatten()
    
    for i in range(num_states):
        ax[i].plot(time, x_history[:, i], label="True", color="black", linewidth=1.5)
        ax[i].plot(time, x_hat_history[:, i], label="Estimated", color="red", linewidth=1.5)

        label_text = STATE_LABELS[i] if i < len(STATE_LABELS) else f"State Component {i+1}"
        ax[i].set_ylabel(label_text)
        ax[i].grid(True, alpha=0.3)
        ax[i].legend(loc="upper right")

    ax[-1].set_xlabel("Time (s)")

    # Duplicate the x axis to the very top subplot
    ax_top = ax[0].secondary_xaxis("top")
    ax_top.set_xlabel('Time (s)')

    fig.suptitle("True and Estimated State Trajectories", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.show()

def plot_estimation_error(time, error_history):
    """Plot the estimation error."""

    # Determine the number of state variables
    num_states = error_history.shape[1]

    fig, ax = plt.subplots(num_states, 1, figsize=(10,3 * num_states), sharex=True, squeeze=False)
    
    ax = ax.flatten()
    
    for i in range(num_states):
        ax[i].plot(time, error_history[:, i], label="Estimation Error", color="blue", linewidth=1.5)

        #include a zero reference line for the estimation error plots
        ax[i].axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7,)

        label_text = f"Error: {STATE_LABELS[i]}" if i < len(STATE_LABELS) else f"State Error Component {i+1}"
        ax[i].set_ylabel(label_text)
        ax[i].grid(True, alpha=0.3)
        ax[i].legend(loc="upper right")

    ax[-1].set_xlabel("Time (s)")

    # Duplicate the x axis to the very top subplot
    ax_top = ax[0].secondary_xaxis("top")
    ax_top.set_xlabel('Time (s)')

    fig.suptitle("State Estimation Errors", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.show()

def plot_control_input(time, u_history):
    """Plot the control input to the system."""

    # Create a single plot because the system has one control input
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, u_history, label=r"Control Force $u(t)$", color="green", linewidth=1.5)

    #include a zero reference line for the control input plot
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7,)

    ax.set_ylabel("Force (N)")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    fig.suptitle("Control Force Applied to Cart", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.show()


def plot_innovation(time, innovation_history):
    """Plot the observer innovation signals."""

    # Determine the number of measured outputs
    num_outputs = innovation_history.shape[1]

    fig, ax = plt.subplots(num_outputs, 1, figsize=(10,3 * num_outputs), sharex=True, squeeze=False)
    
    ax = ax.flatten()
    
    for i in range(num_outputs):
        ax[i].plot(time, innovation_history[:, i], label="Innovation", color="purple", linewidth=1.5)

        #include a zero reference line for the Innovation signals
        ax[i].axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7,)

        label_text = INNOVATION_LABELS[i] if i < len(INNOVATION_LABELS) else f"Innovation Component {i+1}"
        ax[i].set_ylabel(label_text)
        ax[i].grid(True, alpha=0.3)
        ax[i].legend(loc="upper right")

    ax[-1].set_xlabel("Time (s)")

    # Duplicate the x axis to the very top subplot
    ax_top = ax[0].secondary_xaxis("top")
    ax_top.set_xlabel('Time (s)')

    fig.suptitle("Observer Innovation Signals", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results = run_simulation()

    plot_states(time=results["time"], x_history=results["states"], x_hat_history=results["estimated_states"])
    plot_estimation_error(time=results["time"], error_history=results["estimation_error"])
    plot_control_input(time=results["time"], u_history=results["control_input"])
    plot_innovation(time=results["time"], innovation_history=results["innovation"])