"""
check_system.py

Check for Controllability and Observability of Linearized cart-pole inverted pendulum state-space model 

"""

import numpy as np
import control as ctrl

import pendulum_ss

def get_matrices():
    A, B, C, _ = pendulum_ss.get_ss_matrices()

    run_check(A, B, C)

def run_check(A, B, C):
    Ctrb = ctrl.ctrb(A, B)
    ctrb_rank = np.linalg.matrix_rank(Ctrb)

    print("Controllability matrix:")
    print(Ctrb)
    print("Controllability Rank:", ctrb_rank)

    if ctrb_rank == A.shape[0]:
        print("System is controllable.")
    else:
        print("System is NOT controllable.")

    print()

    Obsv = ctrl.obsv(A, C)
    obsv_rank = np.linalg.matrix_rank(Obsv)

    print("Observability matrix:")
    print(Obsv)
    print("Observability Rank:", obsv_rank)

    if obsv_rank == A.shape[0]:
        print("System is observable.")
    else:
        print("System is NOT observable.")

if __name__ == "__main__":
    get_matrices()