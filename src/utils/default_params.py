import numpy as np

# ===========================================================================
#  DEFAULT SIMULATION PARAMETERS
# ===========================================================================
DEFAULT_PARAMS: dict = {
    "trajectories_path": "mission.csv", # path to drone trajectories CSV file
    "n_drones": 3,      # number of drones in the system

    # Physical constants
    "g": 9.81,          # gravitational acceleration, m/s²
    "rho_air": 1.225,       # air density at sea level, kg/m³

    # Drone properties
    "m_drone": 6.8,         # drone mass, kg  (each)
    "drone_inertia": np.diag([2.32e-3, 2.32e-3, 4.00e-3]), # drone inertia matrix, kg·m² (assumed diagonal)

    # Payload properties
    "m_payload": 50.0,         # payload mass, kg
    "payload_cd": 1.0,         # payload drag coefficient (sphere)
    "payload_a_ref": 0.5,      # payload reference area, m²

    # Cable properties
    "L0": 18,          # rest length, m 
    "k_cable": 100000.0,   # spring stiffness, N/m 
    "d_cable":  100.0,   # damping, N·s/m

    # Controller - Orbit radial PD
    "k_prop_attitude"  : 8.0,               # attitude proportional gain
    "k_deriv_attitude"     : 1.5,               # attitude derivative gain
    "prop_error" : 100.0,   # proportional gain for radial error
    "deriv_error" : 50.0,    # derivative gain for radial error
    "int_error" : 50.0,      # integral gain for radial error

    # Integration
    "simulation_dt":      0.01,    # output time-step (not the ODE step)

    # Drone limits
    "max_thrust"       : 700,   # [N] per drone, 2g ceiling

    # Aerodynamics
    "drone_cl_alpha" : 2 * np.pi,    # lift curve slope [1/rad], thin airfoil theory
    "drone_alpha0"   : 0.0,          # zero-lift angle of attack [rad]
    "drone_cl_max"   : 2.0,          # stall clamp
    "drone_cd"       : 0.05,         # drag coefficient (fixed-wing, much lower than sphere)
    "drone_a_ref"    : 0.04,         # reference area [m²]
    "use_aero_lift"  : True,        # toggle lift on/off
}
