import numpy as np

# ===========================================================================
#  DEFAULT SIMULATION PARAMETERS
# ===========================================================================
DEFAULT_PARAMS: dict = {
    "trajectories_path": "mission.csv", # path to drone trajectories CSV file
    "n_drones": 3,      # number of drones in the system

    # Physical constants
    "g": 9.81,          # gravitational acceleration, m/s²

    # Drone properties
    "m_drone": 6.8,         # drone mass, kg  (each)

    # Payload properties
    "m_payload": 50.0,         # payload mass, kg

    # Cable properties
    "L0": 18,          # rest length, m 
    "k_cable": 100000.0,   # spring stiffness, N/m 
    "d_cable":  100.0,   # damping, N·s/m

    # Controller - Orbit radial PD
    "prop_error" : 100.0,   # proportional gain for radial error
    "deriv_error" : 50.0,    # derivative gain for radial error
    "int_error" : 50.0,      # integral gain for radial error

    # Integration
    "simulation_dt":      0.01,    # output time-step (not the ODE step)

    # Mission parameters
    "h_box":                    0.5,     # payload box height above ground, m
    "target_payload_altitude":  30.0,    # payload altitude target at end of SPINNING_UP, m
    "cruise_range":             1000.0,  # ground distance to cover during CRUISE, m
}
