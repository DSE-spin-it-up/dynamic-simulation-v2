import numpy as np

# ===========================================================================
#  DEFAULT SIMULATION PARAMETERS
# ===========================================================================
DEFAULT_PARAMS: dict = {
    "trajectories_path": "mission.csv", # path to drone trajectories CSV file
    "n_drones": 3,      # number of drones in the system

    # Physical constants
    "g": 9.81,          # gravitational acceleration, m/s²
    "rho_air": 1.225,    # air density at sea level, kg/m³

    # Drone properties
    "m_drone": 6.8,         # drone mass, kg  (each)
    "drone_cl": 0.8,                # lift coefficient
    "drone_cd": 0.1,                # drag coefficient
    "drone_area": 1.4,              # wing area for aero forces
    # Payload properties
    "m_payload": 50.0,         # payload mass, kg
    "payload_cd": 0.47,           # payload drag coefficient
    "payload_area": 0.5,          # payload cross-sectional area for aero forces

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

    # Wind
    "wind": False,
    "wind_type": "turbulence",  # "constant", "gust", "oscillating" or "turbulence"
    "wind_constant": np.array([5.0, 0.0, 0.0]),  # m/s (for constant wind)
    "wing_gust_speed": 10.0,   # m/s (for gusts)
    "gust_duration": 2.0,     # seconds (for gusts)
    "gust_probability": 0.01, # probability of gust starting at each time step
    "wind_oscillation_amplitude": 15.0,  # m/s (for oscillating wind)
    "wind_oscillation_frequency": 0.5,   # Hz (for oscillating wind)
    "wind_tau":   5.0,    # correlation time — larger = smoother/slower gusts
    "wind_sigma": 7.0,    # std dev of wind speed [m/s] values go from 0.5ish for light wind to  6-7 for stong wind (Beaufort 5-6)
}
