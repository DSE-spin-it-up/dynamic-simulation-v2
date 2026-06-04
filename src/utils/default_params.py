# ===========================================================================
#  DEFAULT SIMULATION PARAMETERS
# ===========================================================================
DEFAULT_PARAMS: dict = {
    "n_drones": 3,      # number of drones in the system

    # Physical constants
    "g": 9.81,          # gravitational acceleration, m/s²

    # Drone properties
    "m_drone": 10.0,         # drone mass, kg  (each)

    # Payload properties
    "m_payload": 50.0,         # payload mass, kg

    # Cable properties
    "L0": 3.5,          # rest length, m 
    "k_cable": 100000.0,   # spring stiffness, N/m 
    "d_cable":  100.0,   # damping, N·s/m

    # SIU Controller - Orbit geometry
    "R":             3.0,   # nominal orbit radius, m
    "omega_target":  1.0,   # rad/s
    "k_omega":       3.0,   # angular velocity P gain for steady-state orbit tracking

    # Controller - Orbit radial PD
    "kp_alt":   3.0,    # proportional gain (radial)
    "kd_alt":   7.0,    # derivative gain (radial)

    # Controller - Altitude (z)
    "z_target":  3.0,   # target drone height ABOVE payload, m
    "kp_z":     10.0,   # altitude PD proportional gain
    "kd_z":      5.0,   # altitude PD derivative gain

    # Integration
    "t_start": 0.0,
    "t_end":   30.0,
    "dt":      0.001,    # output time-step (not the ODE step)

    # Mission parameters
    "h_box":                    0.5,     # payload box height above ground, m
    "target_payload_altitude":  30.0,    # payload altitude target at end of SPINNING_UP, m
    "cruise_range":             1000.0,  # ground distance to cover during CRUISE, m

}
