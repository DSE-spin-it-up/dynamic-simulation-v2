# ===========================================================================
#  DEFAULT SIMULATION PARAMETERS
# ===========================================================================
DEFAULT_PARAMS: dict = {
    "n_drones": 5,      # number of drones in the system

    # Physical constants
    "g": 9.81,          # gravitational acceleration, m/s²

    # Drone properties
    "m_drone": 10.0,         # drone mass, kg  (each)

    # Payload properties
    "m_payload": 50.0,         # payload mass, kg

    # Cable prpoerties
    "L0": 1.5,          # rest length, m
    "k_cable":  25.0,         # spring stiffness, N/m
    "d_cable":  20.0,         # damping, N·s/m

    # SIU Controller - Orbit geometry
    "R":             2.0,   # nominal orbit radius, m
    "omega_target":  2.5,   # rad/s
    "k_omega":       5.0,   # angular velocity P gain for steady-state orbit tracking

    # Controller - Altitude
    "z_target": 100.0,  # desired payload altitude, m
    "kp_alt":   3.0,    # proportional gain
    "kd_alt":   7.0,    # derivative gain (high damping suppresses orbit-frequency forcing)

    # Integration
    "t_start": 0.0,
    "t_end":   30.0,
    "dt":      0.01,    # output time-step (not the ODE step)

}
