from dataclasses import dataclass
import numpy as np
import casadi as ca

# ===========================================================================
#  DEFAULT SIMULATION PARAMETERS
# ===========================================================================
DEFAULT_PARAMS: dict = {
    "n_drones": 3,      # number of drones in the system

    # Physical constants
    "g": 9.81,          # gravitational acceleration, m/s²

    # Drone properties
    "m_drone": 0.5,         # drone mass, kg  (each)

    # Payload properties
    "m_payload": 2.0,         # payload mass, kg

    # Cable properties
    "L0": 3.5,          # rest length, m 
    "k_cable": 500.0,   # spring stiffness, N/m 
    "d_cable":  40.0,   # damping, N·s/m

    # SIU Controller - Orbit geometry
    "R":             3.0,   # nominal orbit radius, m
    "omega_target":  1.0,   # rad/s
    "k_omega":       3.0,   # angular velocity P gain for steady-state orbit tracking

    # Controller
    "prop_gain": 5.0,   # proportional gain for radial control
    "deriv_gain": 2.0,   # derivative gain for radial control
    "int_gain": 2,   # integral gain for radial control

    "z_target":  3.0,   # target drone height ABOVE payload, m


    # Integration
    "t_start": 0,
    "t_end":   40.0,
    "dt":      0.01,    # output time-step (not the ODE step)

    # Optimiser
    "opti_timepstep_N": 100,   # number of time steps in the trajectory optimization horizon
    "opti_dt": 0.1,    # time step between optimization points (s)
    "Opti_max_iter": 1000,   # maximum iterations for the optimizer

    # Physical limits
    "max_thrust": 1,   # maximum thrust per drone, N
    "min_distance": 1,    # minimum distance between drones, m
    
    # Mission parameters
    "h_box":                    0.5,     # payload box height above ground, m
    "target_payload_altitude":  30.0,    # payload altitude target at end of SPINNING_UP, m
    "cruise_range":             1000.0,  # ground distance to cover during CRUISE, m

    # Aircraft parameters
    "rho": 1.225,    # air density, kg/m^3
    "S": 1.4,        # wing reference area, m^2
    "CL0": 0.8,
    "AR": 6.5,
    "e": 0.85,
    "CLa": 4.6,      # lift-curve slope, 1/rad
    "CD0": 0.02,
    "CD0_payload": 1.07,
    "S_payload": 0.56,   # reference area for payload drag, m^2

}

@dataclass
class SimParams:
    """Simulation / discretization parameters."""
    N_uav:   int   = 3      # number of UAVs (change freely)
    N:       int   = 200   # number of timesteps (total time = (N-1)*dt)
    dt:      float = 0.005   # timestep size [s]
    N_h:     int   = 30     # receding-horizon window length [nodes]
    N_apply: int   = 10     # nodes committed per window [nodes]


@dataclass
class VehicleParams:
    """Vehicle / aero parameters and payload + cable parameters."""
    # Vehicle / aero
    m:           float = DEFAULT_PARAMS["m_drone"]      # mass of each UAV [kg]
    g:           float = DEFAULT_PARAMS["g"]     # gravity [m/s^2]
    rho:         float = DEFAULT_PARAMS["rho"]    # air density [kg/m^3]
    S:           float = DEFAULT_PARAMS["S"]        # wing reference area [m^2]
    CL0:         float = DEFAULT_PARAMS["CL0"]
    AR:          float = DEFAULT_PARAMS["AR"]
    e:           float = DEFAULT_PARAMS["e"]
    CLa:         float = DEFAULT_PARAMS["CLa"]      # lift-curve slope [1/rad]
    CD0:         float = DEFAULT_PARAMS["CD0"]
    CD0_payload: float = DEFAULT_PARAMS["CD0_payload"]
    S_payload:   float = DEFAULT_PARAMS["S_payload"]

    # Payload / cable
    m_L:       float = DEFAULT_PARAMS["m_payload"]    # payload mass [kg]
    cable_len: float = DEFAULT_PARAMS["L0"]    # cable length L_c [m] (nominal)
    cable_tol: float = 0.1     # allowed half-band on the chord length [m]


@dataclass
class StateLimits:
    """State / control limits."""
    V_min:     float = 10.0
    V_max:     float = 30.0
    gam_max:   float = np.deg2rad(45.0)
    T_min:     float = 0.0
    T_max:     float = 140         # maybe change to vtol thrust
    P_max:     float =  6000 # max propulsive power per UAV [W]
    alpha_min: float = np.deg2rad(-15.0)
    alpha_max: float = np.deg2rad(10.0)
    mu_max:    float = np.deg2rad(35.0)
    d_min:     float = 2.0              # min distance between any two UAVs [m]
    V_cruise:  float = 20.0
    Tc_max:    float = 500.0  # max cable tension [N]

@dataclass
class OptiVariables:
    """Optimization variables for the trajectory planner."""
    x:           list   # per-UAV state   (6, N)
    u:           list   # per-UAV control (3, N)
    Tc:          list   # per-UAV cable tension (1, N)
    payload_pos: ca.MX  # payload position (3, N)
    payload_vel: ca.MX  # payload velocity (3, N)