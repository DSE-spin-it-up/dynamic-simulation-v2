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
    "m_drone": 6.8,         # drone mass, kg  (each)

    # Payload properties
    "m_payload": 60.0,         # payload mass, kg
    "pay_CD0": 1.07,           # payload drag coefficient
    "pay_S": 0.31,             # payload reference area for drag, m^2

    # Cable properties
    "L0": 12.5,          # rest length, m 
    "k_cable": 100.0,   # spring stiffness, N/m 
    "d_cable":  40.0,   # damping, N·s/m

    # SIU Controller - Orbit geometry
    "R":             12,   # nominal orbit radius, m

    # Controller
    "prop_gain": 500.0,   # proportional gain for radial control
    "deriv_gain": 20.0,   # derivative gain for radial control
    "int_gain": 100,   # integral gain for radial control

    "z_target":  0.0,   # target drone height ABOVE payload, m


    # Integration
    "t_start": 0,
    "t_end":   10.0,
    "dt":      0.01,    # output time-step (not the ODE step)

    # Optimiser
    "opti_N_h": 55,   # number of time steps in the trajectory optimization horizon
    "opti_N_apply": 5, # number of time steps to apply before re-optimizing
    "opti_dt": 0.01,    # time step between optimization points (s)
    "Opti_max_iter": 1000,   # maximum iterations for the optimizer

    # Physical limits
    "max_thrust": 180,   # maximum thrust per drone, N
    "min_distance": 6.0,    # minimum distance between drones, m
    
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
    "S_payload": 0.31,   # reference area for payload drag, m^2
    "climb_rate": 1,    # max climb rate for reference trajectory, m/s

    "cable_tol": 0.1,     # allowed half-band on the chord length, m

    # Hardware limits
    "V_min": 14.0,     # minimum speed, m/s
    "V_max": 30.0,     # maximum speed, m/s
    "gam_max": np.deg2rad(45.0),   # maximum flight path angle, rad
    "alpha_min": np.deg2rad(-15.0),   # minimum angle
    "alpha_max": np.deg2rad(10.0),    # maximum angle of attack, rad
    "mu_max": np.deg2rad(35.0),       # maximum sideslip angle, rad
    "d_min": 6.0,    # minimum distance between drones, m
    "V_cruise": 20.0,  # cruise speed, m/s
    "Tc_min": 0.0,    # minimum cable tension, N
    "Tc_max": 750.0,  # maximum cable tension, N
    "T_max": 180.0,  # maximum thrust per drone, N
    "T_min": 0.0,    # minimum thrust, N
    "P_max": 2700.0,  # maximum propulsive power per UAV, W

}

@dataclass
class SimParams:
    """Simulation / discretization parameters."""
    N_uav:   int   = DEFAULT_PARAMS["n_drones"]      # number of UAVs (change freely)
    dt:      float = DEFAULT_PARAMS["opti_dt"]   # timestep size [s]
    N_h:     int   = DEFAULT_PARAMS["opti_N_h"]     # receding-horizon window length [nodes]
    N_apply: int   = DEFAULT_PARAMS["opti_N_apply"]     # nodes committed per window [nodes]


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
    cable_tol: float = DEFAULT_PARAMS["cable_tol"]     # allowed half-band on the chord length [m]


@dataclass
class StateLimits:
    """State / control limits."""
    V_min:     float = DEFAULT_PARAMS["V_min"]
    V_max:     float = DEFAULT_PARAMS["V_max"]
    gam_max:   float = DEFAULT_PARAMS["gam_max"]
    T_min:     float = DEFAULT_PARAMS["T_min"]
    T_max:     float = DEFAULT_PARAMS["T_max"]
    P_max:     float = DEFAULT_PARAMS["P_max"]  # max propulsive power per UAV [W]
    alpha_min: float = DEFAULT_PARAMS["alpha_min"]
    alpha_max: float = DEFAULT_PARAMS["alpha_max"]
    mu_max:    float = DEFAULT_PARAMS["mu_max"]
    d_min:     float = DEFAULT_PARAMS["d_min"]
    V_cruise:  float = DEFAULT_PARAMS["V_cruise"]
    Tc_min:    float = DEFAULT_PARAMS["Tc_min"]
    Tc_max:    float = DEFAULT_PARAMS["Tc_max"]  # max cable tension [N]
    climb_rate: float = DEFAULT_PARAMS["climb_rate"]  # max climb rate for reference trajectory, m/s

@dataclass
class OptiVariables:
    """Optimization variables for the trajectory planner."""
    x:           list   # per-UAV state   (6, N)
    u:           list   # per-UAV control (3, N)
    Tc:          list   # per-UAV cable tension (1, N)
    payload_pos: ca.MX  # payload position (3, N)
    payload_vel: ca.MX  # payload velocity (3, N)

@dataclass
class CostWeights:
    """Constant cost weights applied uniformly over the whole trajectory.

    All keys are scalars except the per-control rate weight ``W_du``, stored as a
    ``(3, 1)`` column ([T, alpha, mu]) so it broadcasts against the CasADi cost
    expressions in ``build_nlp``.
    """
    W_track:  float
    W_du:     np.ndarray   # (3, 1) per-control [T, alpha, mu] rate
    W_dTc:    float        # cable-tension rate
    W_dgamma: float        # flight-path-angle rate
    W_dchi:   float        # heading rate
    W_dV:     float        # airspeed rate
    W_T:      float        # thrust magnitude
    W_form:   float        # formation anchor (pull UAVs toward cruise offsets)

@dataclass
class Config:
    """Everything needed to set up and solve a run"""
    sim:         SimParams
    veh:         VehicleParams
    lim:         StateLimits
    maneuver:    str            # name resolved against utils.maneuvers.MANEUVERS
    common:      dict           # params common to every maneuver (altitude, speed, ...)
    weights:     dict           # maneuver name -> partial weight dict (plus "default")