import numpy as np
from scipy.integrate import solve_ivp

from src.classes.cable import Cable
from src.classes.drone import Drone
from src.classes.payload import Payload
from src.utils.default_params import DEFAULT_PARAMS


def _pack_state(drones, payload):
    parts = []
    for drone in drones:
        parts.append(drone.position)
        parts.append(drone.v)
    parts.append(payload.position)
    parts.append(payload.v)
    return np.concatenate(parts)


def _unpack_state(state, drones, payload):
    for i, drone in enumerate(drones):
        base = i * 6
        drone.position = state[base : base + 3].copy()
        drone.v = state[base + 3 : base + 6].copy()
    base = len(drones) * 6
    payload.position = state[base : base + 3].copy()
    payload.v = state[base + 3 : base + 6].copy()


def _equations_of_motion(t, state, drones, payload, cables, g):
    _unpack_state(state, drones, payload)

    n = len(drones)
    derivs = np.zeros(len(state))
    F_payload = np.zeros(3)
    g_vec = np.array([0.0, 0.0, -g])

    for i, (drone, cable) in enumerate(zip(drones, cables)):
        f_on_payload, f_on_drone = cable.force_vectors()
        F_payload += f_on_payload

        F_thrust = drone.controller.compute_thrust(drone, payload)
        a_drone = (f_on_drone + F_thrust) / drone.mass + g_vec

        base = i * 6
        derivs[base : base + 3] = drone.v
        derivs[base + 3 : base + 6] = a_drone

    a_payload = F_payload / payload.mass + g_vec
    base = n * 6
    derivs[base : base + 3] = payload.v
    derivs[base + 3 : base + 6] = a_payload

    return derivs


def simulate(drones, payload, cables, params):
    """
    Legacy simulation function using RK45 integration. This is currently not used in the main code, but is kept for reference and potential future use.
    Integrate equations of motion with RK45 and return trajectory history.

    Returns
    -------
    dict with keys:
        't'      : 1-D array of output times
        'drones' : list of (N_times x 6) arrays — [x, y, z, vx, vy, vz] per drone
        -1       : (N_times x 6) array — [x, y, z, vx, vy, vz] for payload
    """
    y0 = _pack_state(drones, payload)
    t_start = params["t_start"]
    t_end = params["t_end"]
    t_eval = np.linspace(t_start, t_end, int((t_end - t_start) / params["dt"]) + 1)
    g = params.get("g", 0.0)

    result = solve_ivp(
        fun=lambda t, y: _equations_of_motion(t, y, drones, payload, cables, g),
        t_span=(t_start, t_end),
        y0=y0,
        method="RK45",
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9,
    )

    _unpack_state(result.y[:, -1], drones, payload)

    n = len(drones)
    history = {"t": result.t, "drones": [], -1: result.y[n * 6 : n * 6 + 6, :].T}
    for i in range(n):
        history["drones"].append(result.y[i * 6 : i * 6 + 6, :].T)

    return history

def compute_gravity_force(mass: float) -> np.ndarray:
    """Compute gravitational force vector [0, 0, -mg]."""
    return np.array([0, 0, -mass * 9.81])

def compute_drone_aero_forces(drone: Drone) -> np.ndarray:
    """
    Computes aerodynamic lift and drag forces acting on the drone.
    Ensures correct quadratic velocity scaling and proper vector directions.
    """
    # Extract velocity vector
    v_vec = np.asarray(drone.v)
    v_norm = np.linalg.norm(v_vec)

    # Handle stationary/hover case to avoid division by zero (NaNs)
    if v_norm < 1e-6:
        return np.zeros(3)

    # 1. Compute unit vector of velocity (direction of travel)
    v_unit = v_vec / v_norm

    # 2. Compute Drag Force (acts directly opposite to velocity direction)
    # Formula: D = -0.5 * rho * S * CD0 * ||v||^2 * v_unit
    D_mag = 0.5 * DEFAULT_PARAMS["rho"] * DEFAULT_PARAMS["S"] * DEFAULT_PARAMS["CD0"] * (v_norm**2)
    D = -D_mag * v_unit

    # 3. Compute Lift Force (acts perpendicular to velocity, in the vertical plane)
    # We find a unit vector perpendicular to velocity that points generally upwards
    vertical_dir = np.array([0.0, 0.0, 1.0])
    
    # Project vertical direction onto the plane perpendicular to v_unit
    lift_dir = vertical_dir - np.dot(vertical_dir, v_unit) * v_unit
    lift_dir_norm = np.linalg.norm(lift_dir)

    if lift_dir_norm > 1e-6:
        lift_unit = lift_dir / lift_dir_norm
    else:
        # Fallback if flying perfectly vertical (lift acts forward/backward depending on pitching, 
        # default to straight up or zero if pure vertical climbing)
        lift_unit = np.array([0.0, 0.0, 1.0])

    # Formula: L = 0.5 * rho * S * CLa * ||v||^2 * lift_unit
    L_mag = 0.5 * DEFAULT_PARAMS["rho"] * DEFAULT_PARAMS["S"] * DEFAULT_PARAMS["CLa"] * (v_norm**2)
    L = L_mag * lift_unit

    # Total aerodynamic force vector
    return D + L


def compute_payload_aero_forces(payload: Payload) -> np.ndarray:
    """Placeholder for payload aerodynamic forces. Currently returns zero."""
    return DEFAULT_PARAMS["CD0_payload"] * 0.5 * DEFAULT_PARAMS["rho"] * DEFAULT_PARAMS["S_payload"] * np.array([0, 0, -1]) * np.linalg.norm(payload.v) * payload.v**2

def compute_forces(drones: list[Drone], cables: list[Cable], payload: Payload) -> dict[int, dict[str, np.ndarray]]:
    """
    Calculate all forces acting on drones and payload at the current state.
    Returns a dictionary of force components for each object.
    """
    # Compute forces
    forces_snapshot = {id: {} for id in [drone.id for drone in drones]}
    forces_snapshot[-1] = {} # Payload forces accessed with ID -1
        
    # For each cable, compute forces and apply to payload and drone
    forces_snapshot[-1]["cable_tension"] = np.zeros(3) # Initialize payload cable tension
    for cable in cables:
        # There is only one cable per drone, several for the payload
        force_payload, force_drone = cable.force_vectors()
        forces_snapshot[-1]["cable_tension"] += force_payload
        forces_snapshot[cable.drone.id]["cable_tension"] = force_drone

    # Aero forces for drones and payload
    for drone in drones:
        aero_force = compute_drone_aero_forces(drone)
        forces_snapshot[drone.id]["aero"] = aero_force
    aero_force_payload = compute_payload_aero_forces(payload)
    forces_snapshot[-1]["aero"] = aero_force_payload

    # Gravity forces for drones and payload
    for drone in drones:
        gravity_force = compute_gravity_force(drone.mass)
        forces_snapshot[drone.id]["gravity"] = gravity_force
    gravity_force_payload = compute_gravity_force(payload.mass)
    forces_snapshot[-1]["gravity"] = gravity_force_payload

    return forces_snapshot

def compute_net_forces(forces_dict: dict[int, dict[str, np.ndarray]]) -> dict[int, np.ndarray]:
    """
    Compute net force vector for each object by summing all force components.
    """
    net_forces = {}
    for obj_id, components in forces_dict.items():
        net_force = np.zeros(3)
        for _, force_vector in components.items():
            net_force += force_vector
        net_forces[obj_id] = net_force
    return net_forces

def update_state(object: Drone | Payload, net_force: np.ndarray) -> None:
    """
    Update the state of a drone or payload based on the net force acting on it.
    """
    acceleration = net_force / object.mass
    object.v += acceleration * DEFAULT_PARAMS["dt"]
    object.position += object.v * DEFAULT_PARAMS["dt"]