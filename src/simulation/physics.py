import numpy as np
from scipy.integrate import solve_ivp

from src.classes.cable import Cable
from src.classes.drone import Drone
from src.classes.payload import Payload
from src.utils.default_params import DEFAULT_PARAMS

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

def compute_gravity_force(mass: float) -> np.ndarray:
    """Compute gravitational force vector [0, 0, -mg]."""
    return np.array([0, 0, -mass * 9.81])

def compute_drone_aero_forces(drone: Drone) -> np.ndarray:
    """
    Fixed-wing aerodynamic forces on the drone body.

    Drag
    ----
        F_drag = -½ · ρ · Cd · A · |v|² · v_hat

    Lift
    ----
    Dynamic pressure:
            q = ½ · ρ · A · |v|²

    Angle of attack α:
            α = atan2(v · body_x, v · body_z)   [in the pitch plane]

    Lift coefficient:
            Cl = clip(Cl_alpha · (α - α0), -Cl_max, Cl_max)

    Lift direction: perpendicular to velocity, starting straight "up"
            L_dir = Rodrigues(v_hat, φ) · lift_up
    """
    speed = np.linalg.norm(drone.v)

    # ── Drag ──────────────────────────────────────────────────────────
    if speed < 1e-9:
        return np.zeros(3)

    v_hat = drone.v / speed
    q_dyn = 0.5 * DEFAULT_PARAMS["rho_air"] * DEFAULT_PARAMS["drone_a_ref"] * speed**2

    drag = -DEFAULT_PARAMS["drone_cd"] * q_dyn * v_hat

    if not DEFAULT_PARAMS.get("use_aero_lift", False):
        return drag

    # ── Angle of attack ───────────────────────────────────────────────
    # Project velocity onto body axes to get α in the pitch plane
    body_x = drone.rotation.apply([1.0, 0.0, 0.0])   # forward axis
    body_z = drone.body_z                              # up/thrust axis

    v_body_x = np.dot(drone.v, body_x)                # forward component
    v_body_z = np.dot(drone.v, body_z)                # vertical component

    alpha = np.arctan2(v_body_x, v_body_z)            # AoA in pitch plane [rad]

    # ── Lift coefficient (linear + hard stall clamp) ──────────────────
    cl = DEFAULT_PARAMS["drone_cl_alpha"] * (alpha - DEFAULT_PARAMS["drone_alpha0"])
    cl = np.clip(cl, -DEFAULT_PARAMS["drone_cl_max"], DEFAULT_PARAMS["drone_cl_max"])

    # ── Lift direction ────────────────────────────────────────────────
    # Start with "up" perpendicular to velocity: project world-Z onto
    # the plane perpendicular to velocity
    world_z     = np.array([0.0, 0.0, 1.0])
    lift_up     = world_z - np.dot(world_z, v_hat) * v_hat
    lift_up_norm = np.linalg.norm(lift_up)

    if lift_up_norm < 1e-9:
        # Flying straight up/down — lift direction undefined
        return drag

    lift_up = lift_up / lift_up_norm

    # Rotate lift_up around velocity axis by roll angle φ (Rodrigues formula)
    # R(v_hat, φ) · lift_up = lift_up·cos(φ) + (v_hat × lift_up)·sin(φ) + v_hat·(v_hat·lift_up)·(1-cos(φ))
    # Since lift_up ⊥ v_hat, the last term vanishes:
    phi     = drone.euler_angles[0]                    # roll angle [rad]
    lateral = np.cross(v_hat, lift_up)                 # rightward axis in wind frame
    lift_dir = lift_up * np.cos(phi) + lateral * np.sin(phi)

    # ── Lift force ────────────────────────────────────────────────────
    lift = cl * q_dyn * lift_dir

    return drag + lift

def compute_payload_aero_forces(payload: Payload) -> np.ndarray:
    """
    Simple drag on the payload (no attitude, no lift).
    """
    speed = np.linalg.norm(payload.v)
    if speed < 1e-9:
        return np.zeros(3)

    q_dyn = 0.5 * DEFAULT_PARAMS["rho_air"] * speed
    return -DEFAULT_PARAMS["payload_cd"] * DEFAULT_PARAMS["payload_a_ref"] * q_dyn * payload.v


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
    object.v += acceleration * DEFAULT_PARAMS["simulation_dt"]
    object.position += object.v * DEFAULT_PARAMS["simulation_dt"]