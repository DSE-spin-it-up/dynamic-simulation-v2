import numpy as np
from random import random

from src.classes.cable import Cable
from src.classes.drone import Drone
from src.classes.payload import Payload
from src.utils.default_params import DEFAULT_PARAMS

def compute_gravity_force(mass: float) -> np.ndarray:
    """Compute gravitational force vector [0, 0, -mg]."""
    return np.array([0, 0, -mass * 9.81])

def compute_drone_aero_forces(drone: Drone, wind_vector: np.ndarray | None) -> np.ndarray:
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
    """Placeholder for drone aerodynamic forces. Currently returns zero."""
    # lift = 0.5 * rho * v^2 * A * Cl
    # drag = 0.5 * rho * v^2 * A * Cd
    relative_air_velocity = wind_vector - drone.v
    speed = np.linalg.norm(relative_air_velocity)

    # ── Drag ──────────────────────────────────────────────────────────
    if speed < 1e-9:
        return np.zeros(3)

    v_hat = relative_air_velocity / speed
    q_dyn = 0.5 * DEFAULT_PARAMS["rho_air"] * DEFAULT_PARAMS["drone_area"] * speed**2

    drag = DEFAULT_PARAMS["drone_cd"] * q_dyn * v_hat # no negative sign because its relative velocity to wind, which is already in the opposite direction of drag

    # ── Lift direction ────────────────────────────────────────────────
    # Assume lift acts perpendicular to velocity in the "up" direction of simulation z axis. Assume constant Cl
    lift_up = np.array([0, 0, 1])
    lift = DEFAULT_PARAMS["drone_cl"] * q_dyn * lift_up  # Placeholder: Cl is constant

    return drag + lift

def compute_payload_aero_forces(payload: Payload, wind_vector: np.ndarray | None) -> np.ndarray:
    """Payload only creates drag"""

    if wind_vector is None:
        return np.zeros(3)

    relative_air_velocity = wind_vector - payload.v
    speed = np.linalg.norm(relative_air_velocity)

    if speed < 1e-9:
        return np.zeros(3)

    v_hat = relative_air_velocity / speed
    q_dyn = 0.5 * DEFAULT_PARAMS["rho_air"] * DEFAULT_PARAMS["payload_area"] * speed**2

    drag = DEFAULT_PARAMS["payload_cd"] * q_dyn * v_hat

    return drag

def get_wind_vector(t: float, last_gust_t: float | None = None, last_wind_vector: np.ndarray | None = None) -> tuple[np.ndarray, float | None]:
    if not DEFAULT_PARAMS["wind"]:
        return np.zeros(3), last_gust_t

    wind_type = DEFAULT_PARAMS["wind_type"]

    if wind_type == "constant":
        return DEFAULT_PARAMS["wind_constant"], last_gust_t

    elif wind_type == "gust":
        gust_active = (last_gust_t is not None) and ((t - last_gust_t) <= DEFAULT_PARAMS["gust_duration"])

        if gust_active and last_wind_vector is not None:
            return last_wind_vector, last_gust_t          # already scaled, return as-is
        elif not gust_active and random() < DEFAULT_PARAMS["gust_probability"]:
            gust_direction = np.random.randn(3)
            gust_direction /= np.linalg.norm(gust_direction)
            wind_vec = DEFAULT_PARAMS["wing_gust_speed"] * gust_direction
            return wind_vec, t
        else:
            return np.zeros(3), last_gust_t

    elif wind_type == "oscillating":
        wind_speed = DEFAULT_PARAMS["wind_oscillation_amplitude"] * np.sin(
            2 * np.pi * DEFAULT_PARAMS["wind_oscillation_frequency"] * t
        )
        return np.array([0.0, 0.0, wind_speed]), last_gust_t


    elif wind_type == "turbulence":
        # ── Dryden-inspired coloured noise (1st order Gauss-Markov) ──
        # Each axis is an independent Ornstein-Uhlenbeck process:
        #   w[k+1] = w[k] * exp(-dt/tau) + sigma * sqrt(1 - exp(-2dt/tau)) * N(0,1)
        # tau  = correlation time [s]  — controls how "slowly" wind changes
        # sigma = std dev of wind speed [m/s]
        dt    = DEFAULT_PARAMS["simulation_dt"]
        tau   = DEFAULT_PARAMS["wind_tau"]        # e.g. 5.0 s
        sigma = DEFAULT_PARAMS["wind_sigma"]      # e.g. 3.0 m/s

        if last_wind_vector is None:
            last_wind_vector = np.zeros(3)

        decay = np.exp(-dt / tau)
        diffusion = sigma * np.sqrt(1.0 - np.exp(-2.0 * dt / tau))
        wind_vec = decay * last_wind_vector + diffusion * np.random.randn(3)
        return wind_vec, last_gust_t   # last_gust_t unused for turbulence

    else:
        raise ValueError(f"Invalid wind type: {wind_type}")

    

def compute_forces(drones: list[Drone], cables: list[Cable], payload: Payload, last_gust_t: float | None, last_wind_vector: np.ndarray | None, t: float) -> tuple[dict[int, dict[str, np.ndarray]], float | None, np.ndarray | None]:
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
    wind_vector, last_gust_t = get_wind_vector(t, last_gust_t, last_wind_vector)  # Assuming same wind for all objects
    for drone in drones:
        aero_force = compute_drone_aero_forces(drone, wind_vector)
        forces_snapshot[drone.id]["aero"] = aero_force
    aero_force_payload = compute_payload_aero_forces(payload, wind_vector)
    forces_snapshot[-1]["aero"] = aero_force_payload

    # Gravity forces for drones and payload
    for drone in drones:
        gravity_force = compute_gravity_force(drone.mass)
        forces_snapshot[drone.id]["gravity"] = gravity_force
    gravity_force_payload = compute_gravity_force(payload.mass)
    forces_snapshot[-1]["gravity"] = gravity_force_payload

    return forces_snapshot, last_gust_t, wind_vector

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