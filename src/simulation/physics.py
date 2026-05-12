import numpy as np
from scipy.integrate import solve_ivp


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