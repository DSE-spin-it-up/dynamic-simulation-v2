import numpy as np

from .default_params import DEFAULT_PARAMS


def get_initial_states(
    num_drones: int = DEFAULT_PARAMS["n_drones"],
    R: float = DEFAULT_PARAMS["R"],
    L0: float = DEFAULT_PARAMS["L0"],
    payload_pos: np.ndarray = np.zeros(3),
    z_target: float = DEFAULT_PARAMS["z_target"],
) -> dict:
    """
    Drones are placed on a cone above the payload such that:
      - horizontal radius = R
      - cable length = L0 exactly
      - drone height above payload = sqrt(L0^2 - R^2)

    Raises if R >= L0 (geometry impossible).
    """
    if R >= L0:
        raise ValueError(f"Cable length L0={L0} must be greater than formation radius R={R}. "
                         f"Currently R={R} >= L0={L0}, so no valid vertical offset exists.")

    angles = np.linspace(0, 2 * np.pi, num_drones, endpoint=False)

    # Height offset above payload such that |drone - payload| == L0 exactly
    dz = np.sqrt(L0**2 - R**2)
    drone_z = payload_pos[2] + dz  # absolute altitude

    drone_positions = np.column_stack([
        payload_pos[0] + R * np.cos(angles),  # offset from payload, not world origin
        payload_pos[1] + R * np.sin(angles),
        np.full(num_drones, drone_z),
    ])

    drone_velocities = np.tile([20.0, 0.0, 0.0], (num_drones, 1))
    payload_velocity = np.zeros(3)

    states = {}
    for i in range(num_drones):
        states[i] = {
            "position": drone_positions[i],
            "velocity": drone_velocities[i],
        }

    states[-1] = {
        "position": payload_pos.copy(),
        "velocity": payload_velocity,
    }

    return states