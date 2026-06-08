import numpy as np

from .default_params import DEFAULT_PARAMS


def get_initial_states(trajectories: dict[int, np.ndarray], dt: float) -> dict:
    """
    Returns initial positions and velocities for all drones and the payload.

    trajectories[drone_id] is an array of shape (N, 3):
        [[x0, y0, z0],
         [x1, y1, z1],
         ...]
    """
    states = {}

    for drone_id, traj in trajectories.items():
        initial_position = traj[0]

        if len(traj) > 1:
            initial_velocity = (traj[1] - traj[0]) / dt
        else:
            initial_velocity = np.zeros(3)

        states[drone_id] = {
            "position": initial_position,
            "velocity": initial_velocity,
        }

    drone_positions = np.array([state["position"] for state in states.values()])
    payload_x_y = np.mean(drone_positions[:, :2], axis=0)
    
    drone_z = drone_positions[0, 2] 

    r_squared = np.sum((drone_positions[0, :2] - payload_x_y) ** 2)

    L0 = DEFAULT_PARAMS["L0"]
    if L0**2 < r_squared:
        raise ValueError(f"Cable length L0 ({L0}) is too short! Must be greater than the formation radius ({np.sqrt(r_squared):.2f}m).")
        
    payload_z = drone_z - np.sqrt(L0**2 - r_squared)

    payload_position = np.array([payload_x_y[0], payload_x_y[1], payload_z])

    payload_velocity = np.mean(
        [state["velocity"] for state in states.values()],
        axis=0,
    )
    
    states[-1] = {"position": payload_position, "velocity": payload_velocity}

    return states