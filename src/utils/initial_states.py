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

    # Payload at center of drones
    payload_position = np.mean(
        [state["position"] for state in states.values()],
        axis=0,
    )

    payload_velocity = np.mean(
        [state["velocity"] for state in states.values()],
        axis=0,
    )

    states[-1] = {
        "position": payload_position,
        "velocity": payload_velocity,
    }
        
    return states