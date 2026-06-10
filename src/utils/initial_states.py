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

    # Connector at center of drones
    connector_position = np.mean(
        [state["position"] for state in states.values()],
        axis=0,
    )

    connector_velocity = np.mean(
        [state["velocity"] for state in states.values()],
        axis=0,
    )

    states[-1] = {
        "position": connector_position,
        "velocity": connector_velocity,
    }
    # Payload a below connector
    states[-2] = {
        "position": connector_position - np.array([0, 0, DEFAULT_PARAMS["connector_length"]]),
        "velocity": connector_velocity,
    }
        
    return states