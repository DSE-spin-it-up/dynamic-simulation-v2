import numpy as np

from .default_params import DEFAULT_PARAMS


def get_initial_states(
    num_drones: int = DEFAULT_PARAMS["n_drones"],
    R: float = DEFAULT_PARAMS["R"],
    L0: float = DEFAULT_PARAMS["L0"],
    payload_pos: np.ndarray = np.zeros(3),
) -> dict:
    """
    Returns initial positions and velocities for all drones and the payload.

    Drones are evenly spaced on a horizontal circle (x-z plane) of radius R,
    placed at height h = sqrt(max(L0^2 - R^2, 0)) above the payload.

    Output:
        A dictionary containing initial states for each drone and the payload (velocity and position).
    """
    angles = np.linspace(0, 2 * np.pi, num_drones, endpoint=False)

    # For 2D, we can just place the drones at the same height as the payload
    drone_z = payload_pos[2]  

    # FOR 3D
    #h = np.sqrt(max(L0**2 - R**2, 0.0)) # Distance from drone to payload 
    #drone_z = payload_pos[1] + h # Drone altitude

    drone_positions = np.column_stack([
        R * np.cos(angles),  # x-coordinates drones
        R * np.sin(angles),  # y-coordinates drones
        np.full(num_drones, drone_z),  
    ])

    # We assume velocities to be zero for this initial simulation
    drone_velocities = np.zeros((num_drones, 3))
    payload_velocity = np.zeros(3)

    states = {}

    # Add drones with numeric IDs (0, 1, 2, ...)
    for i in range(num_drones):
        states[i] = {
            "position": drone_positions[i],
            "velocity": drone_velocities[i],
        }

    # Add payload with reserved ID -1
    # CONVENTION: Payload always uses id=-1. This is reserved and should not be used for any drone.
    states[-1] = {
        "position": payload_pos.copy(),
        "velocity": payload_velocity,
    }

    return states

# if __name__ == "__main__":

#     states = get_initial_states()

#     print(states)