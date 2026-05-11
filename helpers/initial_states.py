import numpy as np

from default_params import DEFAULT_PARAMS, NUM_DRONES


def get_initial_states(
    num_drones: int = NUM_DRONES,
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
    drone_z = payload_pos[1]  

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

    # Add drones
    for i in range(num_drones):
        states[i] = {
            "position": drone_positions[i],
            "velocity": drone_velocities[i],
        }

    # Add payload
    states["payload"] = {
        "position": payload_pos.copy(),
        "velocity": payload_velocity,
    }

    return states

# if __name__ == "__main__":

#     states = get_initial_states()

#     print(states)