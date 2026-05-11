import numpy as np

from src.utils.initialise_objects import initialise_objects
from src.utils.default_params import DEFAULT_PARAMS
from src.utils.initial_states import get_initial_states
from src.simulation.physics import simulate
from src.visualizations.plot import animate_trajectories


def main():
    initial_states = get_initial_states(
        num_drones=DEFAULT_PARAMS["n_drones"],
        R=DEFAULT_PARAMS["R"],
        L0=DEFAULT_PARAMS["L0"],
        payload_pos=np.array([0.0, 0.0, 0.0]),
    )
    drones, payload, cables = initialise_objects(initial_states)

    history = simulate(drones, payload, cables, DEFAULT_PARAMS)
    animate_trajectories(history)


if __name__ == "__main__":
    main()