import numpy as np

from src.utils.initialise_objects import initialise_objects
from src.utils.default_params import DEFAULT_PARAMS
from src.utils.initial_states import get_initial_states
from src.simulation.physics import simulate
from src.visualizations.plot import animate_trajectories_3d, plot_radius_vs_time, plot_gain_response


def main():
    initial_states = get_initial_states(
        num_drones=DEFAULT_PARAMS["n_drones"],
        R=DEFAULT_PARAMS["R"],
        L0=DEFAULT_PARAMS["L0"],
        payload_pos=np.array([0.0, 0.0, 0.0]),
        z_target=DEFAULT_PARAMS["z_target"],
    )
    drones, payload, cables = initialise_objects(initial_states)

    history = simulate(drones, payload, cables, DEFAULT_PARAMS)
    plot_gain_response(DEFAULT_PARAMS)
    plot_radius_vs_time(history, R=DEFAULT_PARAMS["R"], L0=DEFAULT_PARAMS["L0"])
    animate_trajectories_3d(history, params=DEFAULT_PARAMS)


if __name__ == "__main__":
    main()