import numpy as np

from .classes import *
from .helpers import *

visualise = True
logging = True

def main():
    # Initialise objects
    initial_states = get_initial_states(num_drones=DEFAULT_PARAMS["n_drones"], R=DEFAULT_PARAMS["R"], L0=DEFAULT_PARAMS["L0"], payload_pos=np.array([0, 0, 0]))
    drones, payload, cables = initialise_objects(initial_states)

    # Logging setup
    if logging:
        pass

    # Simulation loop
    for t in np.arange(0, int(DEFAULT_PARAMS["t_end"] / DEFAULT_PARAMS["dt"]), DEFAULT_PARAMS["dt"]):
        # Update controllers
        pass

        # Update physics
        pass

        # Log data
        if logging:
            pass

        # Visualisation
        if visualise:
            pass

if __name__ == "__main__":
    main()