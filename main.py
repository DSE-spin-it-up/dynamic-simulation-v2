import csv

from classes import *
from helpers import *

visualise = True
logging = True

def main():
    # Initialise objects
    initial_states = get_initial_states(n=NUM_DRONES, l0=DEFAULT_PARAMS["L0"])
    drones, payload, cables = initialise_objects(initial_states)

    # Logging setup
    if logging:
        pass

    # Simulation loop
    for t in range(0, SIMULATION_TIME, TIME_STEP):
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