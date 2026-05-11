from classes import *
from helpers import *

def main():
    # Calculate initial states
    initial_states = get_initial_states(n=NUM_DRONES, l0=DEFAULT_PARAMS["L0"])
    drones, payload, cables = initialise_objects(initial_states)

if __name__ == "__main__":
    main()