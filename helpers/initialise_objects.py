import numpy as np

from classes.Drone import Drone
from classes.Payload import Payload
from classes.Cable import Cable
from helpers.default_params import *

def initialise_objects(initial_states: dict[int, list[np.ndarray]]) -> tuple[list[Drone], Payload, list[Cable]]:

    # Payload first to do the wiring of the cables easier
    payload = Payload(mass=DEFAULT_PARAMS["m_payload"], initial_position=initial_states[0][0], initial_velocity=initial_states[0][1])

    # Drones and cables
    drones = []
    cables = []
    for id, state in initial_states.items():
        drone = Drone(id=id, mass=DEFAULT_PARAMS["m_drone"], initial_position=state[0], initial_velocity=state[1])
        drones.append(drone)

        cable = Cable(id=id, length=DEFAULT_PARAMS["L0"], stiffness=DEFAULT_PARAMS["k_cable"], damping=DEFAULT_PARAMS["d_cable"], conection_A=payload, conection_B=drone)
        cables.append(cable)

    return drones, payload, cables