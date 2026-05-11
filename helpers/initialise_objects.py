import numpy as np

from classes.Drone import Drone
from classes.Payload import Payload
from classes.Cable import Cable
from helpers.default_params import *

def initialise_objects(initial_states: dict[int, list[np.ndarray]]) -> tuple[list[Drone], Payload, list[Cable]]:

    payload_state = initial_states["payload"]
    payload = Payload(
        mass=DEFAULT_PARAMS["m_payload"],
        initial_position=payload_state["position"],
        initial_velocity=payload_state["velocity"],
    )

    drones = []
    cables = []
    for id, state in initial_states.items():
        if id == "payload":
            continue
        drone = Drone(id=id, mass=DEFAULT_PARAMS["m_drone"], initial_position=state["position"], initial_velocity=state["velocity"])
        drones.append(drone)

        cable = Cable(id=id, length=DEFAULT_PARAMS["L0"], stiffness=DEFAULT_PARAMS["k_cable"], damping=DEFAULT_PARAMS["d_cable"], conection_A=payload, conection_B=drone)
        cables.append(cable)

    return drones, payload, cables