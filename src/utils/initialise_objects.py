import numpy as np

from ..classes.drone import Drone
from ..classes.payload import Payload
from ..classes.cable import Cable
from .default_params import DEFAULT_PARAMS

def initialise_objects(initial_states: dict[int, dict[str, np.ndarray]]) -> tuple[list[Drone], Payload, Payload, list[Cable]]:
    '''
    Initialise Drone, Payload and Cable objects based on the provided initial states.
    '''

    # Retrieve connector state using reserved ID -1
    connector_state = initial_states[-1]
    connector = Payload(
        id=-1,  # CONNECTOR RESERVED ID: Always -1
        mass=DEFAULT_PARAMS["connector_mass"],
        initial_position=connector_state["position"],
        initial_velocity=connector_state["velocity"],
    )

    # payload state using reserved ID -2
    payload_state = initial_states[-2]
    payload = Payload(
        id=-2,  # PAYLOAD RESERVED ID: Always -2
        mass=DEFAULT_PARAMS["m_payload"],
        initial_position=payload_state["position"],
        initial_velocity=np.zeros(3),  # temporal
    )

    drones = []
    cables = []
    for id, state in initial_states.items():
        if id < 0:  # Skip payload and connector states
            continue
        drone = Drone(id=id, mass=DEFAULT_PARAMS["m_drone"], initial_position=state["position"], initial_velocity=state["velocity"])
        drones.append(drone)

        cable = Cable(id=id, length=DEFAULT_PARAMS["L0"], stiffness=DEFAULT_PARAMS["k_cable"], damping=DEFAULT_PARAMS["d_cable"], payload=connector, drone=drone)
        cables.append(cable)
    cables.append(
        Cable(
            id=999,  # or any unique id
            length=DEFAULT_PARAMS["connector_length"],
            stiffness=DEFAULT_PARAMS["k_cable"],
            damping=DEFAULT_PARAMS["d_cable"],
            payload=payload,
            drone=connector, # type: ignore
        )
    )

    return drones, connector, payload, cables