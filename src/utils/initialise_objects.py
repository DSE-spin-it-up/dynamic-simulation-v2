import numpy as np

from ..classes.drone import Drone
from ..classes.payload import Payload
from ..classes.cable import Cable
from .default_params import DEFAULT_PARAMS

def initialise_objects(initial_states: dict[int | str, dict[str, np.ndarray]]) -> tuple[list[Drone], Payload, list[Cable]]:

    # Payload first to do the wiring of the cables easier
    payload = Payload(mass=DEFAULT_PARAMS["m_payload"], initial_position=initial_states["payload"]["position"], initial_velocity=initial_states["payload"]["velocity"])

    # Drones and cables
    drones = []
    cables = []
    for id, state in initial_states.items():
        # Skip the "payload" key entry to avoid typecheck errors
        if not isinstance(id, int):
            continue
        drone_id: int = id
        drone = Drone(id=drone_id, mass=DEFAULT_PARAMS["m_drone"], initial_position=state["position"], initial_velocity=state["velocity"])
        drones.append(drone)

        cable = Cable(id=drone_id, length=DEFAULT_PARAMS["L0"], stiffness=DEFAULT_PARAMS["k_cable"], damping=DEFAULT_PARAMS["d_cable"], conection_A=payload, conection_B=drone)
        cables.append(cable)

    return drones, payload, cables