import numpy as np

from ..classes.drone import Drone
from ..classes.payload import Payload
from ..classes.cable import Cable
from ..classes.trajectory_planner import TrajectoryPlanner
from ..classes.mission_manager import MissionManager
from .default_params import DEFAULT_PARAMS

def initialise_objects(initial_states: dict[int, dict[str, np.ndarray]]) -> tuple[list[Drone], Payload, list[Cable], TrajectoryPlanner, MissionManager]:
    '''
    Initialise Drone, Payload, Cable, and TrajectoryPlanner objects based on the provided initial states.
    '''

    # Retrieve payload state using reserved ID -1
    payload_state = initial_states[-1]
    payload = Payload(
        id=-1,  # PAYLOAD RESERVED ID: Always -1
        mass=DEFAULT_PARAMS["m_payload"],
        initial_position=payload_state["position"],
        initial_velocity=payload_state["velocity"],
    )

    drones = []
    cables = []
    for id, state in initial_states.items():
        if id == -1:  # Skip payload
            continue
        drone = Drone(id=id, mass=DEFAULT_PARAMS["m_drone"], initial_position=state["position"], initial_velocity=state["velocity"])
        drones.append(drone)

        cable = Cable(id=id, length=DEFAULT_PARAMS["L0"], stiffness=DEFAULT_PARAMS["k_cable"], damping=DEFAULT_PARAMS["d_cable"], payload=payload, drone=drone)
        cables.append(cable)

    trajectory_planner = TrajectoryPlanner()

    mission_planner = MissionManager()

    return drones, payload, cables, trajectory_planner, mission_planner