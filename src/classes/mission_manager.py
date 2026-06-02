from dataclass import dataclass

import numpy as np

"Defines which phase we are and defines trajectory to be followed by payload "

class MissionPhase:
    TAKE_OFF = 0
    SPINNING_GROUND = 1
    SPINNING_UP = 2
    TRANSITION_SPIN_TO_CRUISE = 3
    CRUISE = 4
    TRANSITION_CRUISE_TO_SPIN = 5
    LANDING = 6
    EMERGENCY_LANDING_OPERATOR = 7 
    FAILURE_ONE_DRONE = 8
    OPERATOR_OVERRIDE = 9
    EMERGENCY_LANDING_LOSS_COMMUNICATION = 10


@dataclass(frozen=True)
class MissionCommand:
    phase: MissionPhase

@dataclass(frozen=True)
class MissionInputs:
    t: float
    drones: list
    payload: object
    cables: list

class MissionManager:
    """
    Owns the mission state machine.

    State meanings:
    - TAKE_OFF: lift drones from initial condition.
    - SPINNING_GROUND: drones start spinning and accelerating while payload remains stationary on the ground until they lift the payload.
    - SPINNING_UP: drones stop accelerating and start going up vertically.
    - TRANSITION_SPIN_TO_CRUISE: drones transition from spinning to forward cruise.
    - CRUISE: drones move in a straight line until they reach the target destination.
    - TRANSITION_CRUISE_TO_SPIN: slow payload translation and prepare descent, change heading to spin down heading.
    - LANDING: keep spinning until touchdown.
    - EMERGENCY_LANDING: immediately reduce spin and lower payload to ground. Only activated by operator input, not automatically by the state machine.
    """

    def __init__(self):
        self.phase = MissionPhase.TAKE_OFF

    def update(self, t: float, drones: list, payload, cables: list) -> MissionCommand:
        inputs = MissionInputs(t=t, drones=drones, payload=payload, cables=cables)

        self._update_phase(inputs)

        return MissionCommand(
            phase=self.phase,
        )

    def _update_phase(self, inputs: MissionInputs) -> None:
        if self.phase == MissionPhase.TAKE_OFF and self._takeoff_complete(inputs):
            self.phase = MissionPhase.SPINNING_GROUND

        elif self.phase == MissionPhase.SPINNING_GROUND and self._ready_to_spin_up(inputs):
            self.phase = MissionPhase.SPINNING_UP

        elif self.phase == MissionPhase.SPINNING_UP and self._spinup_complete(inputs):
            self.phase = MissionPhase.TRANSITION_SPIN_TO_CRUISE

        elif self.phase == MissionPhase.TRANSITION_SPIN_TO_CRUISE and self._ready_for_cruise(inputs):
            self.phase = MissionPhase.CRUISE

        elif self.phase == MissionPhase.CRUISE and self._cruise_complete(inputs):
            self.phase = MissionPhase.TRANSITION_CRUISE_TO_SPIN

        elif self.phase == MissionPhase.TRANSITION_CRUISE_TO_SPIN and self._ready_to_land(inputs):
            self.phase = MissionPhase.LANDING

    def _takeoff_complete(self, inputs: MissionInputs) -> bool:
        return inputs.drones[2].z >= h_payload + L0*np.cos(theta) # h_payload = height of the box

    def _ready_to_spin_up(self, inputs: MissionInputs) -> bool:
        return Lift_on_payload >= m_system*g

    def _spinup_complete(self, inputs: MissionInputs) -> bool:
        # Later: compute actual average omega from drone/payload relative motion.
        return h >= target_altitude

    def _ready_for_cruise(self, inputs: MissionInputs) -> bool:
        return #drones heading = reference heading payload

    def _cruise_complete(self, inputs: MissionInputs) -> bool:
        return np.linalg.norm(inputs.payload.position - target) < 0.2

    def _ready_to_land(self, inputs: MissionInputs) -> bool:
        return inputs.t > 20.0