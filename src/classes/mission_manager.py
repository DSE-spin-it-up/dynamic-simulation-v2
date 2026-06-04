from dataclasses import dataclass
from enum import IntEnum
import math
import numpy as np

from .trajectory_planner import TrajectoryPlanner
from ..utils.default_params import DEFAULT_PARAMS


class MissionPhase(IntEnum):
    TAKE_OFF = 0
    SPINNING_GROUND = 1
    SPINNING_UP = 2
    TRANSITION_SPIN_TO_CRUISE = 3
    CRUISE = 4
    TRANSITION_CRUISE_TO_SPIN = 5
    SPIN_DOWN = 6
    TRANSITION_SPIN_TO_HOVER = 7
    GROUND_POWER_OFF = 8
    # Not in the automatic chain for normal missions, these states are triggered externally
    EMERGENCY_LANDING = 9
    OPERATOR_OVERRIDE = 10
    FAILURE_ONE_DRONE = 11


@dataclass
class MissionParams:
    h_box: float = DEFAULT_PARAMS["h_box"]
    m_system: float = DEFAULT_PARAMS["m_payload"] + DEFAULT_PARAMS["n_drones"] * DEFAULT_PARAMS["m_drone"]             
    target_payload_altitude: float = DEFAULT_PARAMS["target_payload_altitude"]
    reference_heading: float = 0.0     # angle between drone trajectory and required payload heading [rad]
    heading_tolerance: float = 0.1     # alignment tolerance for transition to CRUISE [rad]
    cruise_range: float = DEFAULT_PARAMS["cruise_range"]
    # TBD — these two thresholds are placeholders
    motion_threshold_m: float = 0.5    # max payload displacement in window to count as "stopped" [m]
    motion_threshold_s: float = 5.0    # time window for stillness check [s]
    omega_zero_threshold: float = 0.05  # drone angular velocity considered "zero" [rad/s]


@dataclass
class MissionCommand:
    """Output of MissionManager.update()."""
    phase: MissionPhase


@dataclass(frozen=True)
class MissionInputs:
    t: float
    drones: list
    payload: object
    cables: list


class MissionManager:
    """
    Finite state machine that owns the mission phase and drives the trajectory planner.

    State sequence (normal mission):
      TAKE_OFF → SPINNING_GROUND → SPINNING_UP → TRANSITION_SPIN_TO_CRUISE
      → CRUISE → TRANSITION_CRUISE_TO_SPIN → SPIN_DOWN → TRANSITION_SPIN_TO_HOVER
      → GROUND_POWER_OFF
    """

    def __init__(self, params: dict = DEFAULT_PARAMS, mission_params: MissionParams = None):
        self.params = params
        self.phase = MissionPhase.TAKE_OFF
        self.mission_params = mission_params or MissionParams()

        # Derive cable geometry from physics params
        R = params["R"]
        L0 = params["L0"]
        self._theta = math.asin(R / L0)          # cable angle from vertical, rad
        self._z_drone_hover = math.sqrt(L0**2 - R**2)  # vertical cable projection at nominal geometry, m

        # Precompute TAKE_OFF target altitude: h_box + sqrt(L0² - R²)
        mp = self.mission_params
        self._z_drone_takeoff = mp.h_box + self._z_drone_hover

        self.trajectory_planner = TrajectoryPlanner()

        # Tracking state
        self._cruise_start_xy: np.ndarray = None
        # List of (t, payload_xy) pairs for payload motion detection
        self._payload_pos_history: list = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, t: float, drones: list, payload, cables: list) -> MissionCommand:
        inputs = MissionInputs(t=t, drones=drones, payload=payload, cables=cables)

        # Keep a window of payload positions for stillness detection
        self._payload_pos_history.append((t, payload.position[:2].copy()))
        cutoff = t - self.mission_params.motion_threshold_s # Oldest time we still care about
        self._payload_pos_history = [
            (ts, p) for (ts, p) in self._payload_pos_history if ts >= cutoff
        ] # Remove old entries outside the time window

        self._update_phase(inputs)

        return MissionCommand(phase=self.phase)

    # ------------------------------------------------------------------
    # FSM
    # ------------------------------------------------------------------

    def _update_phase(self, inputs: MissionInputs) -> None:
        prev_phase = self.phase

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

        elif self.phase == MissionPhase.TRANSITION_CRUISE_TO_SPIN and self._ready_to_spin_down(inputs):
            self.phase = MissionPhase.SPIN_DOWN

        elif self.phase == MissionPhase.SPIN_DOWN and self._spindown_complete(inputs):
            self.phase = MissionPhase.TRANSITION_SPIN_TO_HOVER

        elif self.phase == MissionPhase.TRANSITION_SPIN_TO_HOVER and self._omega_zero(inputs):
            self.phase = MissionPhase.GROUND_POWER_OFF

        # Record cruise start position on phase entry
        if prev_phase != MissionPhase.CRUISE and self.phase == MissionPhase.CRUISE:
            self._cruise_start_xy = inputs.payload.position[:2].copy()

    # ------------------------------------------------------------------
    # Transition conditions
    # ------------------------------------------------------------------

    def _takeoff_complete(self, inputs: MissionInputs) -> bool:
        """All drones reached H_drones = h_box + sin(theta)*R."""
        return all(drone.z >= self._z_drone_takeoff for drone in inputs.drones)

    def _ready_to_spin_up(self, inputs: MissionInputs) -> bool:
        """Sum of vertical (z) cable forces on payload >= payload weight.
        """
        total_lift = sum(cable.force_vectors()[0][2] for cable in inputs.cables)
        total_weight = self.params["m_payload"] * self.params["g"]
        return total_lift >= total_weight

    def _spinup_complete(self, inputs: MissionInputs) -> bool:
        """Payload reached target cruise altitude."""
        return inputs.payload.z >= self.mission_params.target_payload_altitude

    def _ready_for_cruise(self, inputs: MissionInputs) -> bool:
        """Drone formation heading aligned with required payload heading within tolerance.

        reference_heading is the angle between the drone trajectory direction and the
        payload's required travel heading. A value of 0.0 means the formation is already
        flying in the required direction.
        """
        payload = inputs.payload
        speed_xy = math.hypot(payload.v[0], payload.v[1])
        if speed_xy < 0.01:
            return False
        actual_heading = math.atan2(payload.v[1], payload.v[0])
        angular_error = actual_heading - self.mission_params.reference_heading
        # Wrap to [-pi, pi]
        angular_error = (angular_error + math.pi) % (2 * math.pi) - math.pi
        return abs(angular_error) < self.mission_params.heading_tolerance

    def _cruise_complete(self, inputs: MissionInputs) -> bool:
        """Payload has covered cruise_range along the reference heading direction."""
        if self._cruise_start_xy is None:
            return False
        heading = self.mission_params.reference_heading
        ref_dir = np.array([math.cos(heading), math.sin(heading)])
        displacement = inputs.payload.position[:2] - self._cruise_start_xy
        return float(np.dot(displacement, ref_dir)) >= self.mission_params.cruise_range

    def _ready_to_spin_down(self, inputs: MissionInputs) -> bool:
        """Payload hasn't moved more than motion_threshold_m in the last motion_threshold_s seconds.

        Both thresholds are TBD and marked as placeholders in MissionParams.
        """
        history = self._payload_pos_history
        if len(history) < 2:
            return False
        ref_pos = history[0][1]
        max_displacement = max(np.linalg.norm(p - ref_pos) for (_, p) in history)
        return max_displacement < self.mission_params.motion_threshold_m

    def _spindown_complete(self, inputs: MissionInputs) -> bool:
        """Payload reached ground level (5 cm tolerance)."""
        return inputs.payload.z <= 0.05

    def _omega_zero(self, inputs: MissionInputs) -> bool:
        """All drone angular velocities around the payload are effectively zero."""
        threshold = self.mission_params.omega_zero_threshold
        return all(
            self._drone_omega(drone, inputs.payload) <= threshold
            for drone in inputs.drones
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _drone_omega(self, drone, payload) -> float:
        """Angular velocity of drone around payload in the x-y plane (rad/s)."""
        dx = drone.x - payload.x
        dy = drone.y - payload.y
        r = math.hypot(dx, dy)
        if r < 1e-9:
            return 0.0
        dv = drone.v - payload.v
        return abs((dx * dv[1] - dy * dv[0]) / r**2)
