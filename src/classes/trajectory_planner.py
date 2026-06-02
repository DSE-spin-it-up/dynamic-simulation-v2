import casadi as ca
import numpy as np

from ..utils.default_params import DEFAULT_PARAMS

from .mission_manager import MissionPhase
from .drone import Drone
from .payload import Payload

class TrajectoryPlanner:
    """Build a trajectory optimization problem and let phases add constraints to it."""

    def __init__(self, mission_phase: int = MissionPhase.TAKE_OFF):
        self.mission_phase = mission_phase
        self.payload_target = np.zeros(3)
        self.payload_target_t = 0.0
        self.next_traj_step_t = 0.0

    def set_payload_target(self, target, target_time):
        '''Update payload target for trajectory optimization.'''
        self.payload_target = np.asarray(target, dtype=float)
        self.payload_target_t = float(target_time)

    def calculate_traj_step(self, t, drones: list[Drone], payload: Payload, mission_phase: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
        # ── Update mission phase ──────────────────────────────────────────────────────
        self.mission_phase = mission_phase

        # ── Build generic optimizer ───────────────────────────────────────────────────
        opti, (pos, vel, F) = self.build_generic_optimizer(drones)

        # ── Set specific contraints per phase ─────────────────────────────────────────


        # ── Solve ─────────────────────────────────────────────────────────────────────

        opti.solver('ipopt', {}, {'max_iter': DEFAULT_PARAMS.get("Opti_max_iter", 500), 'print_level': 3})
        sol = opti.solve()

        # ── Extract solution ──────────────────────────────────────────────────────────

        N = DEFAULT_PARAMS.get("n_drones", len(drones))
        pos_sol = [sol.value(pos[i]) for i in range(N)]   # list of (3, N)
        F_sol   = [sol.value(F[i])   for i in range(N)]

        return pos_sol, F_sol

    def build_generic_optimizer(self, drones) -> tuple[ca.Opti, tuple[list, list, list]]:
        '''Build a generic casadi optimizer with basic variables, objective, and constraints.'''
        opti = ca.Opti()

        # ────────────────── Create optimization variables ─────────────────────────────────────────
        N = DEFAULT_PARAMS.get("n_drones", len(drones))
        opti_N = DEFAULT_PARAMS.get("opti_timepstep_N", 20)
        # att = [opti.variable(3, opti_N) for _ in range(N)]
        pos = [opti.variable(3, opti_N) for _ in range(N)]
        vel = [opti.variable(3, opti_N) for _ in range(N)]
        F   = [opti.variable(3, opti_N) for _ in range(N)]

        # ────────────────── Build generic objective and constraints ───────────────────────────────────────
        self.add_generic_constraints(opti, pos, vel, F, drones)
        
        # ────────────── Build objective ───────────────────────────────────────────────────────────────
        self.add_payload_tracking_objective(opti, pos)

        return opti, (pos, vel, F)

    def add_generic_constraints(self, opti: ca.Opti, pos, vel, F, drones) -> None:
        '''Add generic constraints to the optimizer.'''
        # use drones current states as initial positions
        N = DEFAULT_PARAMS["n_drones"]
        opti_N = DEFAULT_PARAMS["opti_timepstep_N"]
        opti_dt = DEFAULT_PARAMS["opti_dt"]
        m_drone = DEFAULT_PARAMS["m_drone"]
        max_thrust = DEFAULT_PARAMS["max_thrust"]
        min_distance = DEFAULT_PARAMS["min_distance"]

        for i, drone in enumerate(drones):
            if i >= N:
                break
            opti.subject_to(pos[i][:, 0] == drone.position)
            opti.subject_to(vel[i][:, 0] == drone.v)

        # Euler integration constraints
        for i in range(N):
            for k in range(opti_N - 1):
                # Double integrator dynamics (Euler integration)
                opti.subject_to(
                    pos[i][:, k + 1] == pos[i][:, k] + opti_dt * vel[i][:, k]
                )
                opti.subject_to(
                    vel[i][:, k + 1] == vel[i][:, k] + opti_dt * (F[i][:, k] / m_drone)
                )  

        # Drone physical limits
        for i in range(N):
            for k in range(opti_N):
                opti.subject_to(
                    ca.dot(F[i][:, k], F[i][:, k]) <= max_thrust**2
                )
            # Collision avoidance: all pairs, all timesteps
            for j in range(i + 1, N):
                for k in range(opti_N):
                    diff = pos[i][:, k] - pos[j][:, k]
                    opti.subject_to(ca.dot(diff, diff) >= min_distance**2)
        return None
    
    def add_payload_tracking_objective(self, opti: ca.Opti, pos) -> None:
        '''Add objective to track payload target at target time.'''
        # Find the index of the optimization timestep closest to the payload target time
        opti_dt = DEFAULT_PARAMS.get("opti_dt", 0.1)
        opti_N = DEFAULT_PARAMS.get("opti_timepstep_N", 20)
        N = DEFAULT_PARAMS.get("n_drones", len(pos))

        target_k = int(self.payload_target_t / opti_dt)
        if target_k >= opti_N:
            target_k = opti_N - 1

        # Average drone position at target time
        avg_pos_at_target = sum(pos[i][:, target_k] for i in range(N)) / N

        # Objective: minimize distance from average position to payload target
        error = avg_pos_at_target - ca.DM(self.payload_target)
        opti.minimize(ca.dot(error, error))

    def custom_constraint_placeholder(self, opti: ca.Opti) -> None:
        # Replace this with actual constraints for specific mission phases as needed
        return None

    def constraint_placeholder(self, opti: ca.Opti) -> None:
        # Replace this with actual constraints for specific mission phases as needed
        return None
        

    # def _shared_objective_builders(self):
    #     return (self._objective_track_payload_target, self._objective_control_effort)

    # def _shared_constraint_builders(self):
    #     return (self._constraint_force_bounds, self._constraint_collision_placeholder)

    # def _shared_initial_guess_builders(self):
    #     return (self._initial_guess_zero,)

    # def _objective_track_payload_target(self, variables: OptimizationVariables, request: TrajectoryRequest):
    #     payload_position = ca.DM(request.payload.position)
    #     error = payload_position - ca.DM(self.payload_target)
    #     return ca.dot(error, error)

    # def _objective_control_effort(self, variables: OptimizationVariables, request: TrajectoryRequest):
    #     return sum(ca.dot(force, force) for force in variables.forces)

    # def _constraint_force_bounds(self, variables: OptimizationVariables, request: TrajectoryRequest) -> None:
    #     for force in variables.forces:
    #         variables.opti.subject_to(ca.dot(force, force) <= 1.0)

    # def _initial_guess_zero(self, variables: OptimizationVariables, request: TrajectoryRequest) -> None:
    #     for position in variables.positions:
    #         variables.opti.set_initial(position, np.zeros((3, 1)))
    #     for velocity in variables.velocities:
    #         variables.opti.set_initial(velocity, np.zeros((3, 1)))
    #     for force in variables.forces:
    #         variables.opti.set_initial(force, np.zeros((3, 1)))