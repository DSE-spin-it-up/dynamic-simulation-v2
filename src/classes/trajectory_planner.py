import casadi as ca
import numpy as np

from ..utils.default_params import DEFAULT_PARAMS

from .drone import Drone
from .payload import Payload

class TrajectoryPlanner:
    """Build a trajectory optimization problem and let phases add constraints to it."""

    def __init__(self, mission_phase: int = 0):
        self.mission_phase = mission_phase
        self.params = DEFAULT_PARAMS
        self.n_drones = self.params["n_drones"]
        self.horizon_steps = self.params["opti_timepstep_N"]
        self.payload_target = np.zeros(3)
        self.payload_target_t = 0.0
        self.next_traj_step_t = 0.0
        self._opti = None
        self._pos = None
        self._vel = None
        self._F = None
        self._X0 = None
        self._last_pos_sol = None
        self._last_vel_sol = None
        self._last_F_sol = None

    def udpate_mission_phase(self, mission_phase: int):
        self.mission_phase = mission_phase

    def set_payload_target(self, target, target_time):
        '''Update payload target for trajectory optimization.'''
        self.payload_target = np.asarray(target, dtype=float)
        self.payload_target_t = float(target_time)

    def calculate_traj_step(self, t, drones: list[Drone], payload: Payload, mission_phase: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
        print(f"Calculating trajectory step at time {t:.2f}")
        # ── Update mission phase ──────────────────────────────────────────────────────
        self.mission_phase = mission_phase

        # ── Build generic optimizer ───────────────────────────────────────────────────
        opti, (pos, vel, F) = self.build_generic_optimizer(drones)

        assert self._X0 is not None

        # Carry the current state as parameters for the first node.
        for i, drone in enumerate(drones):
            opti.set_value(self._X0[i], np.hstack((drone.position, drone.v)))

        # Warm-start the solver with the previous optimal trajectory when available.
        if self._last_pos_sol is not None and self._last_vel_sol is not None and self._last_F_sol is not None:
            shift = 1 if self._last_pos_sol[0].shape[1] > 1 else 0
            for i in range(min(len(drones), len(self._last_pos_sol))):
                pos_guess = np.hstack([
                    self._last_pos_sol[i][:, shift:],
                    np.tile(self._last_pos_sol[i][:, -1:], (1, shift if shift > 0 else 1)),
                ])
                vel_guess = np.hstack([
                    self._last_vel_sol[i][:, shift:],
                    np.tile(self._last_vel_sol[i][:, -1:], (1, shift if shift > 0 else 1)),
                ])
                force_guess = np.hstack([
                    self._last_F_sol[i][:, shift:],
                    np.tile(self._last_F_sol[i][:, -1:], (1, shift if shift > 0 else 1)),
                ])
                opti.set_initial(pos[i], pos_guess[:, :self.horizon_steps])
                opti.set_initial(vel[i], vel_guess[:, :self.horizon_steps])
                opti.set_initial(F[i], force_guess[:, :self.horizon_steps])
        else:
            for i, drone in enumerate(drones):
                opti.set_initial(pos[i], np.tile(drone.position.reshape(3, 1), (1, self.horizon_steps)))
                opti.set_initial(vel[i], np.tile(drone.v.reshape(3, 1), (1, self.horizon_steps)))
                opti.set_initial(F[i], np.zeros((3, self.horizon_steps)))

        # ── Set specific contraints per phase ─────────────────────────────────────────


        # ── Solve ─────────────────────────────────────────────────────────────────────

        opti.solver('ipopt', {}, {'max_iter': DEFAULT_PARAMS.get("Opti_max_iter", 500), 'print_level': 0})
        sol = opti.solve()

        # ── Extract solution ──────────────────────────────────────────────────────────

        N = DEFAULT_PARAMS.get("n_drones", len(drones))
        pos_sol = [sol.value(pos[i]) for i in range(N)]   # list of (3, N)
        F_sol   = [sol.value(F[i])   for i in range(N)]

        self._last_pos_sol = pos_sol
        self._last_vel_sol = [sol.value(vel[i]) for i in range(N)]
        self._last_F_sol = F_sol

        return pos_sol, F_sol

    def build_generic_optimizer(self, drones) -> tuple[ca.Opti, tuple[list, list, list]]:
        '''Build a generic casadi optimizer with basic variables, objective, and constraints.'''
        if self._opti is not None:
            assert self._pos is not None
            assert self._vel is not None
            assert self._F is not None
            return self._opti, (self._pos, self._vel, self._F)

        opti = ca.Opti()

        # ────────────────── Create optimization variables ─────────────────────────────────────────
        N = self.n_drones
        opti_N = self.horizon_steps
        # att = [opti.variable(3, opti_N) for _ in range(N)]
        pos = [opti.variable(3, opti_N) for _ in range(N)]
        vel = [opti.variable(3, opti_N) for _ in range(N)]
        F   = [opti.variable(3, opti_N) for _ in range(N)]
        X0 = [opti.parameter(6) for _ in range(N)]

        # ────────────────── Build generic objective and constraints ───────────────────────────────────────
        self.add_generic_constraints(opti, pos, vel, F, X0)
        
        # ────────────── Build objective ───────────────────────────────────────────────────────────────
        self.add_payload_tracking_objective(opti, pos, F)

        self._opti = opti
        self._pos = pos
        self._vel = vel
        self._F = F
        self._X0 = X0

        return opti, (pos, vel, F)

    def add_generic_constraints(self, opti: ca.Opti, pos, vel, F, X0) -> None:
        '''Add generic constraints to the optimizer.'''
        N = self.n_drones
        opti_N = self.horizon_steps
        opti_dt = self.params["opti_dt"]
        m_drone = self.params["m_drone"]
        max_thrust = self.params["max_thrust"]
        min_distance = self.params["min_distance"]

        for i in range(N):
            opti.subject_to(pos[i][:, 0] == X0[i][:3])
            opti.subject_to(vel[i][:, 0] == X0[i][3:])

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
    
    def add_payload_tracking_objective(self, opti: ca.Opti, pos, F) -> None:
        '''Add objective to track payload target at target time.'''
        # Find the index of the optimization timestep closest to the payload target time
        opti_dt = self.params["opti_dt"]
        opti_N = self.horizon_steps
        N = self.n_drones
        effort_weight = 1e-4
        smoothness_weight = 1e-2

        target_k = int(self.payload_target_t / opti_dt)
        if target_k >= opti_N:
            target_k = opti_N - 1

        # Average drone position at target time
        avg_pos_at_target = sum(pos[i][:, target_k] for i in range(N)) / N

        # Objective: minimize distance from average position to payload target
        error = avg_pos_at_target - ca.DM(self.payload_target)
        cost = ca.dot(error, error)

        for i in range(N):
            cost += effort_weight * ca.sumsqr(F[i])
            cost += smoothness_weight * ca.sumsqr(pos[i][:, 1:] - pos[i][:, :-1])

        opti.minimize(cost)

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