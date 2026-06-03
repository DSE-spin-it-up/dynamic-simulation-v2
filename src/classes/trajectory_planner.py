import casadi as ca
import numpy as np

from ..utils.default_params import DEFAULT_PARAMS, VehicleParams, StateLimits, SimParams, NLP

from .drone import Drone
from .payload import Payload

# --------------- helper physics functions for the aircraft dyamics -------------------------------
def drone_rhs(x, u, F_ext, veh: VehicleParams):
    """3-DOF fixed-wing point-mass dynamics with external force F_ext (CasADi)."""
    V, gamma, chi = x[0], x[1], x[2]
    T, alpha, mu  = u[0], u[1], u[2]

    q = 0.5 * veh.rho * V**2 * veh.S
    CL = veh.CL0 + veh.CLa * alpha
    L = q * CL
    D = q * (veh.CD0 + 1 / (np.pi * veh.AR * veh.e) * CL**2)

    t_hat = ca.vertcat(ca.cos(gamma) * ca.cos(chi),
                       ca.cos(gamma) * ca.sin(chi), ca.sin(gamma))
    n_hat = ca.vertcat(-ca.sin(gamma) * ca.cos(chi),
                       -ca.sin(gamma) * ca.sin(chi), ca.cos(gamma))
    h_hat = ca.vertcat(-ca.sin(chi), ca.cos(chi), 0.0)
    Ft, Fn, Fh = ca.dot(F_ext, t_hat), ca.dot(F_ext, n_hat), ca.dot(F_ext, h_hat)

    V_dot     = (T * ca.cos(alpha) - D) / veh.m - veh.g * ca.sin(gamma) + Ft / veh.m
    gamma_dot = ((L + T * ca.sin(alpha)) * ca.cos(mu) - veh.m * veh.g * ca.cos(gamma)
                 + Fn) / (veh.m * V)
    chi_dot   = ((L + T * ca.sin(alpha)) * ca.sin(mu) + Fh) \
                / (veh.m * V * ca.cos(gamma))
    pn_dot = V * ca.cos(chi) * ca.cos(gamma)
    pe_dot = V * ca.sin(chi) * ca.cos(gamma)
    h_dot  = V * ca.sin(gamma)
    return ca.vertcat(V_dot, gamma_dot, chi_dot, pn_dot, pe_dot, h_dot)


def payload_rhs(pL, vL, xs, Tcs, veh: VehicleParams):
    """Payload point-mass dynamics: gravity + cable-tension reactions.
    Returns the state derivative (pL_dot, vL_dot). Cable directions are
    recomputed from the current UAV positions in xs."""
    F_pay = ca.vertcat(0.0, 0.0, -veh.m_L * veh.g)
    v_norm = ca.sqrt(ca.dot(vL, vL) + 1e-9)
    F_pay = F_pay - 0.5 * veh.rho * veh.CD0_payload * veh.S_payload * v_norm * vL        # gravity on payload
    for i in range(len(xs)):
        d     = xs[i][3:6] - pL
        u_hat = d / ca.sqrt(ca.dot(d, d) + 1e-9)          # payload -> UAV
        F_pay = F_pay + Tcs[i] * u_hat                    # cable reaction
    pL_dot = vL
    vL_dot = F_pay / veh.m_L
    return pL_dot, vL_dot


def coupled_rhs(xs, us, Tcs, pL, vL, veh: VehicleParams, sim: SimParams):
    """Time derivatives of the WHOLE coupled system (all UAVs + payload), with
    cable tensions Tcs held constant over dt. Pure dynamics evaluation; the
    integration scheme is applied by the caller (see build_nlp)."""
    xs_dot = []
    for i in range(sim.N_uav):
        d     = xs[i][3:6] - pL
        u_hat = d / ca.sqrt(ca.dot(d, d) + 1e-9)            # payload -> UAV
        xs_dot.append(drone_rhs(xs[i], us[i], -Tcs[i] * u_hat, veh))

    pL_dot, vL_dot = payload_rhs(pL, vL, xs, Tcs, veh)
    return xs_dot, pL_dot, vL_dot

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
        self.sim = SimParams()
        self.veh = VehicleParams()
        self.lim = StateLimits()

    def udpate_mission_phase(self, mission_phase: int):
        self.mission_phase = mission_phase

    def set_payload_target(self, target, target_time):
        '''Update payload target for trajectory optimization.'''
        self.payload_target = np.asarray(target, dtype=float)
        self.payload_target_t = float(target_time)

    def calculate_traj_step(self, t, drones: list[Drone], payload: Payload, mission_phase: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
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

        x           = [opti.variable(6, self.sim.N) for _ in range(self.sim.N_uav)]
        u           = [opti.variable(3, self.sim.N) for _ in range(self.sim.N_uav)]
        Tc          = [opti.variable(1, self.sim.N) for _ in range(self.sim.N_uav)]
        payload_pos = opti.variable(3, self.sim.N)
        payload_vel = opti.variable(3, self.sim.N)


        # ────────────────── Build generic objective and constraints ───────────────────────────────────────
        self.add_generic_constraints(opti, x, u, Tc)
        
        # ────────────── Build objective ───────────────────────────────────────────────────────────────
        self.add_payload_tracking_objective(opti, x)

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
    
    def add_payload_tracking_objective(self, opti: ca.Opti, payload_pos: ca.MX) -> None:
        '''Add objective to track payload target at target time.'''
        # Cost: track the reference with the payload, plus a thrust-effort penalty.
        cost = ca.sumsqr(payload_pos - ref)
        # for i in range(sim.N_uav):
        #     cost += R_T * ca.sumsqr(u[i][0, :])  # thrust effort
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