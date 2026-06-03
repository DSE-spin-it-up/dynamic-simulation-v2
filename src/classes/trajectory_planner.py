import casadi as ca
import numpy as np

from ..utils.default_params import DEFAULT_PARAMS, VehicleParams, StateLimits, SimParams, OptiVariables

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
        self.ref = np.vstack([
            np.zeros_like(np.arange(DEFAULT_PARAMS["opti_timepstep_N"]) * DEFAULT_PARAMS["dt"]), 
            np.zeros_like(np.arange(DEFAULT_PARAMS["opti_timepstep_N"]) * DEFAULT_PARAMS["dt"]), 
            100.0 + 3.33 * DEFAULT_PARAMS["opti_timepstep_N"] * DEFAULT_PARAMS["dt"]
            ])

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

        x = [opti.variable(6, self.sim.N) for _ in range(self.sim.N_uav)]
        u = [opti.variable(3, self.sim.N) for _ in range(self.sim.N_uav)]
        Tc = [opti.variable(1, self.sim.N) for _ in range(self.sim.N_uav)]
        payload_pos = opti.variable(3, self.sim.N)
        payload_vel = opti.variable(3, self.sim.N)
        opti_variables = OptiVariables(x=x, u=u, Tc=Tc, payload_pos=payload_pos, payload_vel=payload_vel)


        # ────────────────── Build generic objective and constraints ───────────────────────────────────────
        self.add_generic_constraints(opti, x0=None, u0=None, Tc0=None, opt_variables=opti_variables)
        
        # ────────────── Build objective ───────────────────────────────────────────────────────────────
        self.add_payload_tracking_objective(opti, x)

        self._opti = opti
        self._pos = pos
        self._vel = vel
        self._F = F
        self._X0 = X0

        return opti, (pos, vel, F)

    def add_generic_constraints(self, opti: ca.Opti, x0=None, u0=None, Tc0=None, opt_variables=None) -> None:
        '''Add generic constraints to the optimizer.'''
        # Initial condition: free on the first window (x0 None), pinned otherwise.
        if x0 is not None:
            uav_states0, payload_pos0, payload_vel0 = x0
            for i in range(self.sim.N_uav):
                opti.subject_to(x[i][:, 0] == uav_states0[i])
            opti.subject_to(opt_variables.payload_pos[:, 0] == payload_pos0)
            opti.subject_to(opt_variables.payload_vel[:, 0] == payload_vel0)

        # Control continuity across windows. Under backward Euler u[:,0] enters no
        # dynamics constraint (the step into node k uses u[:,k]), so it is otherwise
        # a free variable; pinning it to the previous window's continuation control
        # keeps the committed control history continuous at the window seams.
        if u0 is not None:
            for i in range(self.sim.N_uav):
                opti.subject_to(u[i][:, 0] == u0[i])

        if Tc0 is not None:
            for i in range(self.sim.N_uav):
                opti.subject_to(Tc[i][:, 0] == Tc0[i])

        # Dynamics: backward (implicit) Euler. The increment over each step uses the
        # derivative evaluated at the *next* node: x_{k+1} = x_k + dt * f(x_{k+1}).
        for k in range(self.sim.N - 1):
            xs_dot, pos_dot, vel_dot = coupled_rhs(
                [x[i][:, k + 1] for i in range(self.sim.N_uav)],
                [u[i][:, k + 1] for i in range(self.sim.N_uav)],
                [Tc[i][:, k + 1] for i in range(self.sim.N_uav)],
                payload_pos[:, k + 1], payload_vel[:, k + 1], self.veh, self.sim)
            for i in range(self.sim.N_uav):
                opti.subject_to(x[i][:, k + 1] == x[i][:, k] + self.sim.dt * xs_dot[i])
            opti.subject_to(payload_pos[:, k + 1] == payload_pos[:, k] + self.sim.dt * pos_dot)
            opti.subject_to(payload_vel[:, k + 1] == payload_vel[:, k] + self.sim.dt * vel_dot)

        # Control limits: thrust, propulsive power, angle of attack, bank angle.
        for i in range(self.sim.N_uav):
            opti.subject_to(opti.bounded(self.lim.T_min,     u[i][0, :], self.lim.T_max))
            opti.subject_to(u[i][0, :] * x[i][0, :] <= self.lim.P_max)
            opti.subject_to(opti.bounded(self.lim.alpha_min, u[i][1, :], self.lim.alpha_max))
            opti.subject_to(opti.bounded(-self.lim.mu_max,   u[i][2, :], self.lim.mu_max))

            # State limits: airspeed and flight-path angle.
            opti.subject_to(opti.bounded(self.lim.V_min,     x[i][0, :], self.lim.V_max))
            opti.subject_to(opti.bounded(-self.lim.gam_max,  x[i][1, :], self.lim.gam_max))

            # Cable tension: cables can only pull (Tc >= 0) and have a max rating.
            opti.subject_to(opti.bounded(0.0, Tc[i], self.lim.Tc_max))

            # Taut cables: keep each UAV cable_len from the payload, within a small fixed
            # tolerance band on the squared distance (hard equality is too stiff to solve).
            eps = 1e-1
            for k in range(self.sim.N):
                d = x[i][3:6, k] - payload_pos[:, k]
                opti.subject_to(opti.bounded(self.veh.cable_len**2 - eps,
                                             ca.dot(d, d),
                                             self.veh.cable_len**2 + eps))

            # Collision avoidance: keep every pair of UAVs at least d_min apart.
            for j in range(i + 1, self.sim.N_uav):
                for k in range(self.sim.N):
                    d = x[i][3:6, k] - x[j][3:6, k]
                    opti.subject_to(ca.dot(d, d) >= self.lim.d_min**2)
    
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
        

    def build_nlp(self, ref, x0=None, print_level=0, u0=None, Tc0=None):
        """Build a simple trajectory-tracking NLP over sim.N timesteps.

        If `x0` is None the initial state is left free: the optimizer chooses
        whatever feasible starting state best tracks the payload reference `ref`
        (3, N). If `x0 = (uav_states0, payload_pos0, payload_vel0)` is given, the
        state at node 0 is pinned to it (used to stitch receding-horizon windows).

        The system (UAVs + cable-suspended payload) is constrained by its dynamics,
        the taut cables, and the state/control limits.

        Cost: squared distance from the payload to the reference, plus a
        weighted thrust-effort penalty.
        Constraints: coupled dynamics, taut cables, and box limits on thrust, angle
        of attack, and bank angle.
        """
        assert ref.shape == (3, self.sim.N), f"ref must be (3, {self.sim.N}), got {ref.shape}"
        opti = ca.Opti()

        # Decision variables: one set per UAV, plus the payload.
        x           = [opti.variable(6, self.sim.N) for _ in range(self.sim.N_uav)]
        u           = [opti.variable(3, self.sim.N) for _ in range(self.sim.N_uav)]
        Tc          = [opti.variable(1, self.sim.N) for _ in range(self.sim.N_uav)]
        payload_pos = opti.variable(3, self.sim.N)
        payload_vel = opti.variable(3, self.sim.N)

        # Cost: track the reference with the payload, plus a thrust-effort penalty.
        R_T = 1e-3  # thrust effort weight
        cost = ca.sumsqr(payload_pos - ref)
        # for i in range(self.sim.N_uav):
        #     cost += R_T * ca.sumsqr(u[i][0, :])  # thrust effort
        opti.minimize(cost)

        # Initial condition: free on the first window (x0 None), pinned otherwise.
        if x0 is not None:
            uav_states0, payload_pos0, payload_vel0 = x0
            for i in range(self.sim.N_uav):
                opti.subject_to(x[i][:, 0] == uav_states0[i])
            opti.subject_to(payload_pos[:, 0] == payload_pos0)
            opti.subject_to(payload_vel[:, 0] == payload_vel0)

        # Control continuity across windows. Under backward Euler u[:,0] enters no
        # dynamics constraint (the step into node k uses u[:,k]), so it is otherwise
        # a free variable; pinning it to the previous window's continuation control
        # keeps the committed control history continuous at the window seams.
        if u0 is not None:
            for i in range(self.sim.N_uav):
                opti.subject_to(u[i][:, 0] == u0[i])

        if Tc0 is not None:
            for i in range(self.sim.N_uav):
                opti.subject_to(Tc[i][:, 0] == Tc0[i])

        # Dynamics: backward (implicit) Euler. The increment over each step uses the
        # derivative evaluated at the *next* node: x_{k+1} = x_k + dt * f(x_{k+1}).
        for k in range(self.sim.N - 1):
            xs_dot, pos_dot, vel_dot = coupled_rhs(
                [x[i][:, k + 1] for i in range(self.sim.N_uav)],
                [u[i][:, k + 1] for i in range(self.sim.N_uav)],
                [Tc[i][:, k + 1] for i in range(self.sim.N_uav)],
                payload_pos[:, k + 1], payload_vel[:, k + 1], self.veh, self.sim)
            for i in range(self.sim.N_uav):
                opti.subject_to(x[i][:, k + 1] == x[i][:, k] + self.sim.dt * xs_dot[i])
            opti.subject_to(payload_pos[:, k + 1] == payload_pos[:, k] + self.sim.dt * pos_dot)
            opti.subject_to(payload_vel[:, k + 1] == payload_vel[:, k] + self.sim.dt * vel_dot)

        # Control limits: thrust, propulsive power, angle of attack, bank angle.
        for i in range(self.sim.N_uav):
            opti.subject_to(opti.bounded(self.lim.T_min,     u[i][0, :], self.lim.T_max))
            opti.subject_to(u[i][0, :] * x[i][0, :] <= self.lim.P_max)
            opti.subject_to(opti.bounded(self.lim.alpha_min, u[i][1, :], self.lim.alpha_max))
            opti.subject_to(opti.bounded(-self.lim.mu_max,   u[i][2, :], self.lim.mu_max))

            # State limits: airspeed and flight-path angle.
            opti.subject_to(opti.bounded(self.lim.V_min,     x[i][0, :], self.lim.V_max))
            opti.subject_to(opti.bounded(-self.lim.gam_max,  x[i][1, :], self.lim.gam_max))

            # Cable tension: cables can only pull (Tc >= 0) and have a max rating.
            opti.subject_to(opti.bounded(0.0, Tc[i], self.lim.Tc_max))

            # Taut cables: keep each UAV cable_len from the payload, within a small fixed
            # tolerance band on the squared distance (hard equality is too stiff to solve).
            eps = 1e-1
            for k in range(self.sim.N):
                d = x[i][3:6, k] - payload_pos[:, k]
                opti.subject_to(opti.bounded(self.veh.cable_len**2 - eps,
                                             ca.dot(d, d),
                                             self.veh.cable_len**2 + eps))

            # Collision avoidance: keep every pair of UAVs at least d_min apart.
            for j in range(i + 1, self.sim.N_uav):
                for k in range(self.sim.N):
                    d = x[i][3:6, k] - x[j][3:6, k]
                    opti.subject_to(ca.dot(d, d) >= self.lim.d_min**2)

        opti.solver('ipopt', {'expand': True}, {
            'print_level':     print_level,
            'max_iter':        2000,
            'acceptable_tol':           1e-4,
            'acceptable_iter':          10,
            'acceptable_constr_viol_tol': 1e-4,
            'mu_strategy':     'adaptive',
        })
        return NLP(opti=opti, x=x, u=u, Tc=Tc,
               payload_pos=payload_pos, payload_vel=payload_vel)