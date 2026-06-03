import casadi as ca
import numpy as np

from ..utils.default_params import DEFAULT_PARAMS, VehicleParams, StateLimits, SimParams, OptiVariables

from .drone import Drone
from .payload import Payload
from src.classes import cable, payload

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
        self._x = None
        self._u = None
        self._Tc = None
        self._payload_pos = None
        self._payload_vel = None
        self._last_x_sol = None
        self._last_u_sol = None
        self._last_Tc_sol = None
        self.last_payload_pos_sol = None
        self.last_payload_vel_sol = None
        self.sim = SimParams()
        self.veh = VehicleParams()
        self.lim = StateLimits()
        self.ref = np.vstack([
            np.zeros_like(np.arange(DEFAULT_PARAMS["opti_timepstep_N"]) * DEFAULT_PARAMS["dt"]), 
            np.zeros_like(np.arange(DEFAULT_PARAMS["opti_timepstep_N"]) * DEFAULT_PARAMS["dt"]), 
            100.0 + 3.33 * np.arange(DEFAULT_PARAMS["opti_timepstep_N"]) * DEFAULT_PARAMS["dt"]
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
        opti, x, u = self.build_generic_optimizer(drones, payload)

        # Warm-start the solver with the previous optimal trajectory when available.
        if self._last_x_sol is not None and self._last_u_sol is not None and self._last_Tc_sol is not None and self.last_payload_pos_sol is not None and self.last_payload_vel_sol is not None:
            shift = 1 if self._last_x_sol[0].shape[1] > 1 else 0 # type: ignore
            for i in range(min(len(drones), len(self._last_x_sol))): # type: ignore
                pos_guess = np.hstack([
                    self._last_x_sol[i][:, shift:], # type: ignore
                    np.tile(self._last_x_sol[i][:, -1:], (1, shift if shift > 0 else 1)), # type: ignore
                ])
                u_guess = np.hstack([
                    self._last_u_sol[i][:, shift:], # type: ignore
                    np.tile(self._last_u_sol[i][:, -1:], (1, shift if shift > 0 else 1)), # type: ignore
                ])
                Tc_guess = np.hstack([
                    self._last_Tc_sol[i][:, shift:], # type: ignore
                    np.tile(self._last_Tc_sol[i][:, -1:], (1, shift if shift > 0 else 1)), # type: ignore
                ])
                payload_pos_guess = np.hstack([
                    self.last_payload_pos_sol[:, shift:], # type: ignore
                    np.tile(self.last_payload_pos_sol[:, -1:], (1, shift if shift > 0 else 1)), # type: ignore
                ])
                payload_vel_guess = np.hstack([
                    self.last_payload_vel_sol[:, shift:], # type: ignore
                    np.tile(self.last_payload_vel_sol[:, -1:], (1, shift if shift > 0 else 1)), # type: ignore
                ])
                opti.set_initial(self._x[i], pos_guess[:, :self.horizon_steps]) # type: ignore
                opti.set_initial(self._u[i], u_guess[:, :self.horizon_steps]) # type: ignore
                opti.set_initial(self._Tc[i], Tc_guess[:, :self.horizon_steps]) # type: ignore
                opti.set_initial(self._payload_pos, payload_pos_guess[:, :self.horizon_steps]) # type: ignore
                opti.set_initial(self._payload_vel, payload_vel_guess[:, :self.horizon_steps]) # type: ignore

        else:
            for i, drone in enumerate(drones):

                x_guess = np.hstack(
                    (drone.position,
                     drone.v)
                )

                opti.set_initial(
                    self._x[i], # type: ignore
                    np.tile(
                        x_guess[:, None],
                        (1, self.sim.N)
                    )
                )

                opti.set_initial(
                    self._u[i], # type: ignore
                    np.zeros((3, self.sim.N))
                )

                opti.set_initial(
                    self._Tc[i], # type: ignore
                    np.ones((1, self.sim.N))
                )

            opti.set_initial(
                self._payload_pos, # type: ignore
                np.tile(payload.position[:, None], (1, self.sim.N))
            )

            opti.set_initial(
                self._payload_vel, # type: ignore
                np.tile(payload.v[:, None], (1, self.sim.N))
            )


        # ── Set specific contraints per phase ─────────────────────────────────────────


        # ── Solve ─────────────────────────────────────────────────────────────────────

        opti.solver('ipopt', {'expand': True}, {
            'print_level':     3,
            'max_iter':        500,
            'acceptable_tol':           1e-4,
            'acceptable_iter':          10,
            'acceptable_constr_viol_tol': 1e-4,
            'mu_strategy':     'adaptive',
        })
        sol = opti.solve()

        # ── Extract solution ──────────────────────────────────────────────────────────

        N = DEFAULT_PARAMS.get("n_drones", len(drones))
        x_sol = [sol.value(self._x[i]) for i in range(N)]   # type: ignore
        u_sol = [sol.value(self._u[i]) for i in range(N)]   # type: ignore

        self._last_x_sol = x_sol
        self._last_u_sol = [sol.value(self._u[i]) for i in range(N)] # type: ignore
        self._last_Tc_sol = [sol.value(self._Tc[i]) for i in range(N)] # type: ignore
        self._last_payload_pos_sol = sol.value(self._payload_pos)
        self._last_payload_vel_sol = sol.value(self._payload_vel)

        return x_sol, u_sol
    
    def build_generic_optimizer(self, drones, payload) -> tuple[ca.Opti, list[ca.MX], list[ca.MX]]:
        '''Build a generic casadi optimizer with basic variables, objective, and constraints.'''
        opti = ca.Opti()

        # ────────────────── Create optimization variables ─────────────────────────────────────────

        x = [opti.variable(6, self.sim.N) for _ in range(self.sim.N_uav)]
        u = [opti.variable(3, self.sim.N) for _ in range(self.sim.N_uav)]
        Tc = [opti.variable(1, self.sim.N) for _ in range(self.sim.N_uav)]
        payload_pos = opti.variable(3, self.sim.N)
        payload_vel = opti.variable(3, self.sim.N)
        opti_variables = OptiVariables(x=x, u=u, Tc=Tc, payload_pos=payload_pos, payload_vel=payload_vel)


        # ────────────────── Build generic objective and constraints ───────────────────────────────────────
        self.add_generic_constraints(opti, opt_variables=opti_variables, drones=drones, payload=payload)
        
        # ────────────── Build objective ───────────────────────────────────────────────────────────────
        self.add_payload_tracking_objective(opti, opti_variables.payload_pos)

        self._opti = opti
        self._x = opti_variables.x
        self._u = opti_variables.u
        self._Tc = opti_variables.Tc
        self._payload_pos = opti_variables.payload_pos
        self._payload_vel = opti_variables.payload_vel

        return opti, opti_variables.x, opti_variables.u

    def add_generic_constraints(
        self,
        opti: ca.Opti,
        opt_variables: OptiVariables,
        drones: list[Drone],
        payload: Payload
    ) -> None:

        N = self.sim.N
        dt = self.sim.dt

        # ============================================================
        # 1. INITIAL CONDITIONS (WELL-POSED, NOT OVERCONSTRAINED)
        # ============================================================

        for i in range(self.sim.N_uav):
            opti.subject_to(
                opt_variables.x[i][:, 0] ==
                ca.vertcat(drones[i].position, drones[i].v)
            )

        # Payload: do NOT fix exactly (prevents infeasibility)
        opti.subject_to(
            ca.sumsqr(opt_variables.payload_pos[:, 0] - payload.position) <= 1e-6
        )
        opti.subject_to(
            ca.sumsqr(opt_variables.payload_vel[:, 0] - payload.v) <= 1e-6
        )

        # ============================================================
        # 2. DYNAMICS (START AT k = 1, NOT k = 0)
        # ============================================================

        for k in range(1, N - 1):

            xs_dot, p_dot, v_dot = coupled_rhs(
                [opt_variables.x[i][:, k + 1] for i in range(self.sim.N_uav)],
                [opt_variables.u[i][:, k + 1] for i in range(self.sim.N_uav)],
                [opt_variables.Tc[i][:, k + 1] for i in range(self.sim.N_uav)],
                opt_variables.payload_pos[:, k + 1],
                opt_variables.payload_vel[:, k + 1],
                self.veh,
                self.sim
            )

            for i in range(self.sim.N_uav):
                opti.subject_to(
                    opt_variables.x[i][:, k + 1]
                    == opt_variables.x[i][:, k] + dt * xs_dot[i]
                )

            opti.subject_to(
                opt_variables.payload_pos[:, k + 1]
                == opt_variables.payload_pos[:, k] + dt * p_dot
            )

            opti.subject_to(
                opt_variables.payload_vel[:, k + 1]
                == opt_variables.payload_vel[:, k] + dt * v_dot
            )

        # ============================================================
        # 3. INPUT AND STATE LIMITS (UNCHANGED)
        # ============================================================

        for i in range(self.sim.N_uav):

            opti.subject_to(
                opti.bounded(self.lim.T_min, opt_variables.u[i][0, :], self.lim.T_max) # type: ignore
            )

            opti.subject_to(
                opt_variables.u[i][0, :] * opt_variables.x[i][0, :] <= self.lim.P_max
            )

            opti.subject_to(
                opti.bounded(self.lim.alpha_min, opt_variables.u[i][1, :], self.lim.alpha_max) # type: ignore
            )

            opti.subject_to(
                opti.bounded(-self.lim.mu_max, opt_variables.u[i][2, :], self.lim.mu_max) # type: ignore
            )

            opti.subject_to(
                opti.bounded(self.lim.V_min, opt_variables.x[i][0, :], self.lim.V_max) # type: ignore
            )

            opti.subject_to(
                opti.bounded(-self.lim.gam_max, opt_variables.x[i][1, :], self.lim.gam_max) # type: ignore
            )

            opti.subject_to(
                opti.bounded(0.0, opt_variables.Tc[i], self.lim.Tc_max) # type: ignore
            )

        # ============================================================
        # 4. SOFT CABLE CONSTRAINT (REPLACES HARD INFEASIBLE CONSTRAINT)
        # ============================================================

        eps_cable = 0.3
        w_cable = 100.0

        cable_cost = 0

        for i in range(self.sim.N_uav):
            for k in range(1, N):   # IMPORTANT: skip k=0

                d = opt_variables.x[i][3:6, k] - opt_variables.payload_pos[:, k]

                dist2 = ca.dot(d, d)
                err = dist2 - self.veh.cable_len**2

                cable_cost += w_cable * err**2 + eps_cable * ca.fabs(err)

        # add to objective later via shared cost
        self._cable_cost = cable_cost

        # ============================================================
        # 5. COLLISION AVOIDANCE (KEEP HARD BUT STABLE)
        # ============================================================

        for i in range(self.sim.N_uav):
            for j in range(i + 1, self.sim.N_uav):
                for k in range(1, N):

                    d = opt_variables.x[i][3:6, k] - opt_variables.x[j][3:6, k]

                    opti.subject_to(
                        ca.dot(d, d) >= self.lim.d_min**2
                    )
    
    def add_payload_tracking_objective(self, opti: ca.Opti, payload_pos: ca.MX) -> None:

        tracking = ca.sumsqr(payload_pos - self.ref)

        cable = getattr(self, "_cable_cost", 0)

        cost = tracking + cable

        opti.minimize(cost)

    def custom_constraint_placeholder(self, opti: ca.Opti) -> None:
        # Replace this with actual constraints for specific mission phases as needed
        return None

    def constraint_placeholder(self, opti: ca.Opti) -> None:
        # Replace this with actual constraints for specific mission phases as needed
        return None