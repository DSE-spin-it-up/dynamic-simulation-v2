import casadi as ca
import numpy as np

from ..utils.default_params import DEFAULT_PARAMS, VehicleParams, StateLimits, SimParams, OptiVariables, CostWeights
from ..utils.initial_states import cruise_offsets, _equilibrium_forward_offset
from ..utils.config import Config, build_weights, load_config, build_formation_anchor

from .drone import Drone
from .payload import Payload
from src.classes import payload

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
    v_norm = ca.sqrt(ca.dot(vL, vL) + 1e-5)
    F_pay = F_pay - 0.5 * veh.rho * veh.CD0_payload * veh.S_payload * v_norm * vL        # gravity on payload
    for i in range(len(xs)):
        d     = xs[i][3:6] - pL
        u_hat = d / ca.sqrt(ca.dot(d, d) + 1e-5)          # payload -> UAV
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
        self.sim = SimParams()
        self.veh = VehicleParams()
        self.lim = StateLimits()
        self._config: Config = load_config() # dict to extract weights per mission phase
        self._window_time = DEFAULT_PARAMS["opti_N_h"] * DEFAULT_PARAMS["opti_dt"]

        self._prev_sol = None  # stores previous window solution for warm-start shifting

    def update_mission_phase(self, mission_phase: int):
        self.mission_phase = mission_phase

    def calculate_traj_step(self, t, drones: list[Drone], payload: Payload, mission_phase: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
        # ── Update mission phase ──────────────────────────────────────────────────────
        self.mission_phase = mission_phase
        # Get ref for this problem from mission plan, now placeholder of veritcal lift
        self.ref_window = np.vstack([
            np.zeros(self.sim.N_h),                                                    # pn = 0 (stationary north)
            np.zeros(self.sim.N_h),                                                    # pe = 0 (stationary east)
            payload.position[2] + self.lim.climb_rate * np.arange(self.sim.N_h) * self.sim.dt,  # h climbing from current altitude
            ])
        # Get the weights for this problem from the config, now a placeholder for climb
        self.weights = build_weights(self._config.weights, "climb")

        # ── Build generic optimizer ───────────────────────────────────────────────────
        opti, opti_variables = self.build_optimizer(drones, payload)

        # ────────────── Sliding window reference ─────────────────────────────────────
        # Absolute payload reference for this horizon: east at V_cruise, height=100 m
        t_nodes    = t + np.arange(self.sim.N_h) * self.sim.dt
        ref_window = np.vstack([
            np.zeros(self.sim.N_h),
            self.lim.V_cruise * t_nodes,
            np.full(self.sim.N_h, payload.position[2]),
        ])

        # ────────────── Build objective ───────────────────────────────────────────────
        self.add_optimization_objective(opti, opti_variables, ref_window)

        # ────────────── Warm-start ────────────────────────────────────────────────────
        N_h = self.sim.N_h
        if self._prev_sol is None:
            # Tile current positions across the whole horizon — no trajectory propagation
            approx_tension = (self.veh.m_L * self.veh.g) / self.sim.N_uav
            pL_pos = np.tile(payload.position[:, None], (1, N_h))
            pL_vel = np.tile(payload.v[:, None], (1, N_h)) + 1e-4
            opti.set_initial(opti_variables.payload_pos, pL_pos)
            opti.set_initial(opti_variables.payload_vel, pL_vel)
            for i in range(self.sim.N_uav):
                x_guess = np.zeros((6, N_h))
                x_guess[0, :] = np.linalg.norm(drones[i].v)
                x_guess[1, :] = 0.0
                x_guess[2, :] = 0.0
                x_guess[3:6, :] = np.tile(drones[i].position[:, None], (1, N_h))
                opti.set_initial(opti_variables.x[i], x_guess)
                opti.set_initial(opti_variables.u[i],
                                 np.tile([0.0, 0.0, 0.0],
                                         (N_h, 1)).T)
                opti.set_initial(opti_variables.Tc[i], np.full((1, N_h), approx_tension))
        else:
            # Subsequent windows: shift previous solution shifted by N_apply nodes.
            n = self._prev_sol["n_apply"]

            def shift(a):
                return np.hstack([a[:, n:], np.tile(a[:, -1:], (1, n))])

            for i in range(self.sim.N_uav):
                opti.set_initial(opti_variables.x[i],  shift(self._prev_sol["x"][i]))
                opti.set_initial(opti_variables.u[i],  shift(self._prev_sol["u"][i]))
                opti.set_initial(opti_variables.Tc[i], shift(self._prev_sol["Tc"][i]))
            opti.set_initial(opti_variables.payload_pos,
                             shift(self._prev_sol["payload_pos"]))
            opti.set_initial(opti_variables.payload_vel,
                             shift(self._prev_sol["payload_vel"]))

        # ── Solve ─────────────────────────────────────────────────────────────────────

        opti.solver('ipopt', {'expand': True}, {
        'print_level':     3,
        'max_iter':        2000,
        'acceptable_tol':           1e-4,
        'acceptable_iter':          10,
        'acceptable_constr_viol_tol': 1e-4,
        'mu_strategy':     'adaptive',
    })

        try:
            sol = opti.solve()
            # ── Extract solution ──────────────────────────────────────────────────────────
            x_sol = [sol.value(opti_variables.x[i]) for i in range(self.sim.N_uav)]
            sol_time = self._window_time

            # Store full solution for warm-starting the next window
            self._prev_sol = {
                "x":           [np.asarray(sol.value(opti_variables.x[i])).reshape(6, self.sim.N_h)
                                 for i in range(self.sim.N_uav)],
                "u":           [np.asarray(sol.value(opti_variables.u[i])).reshape(3, self.sim.N_h)
                                 for i in range(self.sim.N_uav)],
                "Tc":          [np.asarray(sol.value(opti_variables.Tc[i])).reshape(1, self.sim.N_h)
                                 for i in range(self.sim.N_uav)],
                "payload_pos": np.asarray(sol.value(opti_variables.payload_pos)).reshape(3, self.sim.N_h),
                "payload_vel": np.asarray(sol.value(opti_variables.payload_vel)).reshape(3, self.sim.N_h),
                "n_apply":     DEFAULT_PARAMS["opti_N_apply"],
            }
            return x_sol, sol_time

        except Exception as e:
            print("\n❌ Solver failed! Exception message:", e)
            print("\n=== CASADI IN-DEPTH CONSTRAINT LIMIT ANALYSIS ===")

            return None, None # type: ignore
    
    def build_optimizer(self, drones, payload) -> tuple[ca.Opti, OptiVariables]:
        '''Build a generic casadi optimizer with basic variables, and constraints.'''
        opti = ca.Opti()

        # ────────────────── Create optimization variables ─────────────────────────────────────────

        x = [opti.variable(6, DEFAULT_PARAMS["opti_N_h"]) for _ in range(self.sim.N_uav)]
        u = [opti.variable(3, DEFAULT_PARAMS["opti_N_h"]) for _ in range(self.sim.N_uav)]
        Tc = [opti.variable(1, DEFAULT_PARAMS["opti_N_h"]) for _ in range(self.sim.N_uav)]
        payload_pos = opti.variable(3, DEFAULT_PARAMS["opti_N_h"])
        payload_vel = opti.variable(3, DEFAULT_PARAMS["opti_N_h"])
        opti_variables = OptiVariables(x=x, u=u, Tc=Tc, payload_pos=payload_pos, payload_vel=payload_vel)


        # ────────────────── Build generic objective and constraints ───────────────────────────────────────
        self.add_constraints(opti, opt_variables=opti_variables, drones=drones, payload=payload)

        return opti, opti_variables

    def add_constraints(self, opti: ca.Opti, opt_variables: OptiVariables, drones: list[Drone], payload: Payload) -> None:
        """
        Enforces ONLY the absolute physical boundaries of the system:
          1. Exact drone initialization at t=0 (Hard match to real-world sensors).
          2. Explicit coupled system dynamics continuity over the horizon.
          3. Absolute actuator saturation bounds (thrust, power, aerodynamic angles).
        
        All flexible elements (cable lengths, collision margins, payload initialization deviations) 
        have been offloaded to soft-penalties within the cost function to ensure solver convergence.
        """
        # ── 1. INITIAL CONDITIONS: DRONES (HARD STATE MATCH) ───────────────────
        for i in range(self.sim.N_uav):
            if self._prev_sol is None:
                # FIRST WINDOW: Extract from the simulator objects as a fallback
                vx, vy, vz = drones[i].v

                V0 = np.linalg.norm([vx, vy, vz])
                chi0 = np.arctan2(vy, vx)
                gamma0 = np.arctan2(vz, np.sqrt(vx**2 + vy**2))

                # Protect against exact 0/0 singular derivatives at initialization
                if V0 < 1e-4:
                    V0 = 1e-4

                x0 = ca.vertcat(
                    V0,
                    gamma0,
                    chi0,
                    drones[i].position[0],
                    drones[i].position[1],
                    drones[i].position[2]
                )
                opti.subject_to(opt_variables.payload_pos[:, 0] == payload.position)
                opti.subject_to(opt_variables.payload_vel[:, 0] == payload.v)
            else:
                # SUBSEQUENT WINDOWS: Match the shifted end-state from the last window
                n = self._prev_sol["n_apply"]
                # Look at the first node after shifting the previous window's planned array
                prev_x_shifted = np.hstack([
                    self._prev_sol["x"][i][:, n:], 
                    np.tile(self._prev_sol["x"][i][:, -1:], (1, n))
                ])
                x0 = prev_x_shifted[:, 0]

                prev_pL = np.hstack([self._prev_sol["payload_pos"][:, n:],
                         np.tile(self._prev_sol["payload_pos"][:, -1:], (1, n))])
                prev_vL = np.hstack([self._prev_sol["payload_vel"][:, n:],
                         np.tile(self._prev_sol["payload_vel"][:, -1:], (1, n))])
                opti.subject_to(opt_variables.payload_pos[:, 0] == prev_pL[:, 0])
                opti.subject_to(opt_variables.payload_vel[:, 0] == prev_vL[:, 0])

            # Enforce the hard state match at index 0
            opti.subject_to(opt_variables.x[i][:, 0] == x0)

        # ── 3. ENFORCE COUPLED DYNAMICS PROGRESSION ───────────────────────────
        for k in range(self.sim.N_h - 1):
            xs_dot, pos_dot, vel_dot = coupled_rhs(
                [opt_variables.x[i][:, k] for i in range(self.sim.N_uav)],      # ← k
                [opt_variables.u[i][:, k] for i in range(self.sim.N_uav)],      # ← k
                [opt_variables.Tc[i][:, k] for i in range(self.sim.N_uav)],     # ← k
                opt_variables.payload_pos[:, k], 
                opt_variables.payload_vel[:, k], 
                self.veh, 
                self.sim
            )
            
            # Forward Euler / RK Integration matches must be strictly respected
            for i in range(self.sim.N_uav):
                opti.subject_to(opt_variables.x[i][:, k + 1] == opt_variables.x[i][:, k] + self.sim.dt * xs_dot[i])
            
            opti.subject_to(opt_variables.payload_pos[:, k + 1] == opt_variables.payload_pos[:, k] + self.sim.dt * pos_dot)
            opti.subject_to(opt_variables.payload_vel[:, k + 1] == opt_variables.payload_vel[:, k] + self.sim.dt * vel_dot)        

        # ── 4. ABSOLUTE HARDWARE & ACTUATOR SATURATION LIMITS ─────────────────
        for i in range(self.sim.N_uav):
            # Engine Thrust saturation
            opti.subject_to(
                opti.bounded(self.lim.T_min, opt_variables.u[i][0, :], self.lim.T_max) # type: ignore
            )

            # Mechanical Power ceiling constraints
            P = opt_variables.u[i][0, :] * opt_variables.x[i][0, :]
            opti.subject_to(P <= self.lim.P_max)

            # Aerodynamic Angle of Attack limits
            opti.subject_to(
                opti.bounded(self.lim.alpha_min, opt_variables.u[i][1, :], self.lim.alpha_max) # type: ignore
            )

            # Airframe Bank angle thresholds
            opti.subject_to(
                opti.bounded(-self.lim.mu_max, opt_variables.u[i][2, :], self.lim.mu_max) # type: ignore
            )

            # Airspeed operational envelope limits
            opti.subject_to(
                opti.bounded(self.lim.V_min, opt_variables.x[i][0, :], self.lim.V_max) # type: ignore
            )

            # Flight Path Angle limits
            opti.subject_to(
                opti.bounded(-self.lim.gam_max, opt_variables.x[i][1, :], self.lim.gam_max) # type: ignore
            )

            # Slack Cable Multi-body Tension limits (cannot push with a rope)
            opti.subject_to(
                opti.bounded(0.0, opt_variables.Tc[i], self.lim.Tc_max) # type: ignore
            )
        L_min = self.veh.cable_len - self.veh.cable_tol
        L_max = self.veh.cable_len + self.veh.cable_tol
        for k in range(self.sim.N_h):
            d = opt_variables.x[i][3:6, k] - opt_variables.payload_pos[:, k]
            opti.subject_to(opti.bounded(L_min**2, ca.dot(d, d), L_max**2)) # type: ignore

    def add_optimization_objective(self, opti: ca.Opti, opt_variables: OptiVariables,
                                       ref: np.ndarray) -> None:

        # ── Cost ──────────────────────────────────────────────────────────────────
        # Every term is weighted by the constant scalars in `weights` (W_du is a
        # (3, 1) column, one entry per control) and summed over nodes. Rate terms
        # span node pairs.
        #
        # Payload position tracking (primary objective).
        cost = self.weights.W_track * ca.sum2(ca.sum1((opt_variables.payload_pos - ref)**2))

        # Per-control rate scaling, one entry per input row of u:
        # [thrust T, angle of attack alpha, bank angle mu].
        du_scale = ca.vertcat(self.lim.T_max - self.lim.T_min,
                              self.lim.alpha_max - self.lim.alpha_min,
                              2 * self.lim.mu_max)
        for i in range(self.sim.N_uav):
            # Per-control rate, weighted per control by W_du.
            du = (opt_variables.u[i][:, 1:] - opt_variables.u[i][:, :-1]) / du_scale
            cost += ca.sum2(ca.sum1(self.weights.W_du * du**2))

            # Cable-tension rate.
            dTc = (opt_variables.Tc[i][:, 1:] - opt_variables.Tc[i][:, :-1]) / self.lim.Tc_max
            cost += self.weights.W_dTc * ca.sum2(dTc**2)

            # Flight-path-angle rate: gamma = x[1], normalized by its admissible
            # range (±gam_max). Penalising the rate smooths the climb/descent profile.
            dgamma = (opt_variables.x[i][1, 1:] - opt_variables.x[i][1, :-1]) / (2 * self.lim.gam_max)
            cost += self.weights.W_dgamma * ca.sum2(dgamma**2)

            # Heading rate: chi = x[2]. sin(Δchi) handles ±π wrap-around; for the
            # small per-step changes expected here sin(Δchi) ≈ Δchi.
            dchi = ca.sin(opt_variables.x[i][2, 1:] - opt_variables.x[i][2, :-1])
            cost += self.weights.W_dchi * ca.sum2(dchi**2)

            # Airspeed rate: V = x[0], normalized by its admissible range.
            dV = (opt_variables.x[i][0, 1:] - opt_variables.x[i][0, :-1]) / (self.lim.V_max - self.lim.V_min)
            cost += self.weights.W_dV * ca.sum2(dV**2)

            # Thrust magnitude penalty to encourage lower thrust when possible (e.g.
            # for energy efficiency). Normalized by T_max to keep it in scale.
            cost += self.weights.W_T * ca.sum2((opt_variables.u[i][0, :] / self.lim.T_max)**2)

        opti.minimize(cost)