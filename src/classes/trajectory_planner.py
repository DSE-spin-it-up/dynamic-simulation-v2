import casadi as ca
import numpy as np

from ..utils.default_params import DEFAULT_PARAMS, VehicleParams, StateLimits, SimParams, OptiVariables
from ..utils.initial_states import cruise_offsets, _equilibrium_forward_offset

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
        self._window_time = DEFAULT_PARAMS["opti_N_h"] * DEFAULT_PARAMS["opti_dt"]

        # # Cruise reference configuration (heading east, payload at 100 m altitude)
        # self.heading        = np.pi / 2
        # self.lateral_offset = 6.0
        # self.payload_pos0   = np.array([0.0, 0.0, 100.0])
        # self.uav_offsets    = cruise_offsets(self.veh, self.lim, self.heading,
        #                                      self.lateral_offset)

        # # Pre-compute cruise trim values for warm-starting
        # nu = self.sim.N_uav
        # _F_drag    = 0.5 * self.veh.rho * self.veh.CD0_payload * self.veh.S_payload * self.lim.V_cruise**2
        # self._Tc_trim = float(np.sqrt((self.veh.m_L * self.veh.g / nu)**2 + (_F_drag / nu)**2))
        # _f_eq      = _equilibrium_forward_offset(self.veh, self.lim, self.lateral_offset)
        # _f_frac    = _f_eq / self.veh.cable_len
        # _v_frac    = np.sqrt(max(1.0 - _f_frac**2, 0.0))
        # _q         = 0.5 * self.veh.rho * self.lim.V_cruise**2 * self.veh.S
        # _CL_trim   = (self.veh.m * self.veh.g + self._Tc_trim * _v_frac) / _q
        # _D_trim    = _q * (self.veh.CD0 + _CL_trim**2 / (np.pi * self.veh.AR * self.veh.e))
        # self._T_trim     = float(np.clip(_D_trim + self._Tc_trim * _f_frac,
        #                                  self.lim.T_min, self.lim.T_max))
        # self._alpha_trim = float(np.clip((_CL_trim - self.veh.CL0) / self.veh.CLa,
        #                                  self.lim.alpha_min, self.lim.alpha_max))

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
        self.add_payload_tracking_objective(opti, opti_variables, ref_window)

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
            'print_level': 3,
            'max_iter': 1000,
            'nlp_scaling_method': 'none',             # Keeps Ipopt from tripping over early gradients
            'obj_scaling_factor': 1e-3,              
            'acceptable_tol': 1e-3,
            'acceptable_iter': 20,
            'mu_strategy': 'adaptive',
            'warm_start_init_point': 'yes',          
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
            
            try:
                # Fetch structural info from the debugger
                vals = opti.debug.value(opti.g)
                lbg = opti.debug.lbg
                ubg = opti.debug.ubg
                
                # Try to grab the dual multipliers (lam_g) if available before failure
                # High multipliers mean the solver is fighting desperately against this bound.
                try:
                    lam_g = opti.debug.value(opti.lam_g)
                except Exception:
                    lam_g = None

                print(f"{'Row':<6} | {'Current Value':<15} | {'Lower Bound':<15} | {'Upper Bound':<15} | {'Status/Limiting Reason'}")
                print("-" * 100)

                for row in range(opti.g.shape[0]):
                    g_val = float(vals[row])
                    low = float(lbg[row]) if lbg is not None else -np.inf
                    up = float(ubg[row]) if ubg is not None else np.inf
                    
                    # Track why it's limiting
                    status = ""
                    is_limiting = False
                    
                    # 1. Check for absolute violations or tight constraints
                    if np.isnan(g_val) or np.isinf(g_val):
                        status = "❌ NaN / INF VALUE"
                        is_limiting = True
                    elif g_val < low or np.isclose(g_val, low, atol=1e-4):
                        status = f"💥 CRUSHING LOWER BOUND (Diff: {g_val - low:.5f})"
                        is_limiting = True
                    elif g_val > up or np.isclose(g_val, up, atol=1e-4):
                        status = f"💥 CRUSHING UPPER BOUND (Diff: {up - g_val:.5f})"
                        is_limiting = True

                    # 2. Add multiplier context if available
                    if lam_g is not None and abs(lam_g[row]) > 1e-2:
                        status += f" [High Pressure/Multiplier: {lam_g[row]:.2f}]"
                        is_limiting = True

                    # Only print constraints that are actively active, limiting, or violated
                    if is_limiting:
                        print(f"{row:<6} | {g_val:<15.4f} | {low:<15.4f} | {up:<15.4f} | {status}")
                        print(f"  └─ Description: {opti.debug.g_describe(row)}")
                        
            except Exception as e:
                print("\n❌ Solver failed! Exception message:", e)
                print("\n=== CASADI IN-DEPTH CONSTRAINT LIMIT ANALYSIS ===")
            
            try:
                # Use opti.debug.value to turn symbolic expressions into actual numbers
                vals = opti.debug.value(opti.g)
                lbg_vals = opti.debug.value(opti.lbg)
                ubg_vals = opti.debug.value(opti.ubg)
                
                try:
                    lam_g = opti.debug.value(opti.lam_g)
                except Exception:
                    lam_g = None

                print(f"{'Row':<6} | {'Current Value':<14} | {'Lower Bound':<14} | {'Upper Bound':<14} | {'Status/Limiting Reason'}")
                print("-" * 110)

                violation_count = 0
                for row in range(opti.g.shape[0]):
                    g_val = float(vals[row])
                    low = float(lbg_vals[row]) if lbg_vals is not None else -np.inf
                    up = float(ubg_vals[row]) if ubg_vals is not None else np.inf
                    
                    status = ""
                    is_limiting = False
                    
                    # 1. Check for absolute violations or highly active bounds
                    if np.isnan(g_val) or np.isinf(g_val):
                        status = "❌ NaN / INF VALUE"
                        is_limiting = True
                    elif g_val < low - 1e-4:
                        status = f"❌ VIOLATED LOWER (Diff: {g_val - low:.5f})"
                        is_limiting = True
                    elif g_val > up + 1e-4:
                        status = f"❌ VIOLATED UPPER (Diff: {up - g_val:.5f})"
                        is_limiting = True
                    elif np.isclose(g_val, low, atol=1e-3):
                        status = "⚠️ AT LOWER BOUND"
                        is_limiting = True
                    elif np.isclose(g_val, up, atol=1e-3):
                        status = "⚠️ AT UPPER BOUND"
                        is_limiting = True

                    # 2. Add multiplier context if available
                    if lam_g is not None and abs(lam_g[row]) > 1e-1:
                        status += f" [Multiplier: {lam_g[row]:.2f}]"
                        is_limiting = True

                    # Print the row if it's broken or pinned to a bound
                    if is_limiting:
                        violation_count += 1
                        print(f"{row:<6} | {g_val:<14.4f} | {low:<14.4f} | {up:<14.4f} | {status}")
                        print(f"  └─ Description: {opti.debug.g_describe(row)}")
                
                if violation_count == 0:
                    print("No explicit bound violations found at last iteration. Check initial state definitions.")
                        
            except Exception as ex:
                print(f"Failed to process limits even with evaluation: {ex}")
                print("\n=== FALLBACK SIMPLE SWEEP (Printing All Non-Zero Constraints) ===")
                # If everything else fails, this will dump the raw CasADi descriptions of your constraints
                for row in range(opti.g.shape[0]):
                    try:
                        g_val = float(opti.debug.value(opti.g[row]))
                        # Print anything that isn't zero just to see what exists
                        if abs(g_val) > 1e-3:
                            print(f"Row {row} (Value: {g_val:.4f}): {opti.debug.g_describe(row)}")
                    except:
                        pass
            
            print("\n=== CHECKING INDIVIDUAL VARIABLE GUESSES FOR NaNs ===")
            for i in range(self.sim.N_uav):
                x_val = opti.debug.value(opti_variables.x[i])
                u_val = opti.debug.value(opti_variables.u[i])
                Tc_val = opti.debug.value(opti_variables.Tc[i])
                if np.any(np.isnan(x_val)): print(f"❌ Drone {i} states 'x' contain NaNs!")
                if np.any(np.isnan(u_val)): print(f"❌ Drone {i} controls 'u' contain NaNs!")
                if np.any(np.isnan(Tc_val)): print(f"❌ Drone {i} tensions 'Tc' contain NaNs!")
            
            p_pos_val = opti.debug.value(opti_variables.payload_pos)
            p_vel_val = opti.debug.value(opti_variables.payload_vel)
            if np.any(np.isnan(p_pos_val)): print("❌ Payload positions contain NaNs!")
            if np.any(np.isnan(p_vel_val)): print("❌ Payload velocities contain NaNs!")

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
            else:
                # SUBSEQUENT WINDOWS: Match the shifted end-state from the last window
                n = self._prev_sol["n_apply"]
                # Look at the first node after shifting the previous window's planned array
                prev_x_shifted = np.hstack([
                    self._prev_sol["x"][i][:, n:], 
                    np.tile(self._prev_sol["x"][i][:, -1:], (1, n))
                ])
                x0 = prev_x_shifted[:, 0]

            # Enforce the hard state match at index 0
            opti.subject_to(opt_variables.x[i][:, 0] == x0)
        # for i in range(self.sim.N_uav):
        #     vx, vy, vz = drones[i].v

        #     V0 = np.linalg.norm([vx, vy, vz])
        #     chi0 = np.arctan2(vy, vx)
        #     gamma0 = np.arctan2(vz, np.sqrt(vx**2 + vy**2))

        #     # Protect against exact 0/0 singular derivatives at initialization
        #     if V0 < 1e-4:
        #         V0 = 1e-4

        #     x0 = ca.vertcat(
        #         V0,
        #         gamma0,
        #         chi0,
        #         drones[i].position[0],
        #         drones[i].position[1],
        #         drones[i].position[2]
        #     )
        #     # The drone must start exactly where it physically is right now
        #     opti.subject_to(opt_variables.x[i][:, 0] == x0)

        # ── 2. PAYLOAD INITIAL CONDITIONS (FULLY SOFTENED) ─────────────────────
        # REMOVED: Strict epsilon bounds on payload position/velocity at k=0 are deleted.
        # This prevents initial measurement noise or snap-transients from breaking feasibility.

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

    def add_payload_tracking_objective(self, opti: ca.Opti, opt_variables: OptiVariables,
                                       ref: np.ndarray) -> None:

        cost = ca.sumsqr(opt_variables.payload_pos - ref)


        # # Cruise weights for UAV state terms.
        # W_gamma = 1.0   # flight-path angle penalty weight [rad²]
        # W_chi   = 1.0   # heading-rate penalty weight [dimensionless]

        # for i in range(self.sim.N_uav):
        #     # Formation tracking: pull each UAV toward its designated offset from the
        #     # payload. Without this, the optimizer is free to move drones anywhere as
        #     # long as the cable-length constraint is satisfied, causing curved paths.

        #     # Level flight: penalise non-zero flight-path angle (gamma).
        #     # Keep each UAV approximately level during cruise
        #     cost += W_gamma * ca.sumsqr(opt_variables.x[i][1, :])

        #     # Heading-rate penalty: penalise chi changes between consecutive nodes to
        #     # suppress heading oscillations. sin(Δchi) handles ±π wrap-around; for
        #     # the small per-step changes expected in cruise sin(Δchi) ≈ Δchi.
        #     dchi = ca.sin(opt_variables.x[i][2, 1:] - opt_variables.x[i][2, :-1])
        #     cost += W_chi * ca.sumsqr(dchi)

        # # Control-rate penalty: discourage sharp input gradients between nodes. The
        # # three channels (thrust, alpha, bank) live on very different scales, so each
        # # rate is normalized by its admissible range before being weighted, making
        # # W_du a single dimensionless knob traded against tracking error.
        # W_du = 1e-4
        # du_scale = ca.vertcat(self.lim.T_max - self.lim.T_min,
        #                       self.lim.alpha_max - self.lim.alpha_min,
        #                       2 * self.lim.mu_max)
        # for i in range(self.sim.N_uav):
        #     du = (opt_variables.u[i][:, 1:] - opt_variables.u[i][:, :-1]) / du_scale
        #     cost += W_du * ca.sumsqr(du)
        opti.minimize(cost)

    def custom_constraint_placeholder(self, opti: ca.Opti) -> None:
        # Replace this with actual constraints for specific mission phases as needed
        return None

    def constraint_placeholder(self, opti: ca.Opti) -> None:
        # Replace this with actual constraints for specific mission phases as needed
        return None
    
    def rk4_step(self, xs_k, us_k, Tcs_k, pL_k, vL_k):
        dt = self.sim.dt

        def f(xs, pL, vL):
            return coupled_rhs(xs, us_k, Tcs_k, pL, vL, self.veh, self.sim)

        k1_xs, k1_pL, k1_vL = f(xs_k, pL_k, vL_k)

        xs_m  = [xs_k[i] + 0.5*dt*k1_xs[i] for i in range(self.sim.N_uav)]
        k2_xs, k2_pL, k2_vL = f(xs_m, pL_k + 0.5*dt*k1_pL,
                                       vL_k + 0.5*dt*k1_vL)

        xs_m  = [xs_k[i] + 0.5*dt*k2_xs[i] for i in range(self.sim.N_uav)]
        k3_xs, k3_pL, k3_vL = f(xs_m, pL_k + 0.5*dt*k2_pL,
                                       vL_k + 0.5*dt*k2_vL)

        xs_m  = [xs_k[i] + dt*k3_xs[i] for i in range(self.sim.N_uav)]
        k4_xs, k4_pL, k4_vL = f(xs_m, pL_k + dt*k3_pL,
                                       vL_k + dt*k3_vL)

        xs_next = [xs_k[i] + (dt/6.)*(k1_xs[i] + 2*k2_xs[i] +
                                       2*k3_xs[i] + k4_xs[i])
                   for i in range(self.sim.N_uav)]
        pL_next = pL_k + (dt/6.)*(k1_pL + 2*k2_pL + 2*k3_pL + k4_pL)
        vL_next = vL_k + (dt/6.)*(k1_vL + 2*k2_vL + 2*k3_vL + k4_vL)

        return xs_next, pL_next, vL_next