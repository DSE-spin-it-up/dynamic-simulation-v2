import casadi as ca
from matplotlib.pylab import gamma
import numpy as np

from ..utils.default_params import DEFAULT_PARAMS, VehicleParams, StateLimits, SimParams, OptiVariables

from .drone import Drone
from .payload import Payload
from src.classes import cable, payload
from src.classes import drone

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
        self.sim = SimParams()
        self.veh = VehicleParams()
        self.lim = StateLimits()
        self._window_time = DEFAULT_PARAMS["opti_N_h"] * DEFAULT_PARAMS["opti_dt"]
        self.ref = np.vstack([
            np.zeros_like(np.arange(DEFAULT_PARAMS["opti_N_h"]) * DEFAULT_PARAMS["opti_dt"]), 
            np.zeros_like(np.arange(DEFAULT_PARAMS["opti_N_h"]) * DEFAULT_PARAMS["opti_dt"]), 
            100.0 + 3.33 * np.arange(DEFAULT_PARAMS["opti_N_h"]) * DEFAULT_PARAMS["opti_dt"]
            ])

    def update_mission_phase(self, mission_phase: int):
        self.mission_phase = mission_phase

    def calculate_traj_step(self, t, drones: list[Drone], payload: Payload, mission_phase: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
        # ── Update mission phase ──────────────────────────────────────────────────────
        self.mission_phase = mission_phase

        # ── Build generic optimizer ───────────────────────────────────────────────────
        opti, opti_variables = self.build_optimizer(drones, payload)

        # ────────────── Build objective ───────────────────────────────────────────────
        # Add other objectives depending on mission phase
        self.add_payload_tracking_objective(opti, opti_variables)

        # ────────────── Warm-start ────────────────────────────────────────────────────

        # Warm-start the solver (to be implemented, for now use current states
        for i, drone in enumerate(drones):
            V = np.linalg.norm(drone.v)
            V = max(V, self.lim.V_min)  # never let V=0

            if V > 1e-3:
                gamma = np.arcsin(np.clip(drone.v[2] / V, -1, 1))
                chi   = np.arctan2(drone.v[1], drone.v[0])
            else:
                gamma = 0.0
                chi   = 0.0

            x0 = np.array([V, gamma, chi,
                   drone.position[0],
                   drone.position[1],
                   drone.position[2]])

            # Tile constant initial state across horizon as warm-start
            opti.set_initial(opti_variables.x[i], np.tile(x0[:, None], (1, self.sim.N_h)))
        for i in range(self.sim.N_uav):
            # Trim: T*cos(alpha) ≈ D, L ≈ m*g  →  roughly level flight
            alpha_trim = 0.05   # small positive AoA
            CL_trim    = self.veh.CL0 + self.veh.CLa * alpha_trim
            V_trim     = max(np.linalg.norm(drones[i].v), self.lim.V_min)
            q_trim     = 0.5 * self.veh.rho * V_trim**2 * self.veh.S
            T_trim     = q_trim * (self.veh.CD0 + CL_trim**2 /
                           (np.pi * self.veh.AR * self.veh.e))

            u0 = np.array([T_trim, alpha_trim, 0.0])   # [T, alpha, mu]
            opti.set_initial(opti_variables.u[i], np.tile(u0[:, None], (1, self.sim.N_h)))

        T_cable_0 = (self.veh.m_L * self.veh.g) / self.sim.N_uav
        for i in range(self.sim.N_uav):
            opti.set_initial(opti_variables.Tc[i],
                     np.full((1, self.sim.N_h), T_cable_0))

        # ── Solve ─────────────────────────────────────────────────────────────────────

        opti.solver('ipopt', {'expand': True}, {
        'print_level': 3,
        'max_iter': 1000,
        'nlp_scaling_method': 'gradient-based',   # add this
        'obj_scaling_factor': 1e-3,               # normalise large costs
        'acceptable_tol': 1e-3,
        'acceptable_iter': 15,
        'mu_strategy': 'adaptive',
        'warm_start_init_point': 'yes',           # enable if re-solving
        })
        try:
            sol = opti.solve()
        except Exception as e:
            print("Failed:", e)
    
            # Check each constraint group individually
            print("=== CONSTRAINT DEBUG ===")
    
            # Dynamics residuals
            for i in range(self.sim.N_uav):
                for k in range(self.sim.N_h - 1):
                    res = opti.debug.value(
                        opti_variables.x[i][:, k+1] - opti_variables.x[i][:, k]
                    )
                    if np.any(np.abs(res) > 0.1):
                        print(f"Drone {i}, step {k}: dynamics residual = {res}")

            # Initial condition residuals
            for i in range(self.sim.N_uav):
                ic = opti.debug.value(opti_variables.x[i][:, 0])
                print(f"Drone {i} IC: {ic}")
    
            # Cable lengths
            for i in range(self.sim.N_uav):
                for k in range(self.sim.N_h):
                    d = opti.debug.value(
                        opti_variables.x[i][3:6, k] - opti_variables.payload_pos[:, k]
                    )
                    print(f"Cable {i} step {k} length: {np.linalg.norm(d):.2f} "
                        f"(target {self.veh.cable_len:.2f})")
    
            # Hardware bounds
            for i in range(self.sim.N_uav):
                V   = opti.debug.value(opti_variables.x[i][0, :])
                T   = opti.debug.value(opti_variables.u[i][0, :])
                Tc  = opti.debug.value(opti_variables.Tc[i])
                print(f"Drone {i}: V in [{V.min():.2f}, {V.max():.2f}], "
                f"T in [{T.min():.2f}, {T.max():.2f}], "
                f"Tc in [{Tc.min():.2f}, {Tc.max():.2f}]")

        # ── Extract solution ──────────────────────────────────────────────────────────

        x_sol = [sol.value(opti_variables.x[i]) for i in range(self.sim.N_uav)]
        u_sol = [sol.value(opti_variables.u[i]) for i in range(self.sim.N_uav)]

        return x_sol, u_sol
    
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

        # Initial conditions: fix the initial state of the system to the current state (soft for payload)
        # Drones
        for i in range(self.sim.N_uav):

            vx, vy, vz = drones[i].v

            V0 = np.linalg.norm([vx, vy, vz])
            chi0 = np.arctan2(vy, vx)
            gamma0 = np.arctan2(vz, np.sqrt(vx**2 + vy**2))

            x0 = ca.vertcat(
                V0,
                gamma0,
                chi0,
                drones[i].position[0],
                drones[i].position[1],
               drones[i].position[2]
            )
            # maybe soften this later?
            opti.subject_to(opt_variables.x[i][:, 0] == x0)

        # Payload (soft constraints)

        opti.subject_to(
            ca.sumsqr(opt_variables.payload_pos[:, 0] - payload.position) <= 1e-3
        )
        opti.subject_to(
            ca.sumsqr(opt_variables.payload_vel[:, 0] - payload.v) <= 1e-3
        )

        # Enforce dynamics
        for k in range(self.sim.N_h - 1):
            xs_dot, pos_dot, vel_dot = coupled_rhs(
                [opt_variables.x[i][:, k + 1] for i in range(self.sim.N_uav)],
                [opt_variables.u[i][:, k + 1] for i in range(self.sim.N_uav)],
                [opt_variables.Tc[i][:, k + 1] for i in range(self.sim.N_uav)],
                opt_variables.payload_pos[:, k + 1], opt_variables.payload_vel[:, k + 1], self.veh, self.sim)
            for i in range(self.sim.N_uav):
                opti.subject_to(opt_variables.x[i][:, k + 1] == opt_variables.x[i][:, k] + self.sim.dt * xs_dot[i])
            opti.subject_to(opt_variables.payload_pos[:, k + 1] == opt_variables.payload_pos[:, k] + self.sim.dt * pos_dot)
            opti.subject_to(opt_variables.payload_vel[:, k + 1] == opt_variables.payload_vel[:, k] + self.sim.dt * vel_dot)
        

        # Hardware limits

        for i in range(self.sim.N_uav):

            opti.subject_to(
                opti.bounded(self.lim.T_min, opt_variables.u[i][0, :], self.lim.T_max) # type: ignore
            )

            P = opt_variables.u[i][0, :] * opt_variables.x[i][0, :]
            opti.subject_to(P <= self.lim.P_max)

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

        # Soft hardware cable constraint

        L_min = self.veh.cable_len - self.veh.cable_tol   # 12.4 m
        L_max = self.veh.cable_len + self.veh.cable_tol   # 12.6 m
        for k in range(self.sim.N_h):
            d = opt_variables.x[i][3:6, k] - opt_variables.payload_pos[:, k]
            opti.subject_to(opti.bounded(L_min**2, ca.dot(d, d), L_max**2)) # type: ignore


        # Collision avoidance constraint (soft)

        for j in range(i + 1, self.sim.N_uav):
            for k in range(self.sim.N_h):
                d = opt_variables.x[i][3:6, k] - opt_variables.x[j][3:6, k]
                opti.subject_to(ca.dot(d, d) >= self.lim.d_min**2)

    def add_payload_tracking_objective(self, opti: ca.Opti, opt_variables: OptiVariables) -> None:

        cost = ca.sumsqr(opt_variables.payload_pos - self.ref)


        # Cruise weights for UAV state terms.
        W_gamma = 1.0   # flight-path angle penalty weight [rad²]
        W_chi   = 1.0   # heading-rate penalty weight [dimensionless]

        for i in range(self.sim.N_uav):
            # Formation tracking: pull each UAV toward its designated offset from the
            # payload. Without this, the optimizer is free to move drones anywhere as
            # long as the cable-length constraint is satisfied, causing curved paths.

            # Level flight: penalise non-zero flight-path angle (gamma).
            # Keep each UAV approximately level during cruise
            cost += W_gamma * ca.sumsqr(opt_variables.x[i][1, :])

            # Heading-rate penalty: penalise chi changes between consecutive nodes to
            # suppress heading oscillations. sin(Δchi) handles ±π wrap-around; for
            # the small per-step changes expected in cruise sin(Δchi) ≈ Δchi.
            dchi = ca.sin(opt_variables.x[i][2, 1:] - opt_variables.x[i][2, :-1])
            cost += W_chi * ca.sumsqr(dchi)

        # Control-rate penalty: discourage sharp input gradients between nodes. The
        # three channels (thrust, alpha, bank) live on very different scales, so each
        # rate is normalized by its admissible range before being weighted, making
        # W_du a single dimensionless knob traded against tracking error.
        W_du = 1e-4
        du_scale = ca.vertcat(self.lim.T_max - self.lim.T_min,
                              self.lim.alpha_max - self.lim.alpha_min,
                              2 * self.lim.mu_max)
        for i in range(self.sim.N_uav):
            du = (opt_variables.u[i][:, 1:] - opt_variables.u[i][:, :-1]) / du_scale
            cost += W_du * ca.sumsqr(du)
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