import casadi as ca
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
        self.params = DEFAULT_PARAMS
        self.n_drones = self.params["n_drones"]
        self.horizon_steps = self.params["opti_timepstep_N"]
        self.payload_target = np.zeros(3)
        self.payload_target_t = 0.0
        self.next_traj_step_t = 0.0
        self.sim = SimParams()
        self.veh = VehicleParams()
        self.lim = StateLimits()
        self.ref = np.vstack([
            np.zeros_like(np.arange(DEFAULT_PARAMS["opti_timepstep_N"]) * DEFAULT_PARAMS["dt"]), 
            np.zeros_like(np.arange(DEFAULT_PARAMS["opti_timepstep_N"]) * DEFAULT_PARAMS["dt"]), 
            20.0 + 3.33 * np.arange(DEFAULT_PARAMS["opti_timepstep_N"]) * DEFAULT_PARAMS["dt"]
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
        opti, opti_variables = self.build_generic_optimizer(drones, payload)

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
            opti.set_initial(opti_variables.x[i], np.tile(x0[:, None], (1, self.sim.N)))
        for i in range(self.sim.N_uav):
            # Trim: T*cos(alpha) ≈ D, L ≈ m*g  →  roughly level flight
            alpha_trim = 0.05   # small positive AoA
            CL_trim    = self.veh.CL0 + self.veh.CLa * alpha_trim
            V_trim     = max(np.linalg.norm(drones[i].v), self.lim.V_min)
            q_trim     = 0.5 * self.veh.rho * V_trim**2 * self.veh.S
            T_trim     = q_trim * (self.veh.CD0 + CL_trim**2 /
                           (np.pi * self.veh.AR * self.veh.e))

            u0 = np.array([T_trim, alpha_trim, 0.0])   # [T, alpha, mu]
            opti.set_initial(opti_variables.u[i], np.tile(u0[:, None], (1, self.sim.N)))

        T_cable_0 = (self.veh.m_L * self.veh.g) / self.sim.N_uav
        for i in range(self.sim.N_uav):
            opti.set_initial(opti_variables.Tc[i],
                     np.full((1, self.sim.N), T_cable_0))


        # ── Set specific contraints per phase ─────────────────────────────────────────


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
                for k in range(self.sim.N - 1):
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
                for k in range(self.sim.N):
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
    
    def build_generic_optimizer(self, drones, payload) -> tuple[ca.Opti, OptiVariables]:
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

        return opti, opti_variables

    def add_generic_constraints(
        self,
        opti: ca.Opti,
        opt_variables: OptiVariables,
        drones: list[Drone],
        payload: Payload
    ) -> None:

        N = self.sim.N
        dt = self.sim.dt

        # Initial conditions: fix the initial state of the system to the current state (soft for payload)
        for i in range(self.sim.N_uav):
            opti.subject_to(
                opt_variables.x[i][:, 0] ==
                ca.vertcat(drones[i].position, drones[i].v)
            )

        opti.subject_to(
            ca.sumsqr(opt_variables.payload_pos[:, 0] - payload.position) <= 1e-3
        )
        opti.subject_to(
            ca.sumsqr(opt_variables.payload_vel[:, 0] - payload.v) <= 1e-3
        )

        # Enforce dynamics

        for k in range(N - 1):
            print(f"N={self.sim.N}, dt={self.sim.dt}, N_uav={self.sim.N_uav}")
            print(f"Equality constraints should be: {self.sim.N_uav * 6 * (self.sim.N-1) + 6}")
            xs_next, pL_next, vL_next = self.rk4_step(
            [opt_variables.x[i][:, k] for i in range(self.sim.N_uav)],
            [opt_variables.u[i][:, k] for i in range(self.sim.N_uav)],
            [opt_variables.Tc[i][:, k] for i in range(self.sim.N_uav)],
            opt_variables.payload_pos[:, k],
            opt_variables.payload_vel[:, k],
        )
        for i in range(self.sim.N_uav):
            opti.subject_to(opt_variables.x[i][:, k+1] == xs_next[i])
        opti.subject_to(opt_variables.payload_pos[:, k+1] == pL_next)
        opti.subject_to(opt_variables.payload_vel[:, k+1] == vL_next)

        # Hardware limits

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

        # Soft hardware cable constraint

        eps_cable = 0.3
        w_cable = 1

        cable_cost = 0

        for i in range(self.sim.N_uav):
            for k in range(1, N):   # IMPORTANT: skip k=0

                d = opt_variables.x[i][3:6, k] - opt_variables.payload_pos[:, k]

                dist2 = ca.dot(d, d)
                err = dist2 - self.veh.cable_len**2

                cable_cost += w_cable * err**2 + eps_cable * ca.fabs(err)

        # add to objective later via shared cost
        self._cable_cost = cable_cost

        # Collision avoidance constraint (soft)

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