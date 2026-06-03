# ── Problem parameters ────────────────────────────────────────────────────────

from dataclasses import dataclass, replace
import numpy as np
import casadi as ca


@dataclass
class SimParams:
    """Simulation / discretization parameters."""
    N_uav:   int   = 3      # number of UAVs (change freely)
    N:       int   = 200   # number of timesteps (total time = (N-1)*dt)
    dt:      float = 0.005   # timestep size [s]
    N_h:     int   = 30     # receding-horizon window length [nodes]
    N_apply: int   = 10     # nodes committed per window [nodes]


@dataclass
class VehicleParams:
    """Vehicle / aero parameters and payload + cable parameters."""
    # Vehicle / aero
    m:           float = 6.8      # mass of each UAV [kg]
    g:           float = 9.81     # gravity [m/s^2]
    rho:         float = 1.225    # air density [kg/m^3]
    S:           float = 1.4      # wing reference area [m^2]
    CL0:         float = 0.8
    AR:          float = 6.5
    e:           float = 0.85
    CLa:         float = 4.6      # lift-curve slope [1/rad]
    CD0:         float = 0.02
    CD0_payload: float = 1.07
    S_payload:   float = 0.56

    # Payload / cable
    m_L:       float = 60    # payload mass [kg]
    cable_len: float = 12.5    # cable length L_c [m] (nominal)
    cable_tol: float = 0.1     # allowed half-band on the chord length [m]


@dataclass
class StateLimits:
    """State / control limits."""
    V_min:     float = 10.0
    V_max:     float = 30.0
    gam_max:   float = np.deg2rad(45.0)
    T_min:     float = 0.0
    T_max:     float = 140         # maybe change to vtol thrust
    P_max:     float =  6000 # max propulsive power per UAV [W]
    alpha_min: float = np.deg2rad(-15.0)
    alpha_max: float = np.deg2rad(10.0)
    mu_max:    float = np.deg2rad(35.0)
    d_min:     float = 2.0              # min distance between any two UAVs [m]
    V_cruise:  float = 20.0
    Tc_max:    float = 500.0  # max cable tension [N]



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


# ── Payload reference trajectories ────────────────────────────────────────────
# Each builder returns the desired payload position as a (3, N) matrix on the
# sim time grid. To use a different trajectory, just pass another builder's
# output as `ref` to build_nlp. Add new shapes here following the same pattern.

def straight_climb(sim: SimParams, h0=100.0, climb=3.33) -> np.ndarray:
    """Climb straight up from height h0 at `climb` m/s."""
    t = np.arange(sim.N) * sim.dt
    return np.vstack([np.zeros_like(t), np.zeros_like(t), h0 + climb * t])


def forward_cruise(sim: SimParams, h0=100.0, speed=20.0) -> np.ndarray:
    """Fly forward (+y) at constant height h0 and `speed` m/s."""
    t = np.arange(sim.N) * sim.dt
    return np.vstack([np.zeros_like(t), speed * t, np.full_like(t, h0)])


# ── Initial condition ─────────────────────────────────────────────────────────

def consistent_ic(sim: SimParams, veh: VehicleParams, lim: StateLimits,
                  payload_pos0=(0.0, 0.0, 100.0), heading=np.pi / 2,
                  r_form=None):
    """Steady level-cruise initial state for the whole system.

    All bodies (UAVs + payload) translate together at V_cruise in the horizontal
    `heading` direction, so every UAV-payload relative velocity is zero and the
    cables start out exactly taut. The UAVs sit on a horizontal circle of radius
    r_form, hanging `depth` above the payload so that the cable is at length
    cable_len: sqrt(r_form^2 + depth^2) == cable_len.

    Returns (uav_states0, payload_pos0, payload_vel0), the inputs build_nlp wants.
    """
    payload_pos0 = np.asarray(payload_pos0, dtype=float)
    if r_form is None:
        r_form = 0.6 * veh.cable_len
    assert r_form < veh.cable_len, "formation radius must be < cable length"
    depth = np.sqrt(veh.cable_len**2 - r_form**2)

    # Shared cruise velocity (level flight, gamma = 0).
    V = lim.V_cruise
    payload_vel0 = V * np.array([np.cos(heading), np.sin(heading), 0.0])

    # Spread the UAVs evenly around the circle, each above the payload.
    angles = np.linspace(0, 2 * np.pi, sim.N_uav, endpoint=False)
    uav_states0 = []
    for a in angles:
        p = payload_pos0 + np.array([r_form * np.cos(a),
                                     r_form * np.sin(a), depth])
        # state = (V, gamma, chi, p_n, p_e, h)
        uav_states0.append(np.array([V, 0.0, heading, p[0], p[1], p[2]]))
    return uav_states0, payload_pos0, payload_vel0


@dataclass
class NLP:
    """Handles for a built trajectory NLP."""
    opti:        ca.Opti
    x:           list   # per-UAV state   (6, N)
    u:           list   # per-UAV control (3, N)
    Tc:          list   # per-UAV cable tension (1, N)
    payload_pos: ca.MX  # payload position (3, N)
    payload_vel: ca.MX  # payload velocity (3, N)


def build_nlp(ref, sim: SimParams, veh: VehicleParams, lim: StateLimits,
              x0=None, print_level=0, u0=None, Tc0=None):
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
    assert ref.shape == (3, sim.N), f"ref must be (3, {sim.N}), got {ref.shape}"
    opti = ca.Opti()

    # Decision variables: one set per UAV, plus the payload.
    x           = [opti.variable(6, sim.N) for _ in range(sim.N_uav)]
    u           = [opti.variable(3, sim.N) for _ in range(sim.N_uav)]
    Tc          = [opti.variable(1, sim.N) for _ in range(sim.N_uav)]
    payload_pos = opti.variable(3, sim.N)
    payload_vel = opti.variable(3, sim.N)

    # Cost: track the reference with the payload, plus a thrust-effort penalty.
    R_T = 1e-3  # thrust effort weight
    cost = ca.sumsqr(payload_pos - ref)
    # for i in range(sim.N_uav):
    #     cost += R_T * ca.sumsqr(u[i][0, :])  # thrust effort
    opti.minimize(cost)

    # Initial condition: free on the first window (x0 None), pinned otherwise.
    if x0 is not None:
        uav_states0, payload_pos0, payload_vel0 = x0
        for i in range(sim.N_uav):
            opti.subject_to(x[i][:, 0] == uav_states0[i])
        opti.subject_to(payload_pos[:, 0] == payload_pos0)
        opti.subject_to(payload_vel[:, 0] == payload_vel0)

    # Control continuity across windows. Under backward Euler u[:,0] enters no
    # dynamics constraint (the step into node k uses u[:,k]), so it is otherwise
    # a free variable; pinning it to the previous window's continuation control
    # keeps the committed control history continuous at the window seams.
    if u0 is not None:
        for i in range(sim.N_uav):
            opti.subject_to(u[i][:, 0] == u0[i])

    if Tc0 is not None:
        for i in range(sim.N_uav):
            opti.subject_to(Tc[i][:, 0] == Tc0[i])

    # Dynamics: backward (implicit) Euler. The increment over each step uses the
    # derivative evaluated at the *next* node: x_{k+1} = x_k + dt * f(x_{k+1}).
    for k in range(sim.N - 1):
        xs_dot, pos_dot, vel_dot = coupled_rhs(
            [x[i][:, k + 1] for i in range(sim.N_uav)],
            [u[i][:, k + 1] for i in range(sim.N_uav)],
            [Tc[i][:, k + 1] for i in range(sim.N_uav)],
            payload_pos[:, k + 1], payload_vel[:, k + 1], veh, sim)
        for i in range(sim.N_uav):
            opti.subject_to(x[i][:, k + 1] == x[i][:, k] + sim.dt * xs_dot[i])
        opti.subject_to(payload_pos[:, k + 1] == payload_pos[:, k] + sim.dt * pos_dot)
        opti.subject_to(payload_vel[:, k + 1] == payload_vel[:, k] + sim.dt * vel_dot)

    # Control limits: thrust, propulsive power, angle of attack, bank angle.
    for i in range(sim.N_uav):
        opti.subject_to(opti.bounded(lim.T_min,     u[i][0, :], lim.T_max))
        opti.subject_to(u[i][0, :] * x[i][0, :] <= lim.P_max)
        opti.subject_to(opti.bounded(lim.alpha_min, u[i][1, :], lim.alpha_max))
        opti.subject_to(opti.bounded(-lim.mu_max,   u[i][2, :], lim.mu_max))

        # State limits: airspeed and flight-path angle.
        opti.subject_to(opti.bounded(lim.V_min,     x[i][0, :], lim.V_max))
        opti.subject_to(opti.bounded(-lim.gam_max,  x[i][1, :], lim.gam_max))

        # Cable tension: cables can only pull (Tc >= 0) and have a max rating.
        opti.subject_to(opti.bounded(0.0, Tc[i], lim.Tc_max))

        # Taut cables: keep each UAV cable_len from the payload, within a small fixed
        # tolerance band on the squared distance (hard equality is too stiff to solve).
        eps = 1e-1
        for k in range(sim.N):
            d = x[i][3:6, k] - payload_pos[:, k]
            opti.subject_to(opti.bounded(veh.cable_len**2 - eps,
                                         ca.dot(d, d),
                                         veh.cable_len**2 + eps))

        # Collision avoidance: keep every pair of UAVs at least d_min apart.
        for j in range(i + 1, sim.N_uav):
            for k in range(sim.N):
                d = x[i][3:6, k] - x[j][3:6, k]
                opti.subject_to(ca.dot(d, d) >= lim.d_min**2)

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




# ── Receding-horizon solve ────────────────────────────────────────────────────

@dataclass
class Solution:
    """Stitched closed-loop trajectory over the full sim horizon."""
    x:           list     # per-UAV state   (6, N)
    u:           list     # per-UAV control (3, N)
    Tc:          list     # per-UAV cable tension (1, N)
    payload_pos: np.ndarray  # (3, N)
    payload_vel: np.ndarray  # (3, N)


def solve_rhc(ref, sim: SimParams, veh: VehicleParams, lim: StateLimits,
              N_h=None, N_apply=None, print_level=0, spawn_ic=None):
    """Receding-horizon solve of the payload-tracking problem.

    Slide a window of `N_h` nodes along the reference. Each window is a small NLP
    (built with `build_nlp`); we commit its first `N_apply` nodes to the output,
    then start the next window from the first uncommitted state so the stitched
    trajectory stays dynamically continuous.

    By default the FIRST window has a free initial state (the optimizer picks the
    start). Pass `spawn_ic=(uav_states0, payload_pos0, payload_vel0)` (e.g. from
    `consistent_ic`) to instead pin the UAVs to a fixed start formation around the
    payload. Every later window is pinned to the previous window's continuation
    state. Each window warm-starts from the previous solution shifted forward.
    """
    assert ref.shape == (3, sim.N), f"ref must be (3, {sim.N}), got {ref.shape}"
    N_h = sim.N_h if N_h is None else N_h
    N_apply = sim.N_apply if N_apply is None else N_apply
    assert N_h >= 2, "N_h must be at least 2 nodes"
    assert N_apply >= 1, "N_apply must be at least 1 node"
    assert N_apply <= N_h, "N_apply cannot exceed N_h"
    N, nu = sim.N, sim.N_uav
    win = replace(sim, N=N_h)                       # window-sized SimParams

    X  = [np.zeros((6, N)) for _ in range(nu)]
    U  = [np.zeros((3, N)) for _ in range(nu)]
    TC = [np.zeros((1, N)) for _ in range(nu)]
    PL = np.zeros((3, N))
    VL = np.zeros((3, N))

    x0, u0, Tc0, prev, prev_commit, k = spawn_ic, None, None, None, None, 0
    while k < N - 1:
        # Window reference; near the end, hold the final node (clamp the index).
        idx = np.minimum(np.arange(k, k + N_h), N - 1)
        nlp = build_nlp(ref[:, idx], win, veh, lim, x0=x0, u0=u0, Tc0=Tc0,
                        print_level=print_level)
        opti = nlp.opti

        # Warm start: shifted previous solution, else broadcast the steady IC
        # (the pinned spawn formation if one was given, otherwise a default one).
        if prev is None:
            xs0, pL0, vL0 = spawn_ic if spawn_ic is not None else consistent_ic(
                win, veh, lim, payload_pos0=ref[:, 0])
            for i in range(nu):
                opti.set_initial(nlp.x[i], np.tile(xs0[i][:, None], (1, N_h)))
            opti.set_initial(nlp.payload_pos, np.tile(pL0[:, None], (1, N_h)))
            opti.set_initial(nlp.payload_vel, np.tile(vL0[:, None], (1, N_h)))
        else:
            def shift(a):  # drop the committed nodes, repeat the last column
                return np.hstack([a[:, prev_commit:],
                                  np.tile(a[:, -1:], (1, prev_commit))])
            for i in range(nu):
                opti.set_initial(nlp.x[i],  shift(prev.x[i]))
                opti.set_initial(nlp.u[i],  shift(prev.u[i]))
                opti.set_initial(nlp.Tc[i], shift(prev.Tc[i]))
            opti.set_initial(nlp.payload_pos, shift(prev.payload_pos))
            opti.set_initial(nlp.payload_vel, shift(prev.payload_vel))

        sol = opti.solve()
        w = Solution(
            x=[np.asarray(sol.value(nlp.x[i])).reshape(6, N_h) for i in range(nu)],
            u=[np.asarray(sol.value(nlp.u[i])).reshape(3, N_h) for i in range(nu)],
            Tc=[np.asarray(sol.value(nlp.Tc[i])).reshape(1, N_h) for i in range(nu)],
            payload_pos=np.asarray(sol.value(nlp.payload_pos)).reshape(3, N_h),
            payload_vel=np.asarray(sol.value(nlp.payload_vel)).reshape(3, N_h))

        # Commit the first n_commit nodes. Keep one uncommitted node inside the
        # solved window so the next horizon has a dynamically consistent start.
        n_commit = min(N_apply, N_h - 1, N - 1 - k)
        for i in range(nu):
            X[i][:, k:k + n_commit]  = w.x[i][:, :n_commit]
            U[i][:, k:k + n_commit]  = w.u[i][:, :n_commit]
            TC[i][:, k:k + n_commit] = w.Tc[i][:, :n_commit]
        PL[:, k:k + n_commit] = w.payload_pos[:, :n_commit]
        VL[:, k:k + n_commit] = w.payload_vel[:, :n_commit]

        # Next window starts from the first uncommitted node of this window:
        # its state, control, and tension all carry over so the seam stays
        # dynamically and control-continuous.
        x0 = ([w.x[i][:, n_commit] for i in range(nu)],
              w.payload_pos[:, n_commit], w.payload_vel[:, n_commit])
        u0  = [w.u[i][:, n_commit] for i in range(nu)]
        Tc0 = [w.Tc[i][:, n_commit] for i in range(nu)]
        prev, prev_commit, k = w, n_commit, k + n_commit

    # Fill the final node from the last continuation state.
    uav_last, pL_last, vL_last = x0
    for i in range(nu):
        X[i][:, -1]  = uav_last[i]
        U[i][:, -1]  = U[i][:, -2]
        TC[i][:, -1] = TC[i][:, -2]
    PL[:, -1], VL[:, -1] = pL_last, vL_last

    return Solution(x=X, u=U, Tc=TC, payload_pos=PL, payload_vel=VL)


if __name__ == "__main__":
    """Solve a payload-tracking trajectory, print a summary, and plot it."""
    from plotting import print_distances, plot_solution, animate_solution

    sim = SimParams()
    veh = VehicleParams()
    lim = StateLimits()

    # Payload reference on the sim time grid. Swap for straight_climb(sim) etc.
    ref = forward_cruise(sim)

    # Set spawn_ic to pin the UAVs to a fixed ring formation around the payload
    # at t=0; set it to None to let the optimizer pick the start formation.
    spawn_ic = consistent_ic(sim, veh, lim, payload_pos0=ref[:, 0])

    sol = solve_rhc(ref, sim, veh, lim, print_level=0, spawn_ic=spawn_ic)

    err = np.linalg.norm(sol.payload_pos - ref, axis=0)
    print(f"horizon: {sim.N} nodes, dt={sim.dt}s, UAVs={sim.N_uav}")
    print(f"payload tracking RMSE: {np.sqrt((err**2).mean()):.3f} m  "
          f"(max {err.max():.3f} m)")

    print_distances(sol, sim, veh)

    # show=True pops up interactive windows (rotatable 3D); the PNGs are saved
    # either way. Set show=False if running headless / over plain SSH.
    plot_solution(sol, ref, sim, lim, show=False)

    # GIF of the formation carrying the payload along the reference.
    animate_solution(sol, ref, sim)
