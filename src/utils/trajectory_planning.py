# ── Problem parameters ────────────────────────────────────────────────────────

from dataclasses import dataclass, replace
import numpy as np
import casadi as ca


@dataclass
class SimParams:
    """Simulation / discretization parameters."""
    N_uav:   int   = 3      # number of UAVs (change freely)
    N:       int   = 500    # number of timesteps (total time = N*dt = 10 s)
    dt:      float = 0.05   # timestep size [s] — coarser than 0.01 s for better feasibility
    N_h:     int   = 20     # receding-horizon window length [nodes] (20*0.05 = 1 s horizon)
    N_apply: int   = 5      # nodes committed per window [nodes] (5*0.05 = 0.25 s)


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

def _equilibrium_forward_offset(veh: VehicleParams, lim: StateLimits,
                                lateral_offset: float) -> float:
    """Forward offset [m] that gives equal positive cable tensions at cruise.

    At cruise the payload is in force equilibrium: vertical cable tension balances
    gravity, forward cable tension balances aerodynamic drag. For equal tensions
    across the three cables the required angle satisfies:

        (v0 + 2·vs) / (3·f) = m_L·g / F_drag

    where v0 = sqrt(L²−f²), vs = sqrt(L²−f²−l²), and l = lateral_offset.
    The solution is found by bisection.  A forward_offset=4 m yields a NEGATIVE
    side-cable tension for this payload, which forces IPOPT to search infeasible
    space — the actual equilibrium value is near 2.7–2.8 m.
    """
    L = veh.cable_len
    l = lateral_offset
    F_drag = 0.5 * veh.rho * veh.CD0_payload * veh.S_payload * lim.V_cruise ** 2
    ratio  = veh.m_L * veh.g / F_drag

    def residual(f):
        v0 = np.sqrt(max(L**2 - f**2, 0.0))
        vs = np.sqrt(max(L**2 - f**2 - l**2, 0.0))
        return (v0 + 2.0 * vs) / (3.0 * f) - ratio

    lo, hi = 1e-3, np.sqrt(max(L**2 - l**2, 0.0)) - 1e-3
    for _ in range(60):          # 60 bisection steps → < 1e-15 m accuracy
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cruise_offsets(veh: VehicleParams, lim: StateLimits, heading: float,
                   lateral_offset: float = 6.0,
                   forward_offset: float = None) -> list:
    """Return the three UAV-from-payload position offsets for straight cruise.

    UAV 0 is centred, UAVs 1/2 are left and right.  All offsets satisfy
    |offset| == cable_len exactly.

    If forward_offset is None (default) it is computed automatically as the
    equilibrium value that lets all three cables carry positive tension at cruise
    (see _equilibrium_forward_offset).  Pass an explicit float only to override.
    """
    if forward_offset is None:
        forward_offset = _equilibrium_forward_offset(veh, lim, lateral_offset)
    assert forward_offset**2 + lateral_offset**2 < veh.cable_len**2, (
        "forward_offset and lateral_offset are too large for the cable length")
    forward = np.array([np.cos(heading), np.sin(heading), 0.0])
    right   = np.array([-np.sin(heading), np.cos(heading), 0.0])
    up      = np.array([0.0, 0.0, 1.0])
    v0  = np.sqrt(veh.cable_len**2 - forward_offset**2)
    vs  = np.sqrt(veh.cable_len**2 - forward_offset**2 - lateral_offset**2)
    return [
        forward_offset * forward + v0 * up,
        forward_offset * forward - lateral_offset * right + vs * up,
        forward_offset * forward + lateral_offset * right + vs * up,
    ]


def consistent_ic(sim: SimParams, veh: VehicleParams, lim: StateLimits,
                  payload_pos0=(0.0, 0.0, 100.0), heading=np.pi / 2,
                  lateral_offset: float = 6.0, forward_offset: float = None):
    """Steady level-cruise initial state for the whole system.

    All bodies (UAVs + payload) translate together at V_cruise in the horizontal
    `heading` direction, so every UAV-payload relative velocity is zero and the
    cables are taut.  UAVs are placed ahead of the payload so the cables can pull
    the payload forward against aerodynamic drag.  The forward offset defaults to
    the equilibrium value (see cruise_offsets / _equilibrium_forward_offset).

    Returns (uav_states0, payload_pos0, payload_vel0) for build_nlp.
    """
    assert sim.N_uav == 3, "consistent_ic currently expects exactly 3 UAVs"
    payload_pos0 = np.asarray(payload_pos0, dtype=float)
    assert 2 * lateral_offset >= lim.d_min, (
        "left/right UAVs must be separated by at least d_min")

    V = lim.V_cruise
    payload_vel0 = V * np.array([np.cos(heading), np.sin(heading), 0.0])

    offsets = cruise_offsets(veh, lim, heading, lateral_offset=lateral_offset,
                             forward_offset=forward_offset)
    uav_states0 = []
    for offset in offsets:
        p = payload_pos0 + offset
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
              x0=None, print_level=0, u0=None, Tc0=None,
              uav_offsets=None, heading_ref=None, v_ref=None):
    """Build a trajectory-tracking NLP over sim.N timesteps.

    If `x0` is None the initial state is left free. If
    `x0 = (uav_states0, payload_pos0, payload_vel0)` is given, the state at
    node 0 is pinned (used to stitch receding-horizon windows).

    Optional cruise-mode cost terms (all are soft penalties):
      uav_offsets  — list of (3,) arrays: desired UAV position relative to payload
                     (world frame). Adds a formation-tracking term for each UAV.
      heading_ref  — float [rad]: reference cruise heading. Adds a heading
                     alignment penalty using sin(chi - heading_ref) so it wraps
                     naturally across ±π without explicit angle-wrapping logic.
      v_ref        — (3,) array [m/s]: desired payload velocity. Adds a payload
                     velocity-tracking term.

    These three parameters should be computed once via cruise_offsets() and
    passed unchanged to every call of build_nlp inside solve_rhc.
    """
    assert ref.shape == (3, sim.N), f"ref must be (3, {sim.N}), got {ref.shape}"
    opti = ca.Opti()

    # Decision variables: one set per UAV, plus the payload.
    x           = [opti.variable(6, sim.N) for _ in range(sim.N_uav)]
    u           = [opti.variable(3, sim.N) for _ in range(sim.N_uav)]
    Tc          = [opti.variable(1, sim.N) for _ in range(sim.N_uav)]
    payload_pos = opti.variable(3, sim.N)
    payload_vel = opti.variable(3, sim.N)

    # ── Cost ──────────────────────────────────────────────────────────────────
    # Payload position tracking (primary objective).
    cost = ca.sumsqr(payload_pos - ref)


    # Cruise weights for UAV state terms.
    W_gamma = 1.0   # flight-path angle penalty weight [rad²]
    W_chi   = 1.0   # heading-rate penalty weight [dimensionless]

    for i in range(sim.N_uav):
        # Formation tracking: pull each UAV toward its designated offset from the
        # payload. Without this, the optimizer is free to move drones anywhere as
        # long as the cable-length constraint is satisfied, causing curved paths.

        # Level flight: penalise non-zero flight-path angle (gamma).
        # Keep each UAV approximately level during cruise
        cost += W_gamma * ca.sumsqr(x[i][1, :])

        # Heading-rate penalty: penalise chi changes between consecutive nodes to
        # suppress heading oscillations. sin(Δchi) handles ±π wrap-around; for
        # the small per-step changes expected in cruise sin(Δchi) ≈ Δchi.
        dchi = ca.sin(x[i][2, 1:] - x[i][2, :-1])
        cost += W_chi * ca.sumsqr(dchi)

    # Control-rate penalty: discourage sharp input gradients between nodes. The
    # three channels (thrust, alpha, bank) live on very different scales, so each
    # rate is normalized by its admissible range before being weighted, making
    # W_du a single dimensionless knob traded against tracking error.
    W_du = 1e-4
    du_scale = ca.vertcat(lim.T_max - lim.T_min,
                          lim.alpha_max - lim.alpha_min,
                          2 * lim.mu_max)
    for i in range(sim.N_uav):
        du = (u[i][:, 1:] - u[i][:, :-1]) / du_scale
        cost += W_du * ca.sumsqr(du)
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

        # Taut cables: keep each UAV within cable_tol of cable_len.
        # IMPORTANT: the tolerance is on cable LENGTH, not squared distance.
        # Applying eps=0.1 to d² gives only ±4 mm for a 12.5 m cable (way too tight).
        # Instead bound d² by (L±tol)² so the tolerance band is ±cable_tol metres.
        L_min = veh.cable_len - veh.cable_tol   # 12.4 m
        L_max = veh.cable_len + veh.cable_tol   # 12.6 m
        for k in range(sim.N):
            d = x[i][3:6, k] - payload_pos[:, k]
            opti.subject_to(opti.bounded(L_min**2,
                                         ca.dot(d, d),
                                         L_max**2))

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
              N_h=None, N_apply=None, print_level=0, spawn_ic=None,
              uav_offsets=None, heading_ref=None, v_ref=None):
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
    # Extend the reference by N_h steps (constant-velocity extrapolation) so
    # the last windows always see a full, consistent horizon. The tail is never
    # committed to the output — it only widens the optimizer's lookahead.
    last_step = (ref[:, -1] - ref[:, -2]) if ref.shape[1] >= 2 else np.zeros(3)
    ref_ext = np.hstack([ref, ref[:, -1:] + last_step[:, None] * np.arange(1, N_h + 1)])
    N_ext = N + N_h

    win = replace(sim, N=N_h)                       # window-sized SimParams

    X  = [np.zeros((6, N)) for _ in range(nu)]
    U  = [np.zeros((3, N)) for _ in range(nu)]
    TC = [np.zeros((1, N)) for _ in range(nu)]
    PL = np.zeros((3, N))
    VL = np.zeros((3, N))

    # ── Cruise trim values for warm-starting the first window ────────────────
    # Without explicit control/tension initial guesses IPOPT starts from zero
    # thrust, which is a free-fall configuration far from any feasible cruise
    # trajectory and forces the solver to spend most of its budget on feasibility
    # restoration before it can optimise.
    _F_drag = 0.5 * veh.rho * veh.CD0_payload * veh.S_payload * lim.V_cruise**2
    # Each cable carries roughly 1/N_uav of the combined gravity+drag load.
    _Tc_trim = float(np.sqrt((veh.m_L * veh.g / nu)**2 + (_F_drag / nu)**2))
    # Approximate cable geometry fractions at equilibrium.
    _f_eq = _equilibrium_forward_offset(veh, lim, 6.0)  # centre cable
    _f_frac = _f_eq / veh.cable_len
    _v_frac = np.sqrt(max(1.0 - _f_frac**2, 0.0))
    # UAV lift must compensate gravity AND the downward cable pull.
    _q = 0.5 * veh.rho * lim.V_cruise**2 * veh.S
    _CL_trim = (veh.m * veh.g + _Tc_trim * _v_frac) / _q
    _D_trim  = _q * (veh.CD0 + _CL_trim**2 / (np.pi * veh.AR * veh.e))
    _T_trim  = float(np.clip(_D_trim + _Tc_trim * _f_frac, lim.T_min, lim.T_max))
    _alpha_trim = float(np.clip((_CL_trim - veh.CL0) / veh.CLa,
                                lim.alpha_min, lim.alpha_max))

    x0, u0, Tc0, prev, prev_commit, k = spawn_ic, None, None, None, None, 0
    while k < N - 1:
        idx = np.minimum(np.arange(k, k + N_h), N_ext - 1)
        nlp = build_nlp(ref_ext[:, idx], win, veh, lim, x0=x0, u0=u0, Tc0=Tc0,
                        print_level=print_level,
                        uav_offsets=uav_offsets, heading_ref=heading_ref,
                        v_ref=v_ref)
        opti = nlp.opti

        if prev is None:
            # First window: warm-start state from spawn_ic, controls from trim.
            xs0, _, vL0 = spawn_ic if spawn_ic is not None else consistent_ic(
                win, veh, lim, payload_pos0=ref[:, 0])
            # Use the reference as payload position guess so the optimizer starts
            # near the goal instead of being stuck at the initial position.
            pL_guess = ref[:, idx]
            vL_guess = (np.tile(np.asarray(v_ref)[:, None], (1, N_h))
                        if v_ref is not None
                        else np.tile(vL0[:, None], (1, N_h)))
            opti.set_initial(nlp.payload_pos, pL_guess)
            opti.set_initial(nlp.payload_vel, vL_guess)
            for i in range(nu):
                # UAV positions: track the reference + formation offset when known.
                if uav_offsets is not None:
                    pos_guess = pL_guess + np.asarray(uav_offsets[i])[:, None]
                else:
                    pos_guess = np.tile(xs0[i][3:6, None], (1, N_h))
                x_guess = np.zeros((6, N_h))
                x_guess[0, :] = lim.V_cruise
                x_guess[1, :] = 0.0
                x_guess[2, :] = (heading_ref if heading_ref is not None
                                 else xs0[i][2])
                x_guess[3:6, :] = pos_guess
                opti.set_initial(nlp.x[i], x_guess)
                # Controls: cruise trim (non-zero thrust and cable tension).
                opti.set_initial(nlp.u[i],
                                 np.tile([_T_trim, _alpha_trim, 0.0],
                                         (N_h, 1)).T)
                opti.set_initial(nlp.Tc[i], np.full((1, N_h), _Tc_trim))
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

    # Payload reference: straight cruise in +y (east) direction.
    ref = forward_cruise(sim)

    # Cruise heading: π/2 = east (+p_e direction, matching forward_cruise).
    heading = np.pi / 2
    lat_off = 6.0  # lateral spacing between side UAVs and centreline [m]

    # Formation offsets: equilibrium forward offset is computed automatically.
    # Passing these to solve_rhc enables per-UAV formation and heading cost terms.
    offsets = cruise_offsets(veh, lim, heading, lateral_offset=lat_off)

    # Pin t=0 to the physically consistent cruise formation.
    spawn_ic = consistent_ic(sim, veh, lim, payload_pos0=ref[:, 0],
                             heading=heading, lateral_offset=lat_off)

    # Payload cruise velocity reference (used inside build_nlp velocity cost).
    v_ref = np.array([0.0, lim.V_cruise, 0.0])

    sol = solve_rhc(ref, sim, veh, lim, print_level=0, spawn_ic=spawn_ic, # change spawn_ic to None for free initial state
                    uav_offsets=offsets, heading_ref=heading, v_ref=v_ref)

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
