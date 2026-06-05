# ── Problem parameters ────────────────────────────────────────────────────────
# The problem parameter dataclasses and cost-weight plumbing live in utils/config.py
# so the whole run is driven by config.yaml; the reference trajectories and maneuver
# registry live in utils/maneuvers.py. They are re-imported here (and thus also
# importable from this module, which utils/plotting.py relies on).

from dataclasses import dataclass, replace
import time
import numpy as np
import casadi as ca

from utils.config import SimParams, VehicleParams, StateLimits, CostWeights


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


_FORMATION_CACHE = "cruise_formation.npz"


def optimal_cruise_offsets(sim: SimParams, veh: VehicleParams, lim: StateLimits,
                           weights: "CostWeights", heading: float,
                           lateral_offset: float, speed: float, altitude: float,
                           n_short: int = 120, state_dir="states") -> list:
    """Per-UAV cruise offsets discovered by the optimizer, in the heading=0 frame.

    Instead of the hand-derived equal-tension geometry (``cruise_offsets``), this
    runs a short *standalone* level-cruise solve with the formation anchor disabled
    (``W_form = 0``) and a free initial condition, then reads the steady offsets at
    the final (most converged) node. The offsets are de-rotated by the cruise
    heading into the canonical forward=+x frame so ``_formation_target`` can rotate
    them to follow the reference like the analytic ones.

    The result is cached to ``<state_dir>/cruise_formation.npz`` keyed by the
    parameters that affect it; a matching cache is reused so the extra solve is
    paid only once.
    """
    import os
    from dataclasses import replace as _replace
    from utils.maneuvers import MANEUVERS

    sig = np.array([speed, altitude, lateral_offset, heading, veh.cable_len,
                    veh.m, veh.m_L, veh.S, lim.V_cruise, sim.N_uav], dtype=float)
    cache = os.path.join(state_dir, _FORMATION_CACHE)
    if os.path.exists(cache):
        with np.load(cache) as d:
            if d["sig"].shape == sig.shape and np.allclose(d["sig"], sig):
                print(f"using cached optimized cruise formation ({cache})")
                return [d["base"][i] for i in range(d["base"].shape[0])]

    print("discovering optimal cruise formation (short standalone cruise solve)...")
    sim_s = _replace(sim, N=min(int(n_short), sim.N))
    ref_c, _ = MANEUVERS["cruise"](sim_s, altitude, speed)
    w0 = _replace(weights, W_form=0.0)
    v_ref = speed * np.array([np.cos(heading), np.sin(heading), 0.0])
    warm = cruise_offsets(veh, lim, heading, lateral_offset=lateral_offset)
    sol = solve_rhc(ref_c, sim_s, veh, lim, w0, print_level=0,
                    spawn_ic=None, uav_offsets=warm, heading_ref=heading,
                    v_ref=v_ref)

    # Steady offsets at the final node, de-rotated into the heading=0 frame.
    c, s = np.cos(heading), np.sin(heading)
    base = []
    for i in range(sim.N_uav):
        off = sol.x[i][3:6, -1] - sol.payload_pos[:, -1]
        base.append(np.array([c * off[0] + s * off[1],
                              -s * off[0] + c * off[1],
                               off[2]]))

    os.makedirs(state_dir, exist_ok=True)
    np.savez(cache, base=np.stack(base), sig=sig)
    print(f"saved optimized cruise formation to {cache}")
    return base


# ── Formation anchor ──────────────────────────────────────────────────────────
# A phase-gated, soft pull of each UAV toward its nominal cruise offset from the
# payload. The point is to remove the geometric degeneracy that lets one UAV
# slack its cable (carrying little load) while the others lift: the cruise offsets
# are the EQUAL-tension equilibrium (see _equilibrium_forward_offset), so anchoring
# the formation to them makes the three UAVs share the load almost equally.
#
# The gate turns the anchor OFF during free phases (spin_up / climb / descent),
# where the formation is allowed to spin, and ramps it ON across a `transition`
# phase so the drones settle into the balanced cruise formation by the time the
# payload levels off. See build_formation_anchor.

# Phases that pin a definite heading -> anchor the formation (gate = 1).
_ANCHORED_PHASES = {"cruise", "turn"}
# Phases with no preferred horizontal direction -> formation free (gate = 0).
_FREE_PHASES = {"spin_up", "climb", "descent"}


def _smoothstep(x):
    """Smoothstep on a clipped [0, 1] argument."""
    s = np.clip(x, 0.0, 1.0)
    return s * s * (3 - 2 * s)


def _formation_gate(phases, sim: SimParams, ramp_frac: float = 1.0,
                    release_frac: float = 0.0) -> np.ndarray:
    """Per-node anchor weight in [0, 1] built from a maneuver's phase schedule.

    `phases` is the ``[(name, t_start), ...]`` list returned alongside the
    reference by every maneuver. Anchored phases give 1, free phases 0, and a
    ``transition`` phase smoothly ramps (smoothstep) between the levels of the
    phases on either side of it.

    ``ramp_frac`` in (0, 1] sets what fraction of the transition window the ramp
    occupies, anchored to the END of the window: 1.0 ramps across the whole phase
    (default), 0.4 holds the start level for the first 60% then ramps over the last
    40% — i.e. the anchor engages only on the last bit of the curve.

    ``release_frac`` in [0, 1) ramps the anchor *back down to 0 once the cruise
    section is reached* (i.e. after the transition has fully ramped it up). The
    down-ramp (smoothstep) occupies the last ``release_frac`` of the post-transition
    anchored region and reaches 0 at the end of the run, so a transition run ends in
    an un-anchored, natural equilibrium and a chained ``W_form == 0`` cruise can
    start from it cleanly. 0 (default) keeps the plain ramp-and-hold.
    """
    t = np.arange(sim.N) * sim.dt
    T = sim.N * sim.dt
    starts = [s for _, s in phases]
    ends = starts[1:] + [T]
    up_frac  = min(max(ramp_frac, 1e-6), 1.0)
    rel_frac = min(max(release_frac, 0.0), 1.0 - 1e-6)

    def level(name):
        return 1.0 if name in _ANCHORED_PHASES else 0.0

    gate = np.zeros(sim.N)
    for idx, (name, t0) in enumerate(phases):
        mask = (t >= t0) & (t < ends[idx])
        if name == "transition":
            prev_lvl = level(phases[idx - 1][0]) if idx > 0 else 0.0
            next_lvl = level(phases[idx + 1][0]) if idx + 1 < len(phases) else 1.0
            # Ramp only over the last `up_frac` of the window: hold prev_lvl until
            # t_ramp = t_end - up_frac*span, then smoothstep to next_lvl.
            span   = max(ends[idx] - t0, 1e-9)
            t_ramp = ends[idx] - up_frac * span
            gate[mask] = prev_lvl + (next_lvl - prev_lvl) * _smoothstep(
                            (t[mask] - t_ramp) / (up_frac * span))
        else:
            gate[mask] = level(name)

    # Release: once in the cruise section (the first anchored phase after a
    # transition), ramp the anchor back down to 0 over the last `rel_frac` of that
    # region, reaching 0 at the end of the run.
    if rel_frac > 0.0:
        anchored_after_transition = next(
            (i for i in range(1, len(phases))
             if phases[i][0] in _ANCHORED_PHASES
             and phases[i - 1][0] == "transition"),
            None)
        if anchored_after_transition is not None:
            t_cruise = phases[anchored_after_transition][1]
            t_rel    = T - rel_frac * (T - t_cruise)
            down = 1.0 - _smoothstep((t - t_rel) / max(rel_frac * (T - t_cruise), 1e-9))
            gate[t >= t_cruise] *= down[t >= t_cruise]
    return gate


def _formation_target(ref, sim: SimParams, veh: VehicleParams, lim: StateLimits,
                      heading: float, lateral_offset: float,
                      base_offsets=None) -> list:
    """Per-UAV target offset-from-payload over the horizon, shape (3, N) each.

    The cruise offsets are rotated about the vertical axis to follow the
    reference's local heading at each node, so the anchor stays valid through
    turns. Where the payload has no horizontal velocity (e.g. a vertical climb) the
    heading is undefined, but the gate is 0 there so the target is irrelevant; we
    fall back to `heading` for those nodes.

    By default the base offsets are the analytic equal-tension geometry
    (``cruise_offsets``). Pass ``base_offsets`` (heading=0 frame, e.g. from
    ``optimal_cruise_offsets``) to anchor to the optimizer's own cruise formation
    instead.
    """
    base = (base_offsets if base_offsets is not None
            else cruise_offsets(veh, lim, 0.0, lateral_offset=lateral_offset))

    dref = np.diff(ref, axis=1)
    psi = np.full(sim.N, heading)
    vh = np.hypot(dref[0], dref[1])
    good = vh > 1e-6
    psi[:-1][good] = np.arctan2(dref[1][good], dref[0][good])
    psi[-1] = psi[-2] if sim.N >= 2 else heading

    cos, sin = np.cos(psi), np.sin(psi)
    tgt = []
    for off in base:
        x = off[0] * cos - off[1] * sin
        y = off[0] * sin + off[1] * cos
        tgt.append(np.vstack([x, y, np.full(sim.N, off[2])]))
    return tgt


def build_formation_anchor(ref, phases, sim: SimParams, veh: VehicleParams,
                           lim: StateLimits, heading: float,
                           lateral_offset: float, ramp_frac: float = 1.0,
                           release_frac: float = 0.0, base_offsets=None):
    """Return ``(form_tgt, form_gate)`` for solve_rhc.

    `form_tgt` is a list of per-UAV (3, N) target offsets-from-payload; `form_gate`
    is a length-N per-node weight in [0, 1]. Pass both to solve_rhc together with a
    non-zero ``W_form`` weight to enable the anchor. ``ramp_frac`` controls where in
    the transition window the gate ramps up and ``release_frac`` whether/where it
    ramps back down to 0 before the window ends (see _formation_gate).
    ``base_offsets`` (heading=0 frame) overrides the analytic anchor geometry with a
    discovered one, e.g. from ``optimal_cruise_offsets``.
    """
    return (_formation_target(ref, sim, veh, lim, heading, lateral_offset,
                              base_offsets=base_offsets),
            _formation_gate(phases, sim, ramp_frac, release_frac))


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
              weights: CostWeights, x0=None, print_level=0, u0=None, Tc0=None,
              form_tgt=None, form_gate=None):
    """Build a trajectory-tracking NLP over sim.N timesteps.

    `weights` carries the constant cost weights for the run (see
    config.build_weights); the same weights are used for every window. If `x0` is
    None the initial state is left free. If `x0 = (uav_states0, payload_pos0,
    payload_vel0)` is given, the state at node 0 is pinned (used to stitch
    receding-horizon windows).
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
    # Every term is weighted by the constant scalars in `weights` (W_du is a
    # (3, 1) column, one entry per control) and summed over nodes. Rate terms
    # span node pairs.
    #
    # Payload position tracking (primary objective).
    cost = weights.W_track * ca.sum2(ca.sum1((payload_pos - ref)**2))

    # Per-control rate scaling, one entry per input row of u:
    # [thrust T, angle of attack alpha, bank angle mu].
    du_scale = ca.vertcat(lim.T_max - lim.T_min,
                          lim.alpha_max - lim.alpha_min,
                          2 * lim.mu_max)
    for i in range(sim.N_uav):
        # Per-control rate, weighted per control by W_du.
        du = (u[i][:, 1:] - u[i][:, :-1]) / du_scale
        cost += ca.sum2(ca.sum1(weights.W_du * du**2))

        # Cable-tension rate.
        dTc = (Tc[i][:, 1:] - Tc[i][:, :-1]) / lim.Tc_max
        cost += weights.W_dTc * ca.sum2(dTc**2)

        # Flight-path-angle rate: gamma = x[1], normalized by its admissible
        # range (±gam_max). Penalising the rate smooths the climb/descent profile.
        dgamma = (x[i][1, 1:] - x[i][1, :-1]) / (2 * lim.gam_max)
        cost += weights.W_dgamma * ca.sum2(dgamma**2)

        # Heading rate: chi = x[2]. sin(Δchi) handles ±π wrap-around; for the
        # small per-step changes expected here sin(Δchi) ≈ Δchi.
        dchi = ca.sin(x[i][2, 1:] - x[i][2, :-1])
        cost += weights.W_dchi * ca.sum2(dchi**2)

        # Airspeed rate: V = x[0], normalized by its admissible range.
        dV = (x[i][0, 1:] - x[i][0, :-1]) / (lim.V_max - lim.V_min)
        cost += weights.W_dV * ca.sum2(dV**2)

        # Thrust magnitude penalty to encourage lower thrust when possible (e.g.
        # for energy efficiency). Normalized by T_max to keep it in scale.
        cost += weights.W_T * ca.sum2((u[i][0, :] / lim.T_max)**2)

    # Formation anchor: pull each UAV toward its nominal offset-from-payload, gated
    # per node so it is inactive while the formation is meant to spin (climb) and
    # active in cruise (see build_formation_anchor). Error normalized by cable_len²
    # to keep W_form on the same scale as the other (normalized) cost terms.
    if form_tgt is not None and weights.W_form > 0:
        gate = ca.DM(np.asarray(form_gate, dtype=float).reshape(1, sim.N))
        for i in range(sim.N_uav):
            rel  = x[i][3:6, :] - payload_pos
            err2 = ca.sum1((rel - form_tgt[i])**2) / veh.cable_len**2   # (1, N)
            cost += weights.W_form * ca.sum2(gate * err2)

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
              weights: CostWeights, N_h=None, N_apply=None, print_level=0,
              spawn_ic=None, uav_offsets=None, heading_ref=None, v_ref=None,
              form_tgt=None, form_gate=None):
    """Receding-horizon solve of the payload-tracking problem.

    Slide a window of `N_h` nodes along the reference. Each window is a small NLP
    (built with `build_nlp`); we commit its first `N_apply` nodes to the output,
    then start the next window from the first uncommitted state so the stitched
    trajectory stays dynamically continuous.

    The cost `weights` are constant: the same set is passed to every window
    (see config.build_weights), so the cost is identical across the whole run.

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

    # Extend the formation anchor over the same padded horizon (hold the last
    # value), so every window can be sliced with the same `idx` as ref_ext.
    if form_tgt is not None:
        gate_ext = np.concatenate([form_gate, np.full(N_h, form_gate[-1])])
        tgt_ext  = [np.hstack([t, np.tile(t[:, -1:], (1, N_h))]) for t in form_tgt]

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
        win_tgt  = [t[:, idx] for t in tgt_ext] if form_tgt is not None else None
        win_gate = gate_ext[idx] if form_tgt is not None else None
        nlp = build_nlp(ref_ext[:, idx], win, veh, lim, weights,
                        x0=x0, u0=u0, Tc0=Tc0, print_level=print_level,
                        form_tgt=win_tgt, form_gate=win_gate)
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


@dataclass
class RunResult:
    """Everything a caller needs to report/plot one solved maneuver."""
    maneuver:    str
    sol:         Solution
    ref:         np.ndarray
    phases:      list
    form_gate:   np.ndarray   # (N,) per-node anchor gate (zeros when no anchor)
    weights:     "CostWeights"

    @property
    def form_weight(self):
        """Effective per-node formation weight (``W_form * gate``)."""
        return self.weights.W_form * np.asarray(self.form_gate)


def run_maneuver(cfg, maneuver: str, init_from: str = "", *, save: bool = True,
                 verbose: bool = True) -> RunResult:
    """Solve one maneuver (optionally chained from a saved run) and return it.

    Mirrors the standalone-script flow but is parametrised by `maneuver` and
    `init_from` instead of reading them from `cfg`, so a mission script can call it
    repeatedly to chain climb -> transition -> cruise. All other settings (sim,
    vehicle, limits, weights, formation options) still come from `cfg`.
    """
    from utils.config import build_weights
    from utils.maneuvers import MANEUVERS
    from utils.state_io import save_run, load_end_state

    sim, veh, lim = cfg.sim, cfg.veh, cfg.lim
    altitude, speed = cfg.common["altitude"], cfg.common["speed"]
    ref, phases = MANEUVERS[maneuver](sim, altitude, speed)
    weights = build_weights(cfg.weights, maneuver)

    heading = np.deg2rad(cfg.common["heading_deg"])
    lat_off = cfg.common["lateral_offset"]

    # ── Initial conditions: chain from a saved run, or start standalone. ────────
    init_from = (init_from or "").strip()
    if init_from:
        end = load_end_state(init_from)
        if verbose:
            print(f"chaining: starting '{maneuver}' from saved end-state of "
                  f"'{init_from}'")
        ref = ref + (end.payload_pos[:, None] - ref[:, :1])
        spawn_ic = end.as_spawn_ic()
        offsets = end.uav_offsets()
        heading = end.heading()
        v_ref = end.payload_vel
    else:
        offsets = cruise_offsets(veh, lim, heading, lateral_offset=lat_off)
        spawn_ic = (consistent_ic(sim, veh, lim, payload_pos0=ref[:, 0],
                                  heading=heading, lateral_offset=lat_off)
                    if cfg.common["spawn_ic"] else None)
        v_ref = speed * np.array([np.cos(heading), np.sin(heading), 0.0])

    # ── Phase-gated formation anchor (inactive when W_form == 0). ───────────────
    cruise_heading = np.deg2rad(cfg.common["heading_deg"])
    ramp_frac = float(cfg.common.get("form_ramp_frac", 1.0))
    release_frac = float(cfg.common.get("form_release_frac", 0.0))
    form_ref = (cfg.common.get("form_ref") or "analytic").strip().lower()
    base_offsets = None
    if form_ref == "optimized" and weights.W_form > 0:
        base_offsets = optimal_cruise_offsets(sim, veh, lim, weights,
                                              cruise_heading, lat_off, speed,
                                              altitude)
    form_tgt, form_gate = build_formation_anchor(ref, phases, sim, veh, lim,
                                                 cruise_heading, lat_off,
                                                 ramp_frac=ramp_frac,
                                                 release_frac=release_frac,
                                                 base_offsets=base_offsets)

    t_start = time.perf_counter()
    sol = solve_rhc(ref, sim, veh, lim, weights, print_level=0,
                    spawn_ic=spawn_ic, uav_offsets=offsets,
                    heading_ref=heading, v_ref=v_ref,
                    form_tgt=form_tgt, form_gate=form_gate)
    t_solve = time.perf_counter() - t_start

    if verbose:
        err = np.linalg.norm(sol.payload_pos - ref, axis=0)
        print(f"[{maneuver}] horizon: {sim.N} nodes, dt={sim.dt}s, "
              f"solve {t_solve:.2f}s, payload RMSE "
              f"{np.sqrt((err**2).mean()):.3f} m (max {err.max():.3f} m)")

    if save:
        saved = save_run(sol, maneuver)
        if verbose:
            print(f"saved run state to {saved}")

    return RunResult(maneuver=maneuver, sol=sol, ref=ref, phases=phases,
                     form_gate=form_gate, weights=weights)


if __name__ == "__main__":
    """Solve a payload-tracking trajectory, print a summary, and plot it.

    Everything is driven by config.yaml — pick the maneuver and tune its
    weights there; no edits to this file are needed.
    """
    from utils.plotting import print_distances, plot_solution, animate_solution
    from utils.config import load_config

    cfg = load_config()
    res = run_maneuver(cfg, cfg.maneuver,
                       cfg.common.get("init_from", ""))

    print_distances(res.sol, cfg.sim, cfg.veh)

    # show=True pops up interactive windows (rotatable 3D); the PNGs are saved
    # either way. Set show=False if running headless / over plain SSH.
    plot_solution(res.sol, res.ref, cfg.sim, cfg.lim, prefix=cfg.maneuver,
                  show=True, form_weight=res.form_weight)

    # GIF of the formation carrying the payload along the reference.
    animate_solution(res.sol, res.ref, cfg.sim, prefix=cfg.maneuver)
