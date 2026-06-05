"""Configuration layer for the trajectory planner.

Pure data — no CasADi, no import of :mod:`trajectory_planning`. This module owns

* the problem parameter dataclasses (``SimParams``, ``VehicleParams``,
  ``StateLimits``), which ``trajectory_planning`` re-exports;
* the cost-weight container ``CostWeights`` and the maneuver-keyed builder
  ``build_weights``;
* ``load_config`` / ``Config``, which read ``config.yaml`` into the above.

The dependency direction is one-way: ``config -> (nothing)``,
``trajectory_planning -> config``, ``plotting -> trajectory_planning``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .default_params import SimParams, VehicleParams, StateLimits, CostWeights, Config

# ── Cost weights ──────────────────────────────────────────────────────────────
# The cost weights are constant over the whole trajectory and selected by the
# active maneuver: ``build_weights`` reads the per-maneuver block from config.yaml
# (falling back to ``"default"`` for any omitted weight) into the scalar container
# below, which broadcasts element-wise against the CasADi cost in ``build_nlp``.

# Weight keys carried in the cost. W_du is the only vector-valued key (one entry
# per control row of u: [thrust T, angle of attack alpha, bank angle mu]).
WEIGHT_KEYS = ("W_track", "W_du", "W_dTc", "W_dgamma", "W_dchi", "W_dV", "W_T",
               "W_form")


def build_weights(weights_by_maneuver: dict, maneuver: str) -> CostWeights:
    """Build the constant ``CostWeights`` for the active ``maneuver``.

    Parameters
    ----------
    weights_by_maneuver : dict
        Maneuver name -> partial weight dict. The ``"default"`` entry supplies any
        weight the maneuver block omits (``W_du`` is a length-3 list).
    maneuver : str
        Name of the active maneuver (matches a key in ``MANEUVERS``).
    """
    defaults = weights_by_maneuver.get("default", {})
    block    = weights_by_maneuver.get(maneuver, {})

    def resolved(key):
        return block.get(key, defaults[key])

    kwargs = {}
    for key in WEIGHT_KEYS:
        if key == "W_du":
            kwargs[key] = np.asarray(resolved(key), dtype=float).reshape(3, 1)
        else:
            kwargs[key] = float(resolved(key))
    return CostWeights(**kwargs)


# ── Top-level config ──────────────────────────────────────────────────────────


def load_config(path: str | Path = "config.yaml") -> Config:
    """Read ``config.yaml`` into a :class:`Config`.

    Angles in the ``limits`` block are given in degrees (``*_deg`` keys) and
    converted to radians here. The ``maneuver`` name is kept as a string; it is
    resolved against ``MANEUVERS`` in ``utils.maneuvers``.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    return Config(
        sim=SimParams(),
        veh=VehicleParams(),
        lim=StateLimits(),
        maneuver=raw["maneuver"],
        common=raw.get("common", {}),
        weights=raw["weights"],
    )

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
    psi = np.full(sim.N_h, heading)
    vh = np.hypot(dref[0], dref[1])
    good = vh > 1e-6
    psi[:-1][good] = np.arctan2(dref[1][good], dref[0][good])
    psi[-1] = psi[-2] if sim.N_h >= 2 else heading

    cos, sin = np.cos(psi), np.sin(psi)
    tgt = []
    for off in base:
        x = off[0] * cos - off[1] * sin
        y = off[0] * sin + off[1] * cos
        tgt.append(np.vstack([x, y, np.full(sim.N_h, off[2])]))
    return tgt


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


def cruise_offsets(veh: VehicleParams, lim: StateLimits, heading: float,
                   lateral_offset: float = 6.0,
                   forward_offset: float = None) -> list: #type: ignore
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
