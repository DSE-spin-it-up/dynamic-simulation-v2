import numpy as np

from .default_params import DEFAULT_PARAMS, VehicleParams, StateLimits


def get_initial_states(
    num_drones: int = DEFAULT_PARAMS["n_drones"],
    R: float = DEFAULT_PARAMS["R"],
    L0: float = DEFAULT_PARAMS["L0"],
    payload_pos: np.ndarray = np.zeros(3),
    z_target: float = DEFAULT_PARAMS["z_target"],
    v_tangent_mag: float = 20.0,  # Explicit magnitude configuration
) -> dict:
    """
    Places drones in a flat circle of radius R around the payload's XY position.
    Velocities are strictly tangential (counter-clockwise) to their position vectors.
    """
    angles = np.linspace(0, 2 * np.pi, num_drones, endpoint=False)

    # 1. Position Setup: Coplanar with the payload (same altitude)
    drone_z = payload_pos[2] 
    drone_positions = np.column_stack([
        payload_pos[0] + R * np.cos(angles),
        payload_pos[1] + R * np.sin(angles),
        np.full(num_drones, drone_z),
    ])

    # 2. Velocity Setup: Perfectly perpendicular to the position vector
    # Drone at 0 rad flies UP (+Y). Drone at pi/2 rad flies LEFT (-X).
    drone_velocities = np.column_stack([
        -v_tangent_mag * np.sin(angles),
         v_tangent_mag * np.cos(angles),
         np.zeros(num_drones)  # No vertical velocity component at t=0
    ])

    payload_velocity = np.zeros(3)

    # 3. Package into state dict
    states = {}
    for i in range(num_drones):
        states[i] = {
            "position": drone_positions[i],
            "velocity": drone_velocities[i],
        }

    # Reserved ID for payload remains -1
    states[-1] = {
        "position": payload_pos.copy(),
        "velocity": payload_velocity,
    }

    return states


def _equilibrium_forward_offset(veh: VehicleParams, lim: StateLimits,
                                lateral_offset: float) -> float:
    """Forward offset [m] that gives equal positive cable tensions at cruise.

    At cruise the payload force balance requires the cable angle to satisfy:
        (v0 + 2·vs) / (3·f) = m_L·g / F_drag
    where v0 = sqrt(L²−f²), vs = sqrt(L²−f²−l²), l = lateral_offset.
    Solved by bisection.
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
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cruise_offsets(veh: VehicleParams, lim: StateLimits, heading: float,
                   lateral_offset: float = 6.0) -> list:
    """Return the three UAV-from-payload position offsets for straight cruise.

    UAV 0 is centred ahead, UAVs 1/2 are left and right. All offsets satisfy
    |offset| == cable_len exactly.
    """
    forward_offset = _equilibrium_forward_offset(veh, lim, lateral_offset)
    assert forward_offset**2 + lateral_offset**2 < veh.cable_len**2, (
        "forward_offset and lateral_offset are too large for the cable length")
    forward = np.array([np.cos(heading), np.sin(heading), 0.0])
    right   = np.array([-np.sin(heading), np.cos(heading), 0.0])
    up      = np.array([0.0, 0.0, 1.0])
    v0 = np.sqrt(veh.cable_len**2 - forward_offset**2)
    vs = np.sqrt(veh.cable_len**2 - forward_offset**2 - lateral_offset**2)
    return [
        forward_offset * forward + v0 * up,
        forward_offset * forward - lateral_offset * right + vs * up,
        forward_offset * forward + lateral_offset * right + vs * up,
    ]


def get_cruise_initial_states(veh: VehicleParams, lim: StateLimits,
                              num_drones: int = DEFAULT_PARAMS["n_drones"],
                              payload_pos: np.ndarray = np.array([0.0, 0.0, 100.0]),
                              heading: float = np.pi / 2,
                              lateral_offset: float = 6.0) -> dict:
    """Steady level-cruise initial state for the whole system.

    Returns the same dict format as get_initial_states, but places UAVs in
    the equilibrium forward-cruise formation so that all cables carry positive
    tension from the first timestep.
    """
    assert num_drones == 3, "get_cruise_initial_states currently requires exactly 3 UAVs"
    payload_pos = np.asarray(payload_pos, dtype=float)

    offsets = cruise_offsets(veh, lim, heading, lateral_offset=lateral_offset)
    drone_velocity = lim.V_cruise * np.array([np.cos(heading), np.sin(heading), 0.0])
    payload_velocity = drone_velocity.copy()

    states = {}
    for i, offset in enumerate(offsets):
        states[i] = {
            "position": payload_pos + offset,
            "velocity": drone_velocity.copy(),
        }
    states[-1] = {
        "position": payload_pos.copy(),
        "velocity": payload_velocity,
    }
    return states