from turtle import speed

import numpy as np
from scipy.spatial.transform import Rotation

from src.classes import drone

from ..utils.default_params import DEFAULT_PARAMS


class DroneController:
    def __init__(self, params: dict = DEFAULT_PARAMS):
        self.params = params
        self.integral_error = np.array([0.0, 0.0, 50.0])  # integral PID error term
 
        # Persist last valid body frame to avoid degeneracy at low/vertical speed
        self._body_x = np.array([1.0, 0.0, 0.0])
        self._body_y = np.array([0.0, 1.0, 0.0])
        self._body_z = np.array([0.0, 0.0, 1.0])
 
    def compute_thrust(self, drone, trajectories, t, trajectories_dt) -> tuple[np.ndarray, tuple[float, float, float]]:
        traj = trajectories[drone.id]
 
        k = min(int(round(t / trajectories_dt)), len(traj) - 1)
 
        target_position = traj[k]
        target_velocity = self.estimate_velocity(traj, k, trajectories_dt)
 
        proportional_error = target_position - drone.position
        derivative_error   = target_velocity - drone.v
 
        # ── Compute raw thrust before integral update ──────────────────
        thrust_world = (
            DEFAULT_PARAMS["prop_error"]  * proportional_error
            + DEFAULT_PARAMS["deriv_error"] * derivative_error
            + DEFAULT_PARAMS["int_error"]   * self.integral_error
        )
 
        # ── Body frame from velocity ───────────────────────────────────
        speed = np.linalg.norm(drone.v)
 
        if speed > 1e-3:
            body_x = drone.v / speed
 
            # body_y = right wing: perpendicular to nose and world up
            # using cross product avoids Gram-Schmidt degeneracy
            world_up = np.array([0.0, 0.0, 1.0])
            body_y = np.cross(body_x, world_up)
            y_norm = np.linalg.norm(body_y)
 
            if y_norm > 1e-6:
                body_y /= y_norm
            else:
                # Flying perfectly vertical — pick arbitrary wing direction
                body_y = np.array([0.0, 1.0, 0.0])
 
            # body_z always well-defined as cross of two orthonormal vectors
            body_z = np.cross(body_y, body_x)
 
            # Persist valid frame
            self._body_x = body_x
            self._body_y = body_y
            self._body_z = body_z
 
        else:
            # Reuse last valid frame — avoids noise corrupting projection
            body_x = self._body_x
            body_y = self._body_y
            body_z = self._body_z
 
        # ── Project thrust into body frame ─────────────────────────────
        forward_component = np.dot(thrust_world, body_x)
        lateral_component = np.dot(thrust_world, body_y)
        upward_component  = np.dot(thrust_world, body_z)
 
        # ── Clamp per axis ─────────────────────────────────────────────
        forward_limit = DEFAULT_PARAMS["thrust_limit_forward"]
        lateral_limit = DEFAULT_PARAMS["thrust_limit_lateral"]
        upward_limit  = DEFAULT_PARAMS["thrust_limit_upward"]
 
        forward_saturated = abs(forward_component) > forward_limit
        lateral_saturated = abs(lateral_component) > lateral_limit
        upward_saturated  = abs(upward_component)  > upward_limit
 
        forward_component = np.clip(forward_component, -forward_limit,  forward_limit)
        lateral_component = np.clip(lateral_component, -lateral_limit,  lateral_limit)
        upward_component  = np.clip(upward_component,  -upward_limit,   upward_limit)
 
        # ── Reconstruct thrust in world frame ──────────────────────────
        thrust = (forward_component * body_x
                + lateral_component * body_y
                + upward_component  * body_z)
 
        # ── Anti-windup: only integrate non-saturated axes ─────────────
        # Map body saturation back to world frame and block integration
        # on any world axis that contributes to a saturated body axis.
        saturation_body = np.array([
            float(forward_saturated),
            float(lateral_saturated),
            float(upward_saturated),
        ])
        saturation_world = (
            np.abs(body_x) * saturation_body[0]
            + np.abs(body_y) * saturation_body[1]
            + np.abs(body_z) * saturation_body[2]
        )
        integration_mask = (saturation_world < 0.5).astype(float)
 
        self.integral_error += (
            proportional_error
            * DEFAULT_PARAMS["simulation_dt"]
            * integration_mask
        )
 
        # Per-axis clamp as safety net
        self.integral_error = np.clip(
            self.integral_error,
            -DEFAULT_PARAMS["integral_limit"],
             DEFAULT_PARAMS["integral_limit"],
        )
 
        return thrust, (forward_component, lateral_component, upward_component)
 
    def estimate_velocity(self, traj, k, dt):
        n = len(traj)
 
        if k == 0:
            return (traj[1] - traj[0]) / dt
        elif k == n - 1:
            return (traj[-1] - traj[-2]) / dt
        else:
            return (traj[k + 1] - traj[k - 1]) / (2 * dt)