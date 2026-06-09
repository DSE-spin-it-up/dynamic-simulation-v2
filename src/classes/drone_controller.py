from turtle import speed

import numpy as np
from scipy.spatial.transform import Rotation

from src.classes import drone

from ..utils.default_params import DEFAULT_PARAMS


class DroneController:
    def __init__(self, params: dict = DEFAULT_PARAMS):
        self.params = params
        self.integral_error = np.array([0.0, 0.0, 50.0]) # integral PID error term

    def compute_thrust(self, drone, trajectories, t, trajectories_dt) -> tuple[np.ndarray, tuple[float, float, float]]:
        traj = trajectories[drone.id]

        k = min(int(round(t / trajectories_dt)), len(traj) - 1)

        target_position = traj[k]
        target_velocity = self.estimate_velocity(traj, k, trajectories_dt)

        proportional_error = target_position - drone.position
        derivative_error   = target_velocity - drone.v

        self.integral_error += proportional_error * DEFAULT_PARAMS["simulation_dt"]

        int_norm = np.linalg.norm(self.integral_error)
        if int_norm > DEFAULT_PARAMS["integral_limit"]:
            self.integral_error *= DEFAULT_PARAMS["integral_limit"] / int_norm

        thrust_world = (
            DEFAULT_PARAMS["prop_error"]  * proportional_error
            + DEFAULT_PARAMS["deriv_error"] * derivative_error
            + DEFAULT_PARAMS["int_error"]   * self.integral_error
        )

        # ── Body frame from velocity ───────────────────────────────────
        speed = np.linalg.norm(drone.v)

        if speed > 1e-3:
            # nose points along velocity
            body_x = drone.v / speed

            # body_y perpendicular to body_x in the xy plane
            # project body_x onto xy, rotate 90 degrees
            body_y = np.array([-body_x[1], body_x[0], 0.0])
            y_norm = np.linalg.norm(body_y)
            if y_norm > 1e-6:
                body_y /= y_norm
            else:
                # flying straight up or down — pick arbitrary y
                body_y = np.array([0.0, 1.0, 0.0])

            # body_z points up (world z, orthogonalised against body_x)
            body_z = np.array([0.0, 0.0, 1.0])
            body_z -= np.dot(body_z, body_x) * body_x
            body_z -= np.dot(body_z, body_y) * body_y
            bz_norm = np.linalg.norm(body_z)

            # ── DEBUG ──────────────────────────────────────────────────
            if bz_norm < 1e-6:
                print(f"[DEGENERATE body_z] drone={drone.id} body_x={body_x} body_y={body_y} bz_norm={bz_norm}")
                return np.zeros(3), (0.0, 0.0, 0.0)
            body_z /= np.linalg.norm(body_z)

            # ── Project thrust into body frame ─────────────────────────
            forward_component  = np.dot(thrust_world, body_x)
            lateral_component  = np.dot(thrust_world, body_y)
            upward_component   = np.dot(thrust_world, body_z)

            # ── Clamp per axis ─────────────────────────────────────────
            forward_component  = np.clip(forward_component,
                                         -DEFAULT_PARAMS["thrust_limit_forward"],
                                         DEFAULT_PARAMS["thrust_limit_forward"])
            lateral_component  = np.clip(lateral_component,
                                         -DEFAULT_PARAMS["thrust_limit_lateral"],
                                          DEFAULT_PARAMS["thrust_limit_lateral"])
            upward_component   = np.clip(upward_component,
                                         -DEFAULT_PARAMS["thrust_limit_upward"],
                                          DEFAULT_PARAMS["thrust_limit_upward"])

            # ── Reconstruct in world frame ─────────────────────────────
            thrust = (forward_component * body_x
                    + lateral_component * body_y
                    + upward_component  * body_z)
        else:
            # No velocity — fall back to world-frame clamp
            thrust = np.clip(thrust_world,
                             -DEFAULT_PARAMS["thrust_limit_forward"],
                              DEFAULT_PARAMS["thrust_limit_forward"])
    
        return thrust, (forward_component, lateral_component, upward_component)
    
    def estimate_velocity(self, traj, k, dt):
        n = len(traj)

        if k == 0:
            return (traj[1] - traj[0]) / dt
        elif k == n - 1:
            return (traj[-1] - traj[-2]) / dt
        else:
            return (traj[k + 1] - traj[k - 1]) / (2 * dt)