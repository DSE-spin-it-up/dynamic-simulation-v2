import numpy as np

from ..utils.default_params import DEFAULT_PARAMS


class DroneController:
    def __init__(self, params: dict = DEFAULT_PARAMS):
        self.params = params
        self.integral_error = np.zeros(3) # integral PID error term

    def compute_thrust(self, drone, trajectories, t, trajectories_dt) -> np.ndarray:
        """PID controller to compute thrust vector for a drone.
        It takes which waypoint is currenty active and computes thrust to get there"""

        traj = trajectories[drone.id]

        # Convert time to trajectory index
        k = int(round(t / trajectories_dt))

        # Keep index within bounds
        k = min(k, len(traj) - 1)

        target_position = traj[k]
        target_velocity = self.estimate_velocity(traj, k, trajectories_dt)

        proportional_error = target_position - drone.position
        derivative_error = target_velocity - drone.v

        self.integral_error += proportional_error * DEFAULT_PARAMS["simulation_dt"]

        thrust = (
            DEFAULT_PARAMS["prop_error"] * proportional_error
            + DEFAULT_PARAMS["deriv_error"] * derivative_error
            + DEFAULT_PARAMS["int_error"] * self.integral_error
        )

        return thrust
    
    def estimate_velocity(self, traj, k, dt):
        n = len(traj)

        if k == 0:
            return (traj[1] - traj[0]) / dt
        elif k == n - 1:
            return (traj[-1] - traj[-2]) / dt
        else:
            return (traj[k + 1] - traj[k - 1]) / (2 * dt)