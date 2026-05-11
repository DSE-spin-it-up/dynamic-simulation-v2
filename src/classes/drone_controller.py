import numpy as np

from ..utils.default_params import DEFAULT_PARAMS


class DroneController:
    def __init__(self, params: dict = DEFAULT_PARAMS):
        self.params = params

    def compute_thrust(self, drone, payload) -> np.ndarray:
        """Return thrust vector [Fx, Fy, 0] for circular horizontal orbit."""
        dx = drone.x - payload.x
        dy = drone.y - payload.y
        r = np.hypot(dx, dy)

        if r < 1e-9:
            return np.zeros(3)

        r_hat = np.array([dx / r, dy / r, 0.0])
        t_hat = np.array([-dy / r, dx / r, 0.0])

        dv = drone.v - payload.v
        r_dot = dv[0] * r_hat[0] + dv[1] * r_hat[1]
        omega = (dx * dv[1] - dy * dv[0]) / r**2

        # Centripetal feedforward: full centripetal force toward payload
        F_cf = -drone.mass * self.params["omega_target"] ** 2 * r * r_hat

        # Radial PD: keep drone at orbit radius R
        r_err = r - self.params["R"]
        F_r = -drone.mass * (self.params["kp_alt"] * r_err + self.params["kd_alt"] * r_dot) * r_hat

        # Tangential P: spin up from rest and maintain omega_target
        omega_err = self.params["omega_target"] - omega
        F_t = drone.mass * self.params["R"] * self.params["k_omega"] * omega_err * t_hat

        thrust = F_cf + F_r + F_t
        thrust[2] = 0.0
        return thrust
