import numpy as np

from ..utils.default_params import DEFAULT_PARAMS


class DroneController:
    def __init__(self, params: dict = DEFAULT_PARAMS):
        self.params = params

    def compute_thrust(self, drone, payload) -> np.ndarray:
        """Return thrust vector [Fx, Fy, Fz] for circular 3D orbit."""
        dx = drone.x - payload.x
        dy = drone.y - payload.y
        r = np.hypot(dx, dy)

        if r < 1e-9:
            return np.zeros(3)

        r_hat = np.array([dx / r, dy / r, 0.0])
        t_hat = np.array([-dy / r, dx / r, 0.0])
        z_hat = np.array([0.0, 0.0, 1.0])

        dv = drone.v - payload.v
        r_dot = dv[0] * r_hat[0] + dv[1] * r_hat[1]
        omega = (dx * dv[1] - dy * dv[0]) / r**2

        # Centripetal feedforward: inward force required for circular motion
        F_centripetal = -drone.mass * self.params["omega_target"] ** 2 * r * r_hat

        # Radial PD: drives drone to target orbit radius R
        r_err = r - self.params["R"]
        F_radial = -drone.mass * (self.params["kp_alt"] * r_err + self.params["kd_alt"] * r_dot) * r_hat

        # Tangential P: spins drone up from rest to omega_target
        omega_err = self.params["omega_target"] - omega
        F_tangential = drone.mass * self.params["R"] * self.params["k_omega"] * omega_err * t_hat

        # Cable feedforward: cancel the cable's pull so PD loops act as if no cable is attached
        r_vec_3d = drone.position - payload.position
        L_cable = np.linalg.norm(r_vec_3d)
        if L_cable > self.params["L0"]:
            T_cable = self.params["k_cable"] * (L_cable - self.params["L0"])
            F_cancel_cable_inward_pull   = T_cable * (r / L_cable) * r_hat
            F_cancel_cable_downward_pull = T_cable * r_vec_3d[2] / L_cable * z_hat
        else:
            F_cancel_cable_inward_pull   = np.zeros(3)
            F_cancel_cable_downward_pull = np.zeros(3)

        # Altitude PD: holds drone at absolute z_target
        z_err = drone.z - self.params["z_target"]
        F_altitude = drone.mass * (self.params["g"]
                                   - self.params["kp_z"] * z_err
                                   - self.params["kd_z"] * drone.vz) * z_hat
        
        thrust = F_centripetal + F_radial + F_tangential + F_cancel_cable_inward_pull + F_cancel_cable_downward_pull + F_altitude

        return thrust