import numpy as np

from ..utils.default_params import DEFAULT_PARAMS


class DroneController:
    def __init__(self, params: dict = DEFAULT_PARAMS):
        self.params = params

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def compute_thrust(self, drone, payload, t: float = 0.0, mission=None) -> np.ndarray:
        """
        Return thrust vector [Fx, Fy, Fz] in Newtons.

        Dispatches to one of four control modes based on mission phase:
          - ORBIT        : SPINUP_GROUND, SPIN_CLIMB, and legacy default
          - TAKEOFF      : TAKEOFF  (altitude climb only, no spin)
          - TRANSITION   : orbit winds down + forward velocity builds + y-slot correction
          - POSITION_PD  : CRUISE  (3-D position tracking to V-formation slot)

        get_commands() signals TRANSITION mode by returning pos_cmd=None and
        vel_cmd = [v_forward, y_slot, 0] with omega_cmd > 0.
        """
        if mission is None:
            return self._orbit_thrust(
                drone, payload,
                z_cmd=self.params["z_target"],
                omega_cmd=self.params["omega_target"],
            )

        z_cmd, omega_cmd, pos_cmd, vel_cmd = mission.get_commands(
            drone.id, t, drone, payload
        )

        if pos_cmd is not None:
            # CRUISE: full 3-D position PD to V-formation slot
            return self._position_pd_thrust(drone, payload, pos_cmd, vel_cmd)
        elif omega_cmd > 0.0 and vel_cmd is not None:
            # TRANSITION: orbit winds down, forward speed and y-slot build up
            v_forward = float(vel_cmd[0])
            y_slot    = float(vel_cmd[1])
            return self._transition_thrust(drone, payload, z_cmd, omega_cmd, v_forward, y_slot)
        elif omega_cmd == 0.0:
            # TAKEOFF: altitude climb, no spin
            return self._takeoff_thrust(drone, payload, z_cmd)
        else:
            # SPINUP_GROUND / SPIN_CLIMB: full orbit
            return self._orbit_thrust(drone, payload, z_cmd=z_cmd, omega_cmd=omega_cmd)

    # ------------------------------------------------------------------
    # Mode A — ORBIT  (SPINUP_GROUND, SPIN_CLIMB, and legacy)
    # ------------------------------------------------------------------

    def _orbit_thrust(self, drone, payload, z_cmd: float, omega_cmd: float) -> np.ndarray:
        """Circular-orbit controller with cable feedforward."""
        dx = drone.x - payload.x
        dy = drone.y - payload.y
        r  = np.hypot(dx, dy)

        if r < 1e-9:
            return np.zeros(3)

        r_hat = np.array([dx / r, dy / r, 0.0])
        t_hat = np.array([-dy / r, dx / r, 0.0])
        z_hat = np.array([0.0, 0.0, 1.0])

        dv    = drone.v - payload.v
        r_dot = dv[0] * r_hat[0] + dv[1] * r_hat[1]
        omega = (dx * dv[1] - dy * dv[0]) / r**2

        # Centripetal feedforward
        F_centripetal = -drone.mass * omega_cmd**2 * r * r_hat

        # Radial PD: drives drone to target orbit radius R
        r_err    = r - self.params["R"]
        F_radial = -drone.mass * (
            self.params["kp_alt"] * r_err + self.params["kd_alt"] * r_dot
        ) * r_hat

        # Tangential P: spins drone up to omega_cmd
        omega_err    = omega_cmd - omega
        F_tangential = drone.mass * self.params["R"] * self.params["k_omega"] * omega_err * t_hat

        # Cable feedforward: cancel cable's inward + downward pull
        F_cancel_cable_inward_pull, F_cancel_cable_downward_pull = \
            self._cable_cancel(drone, payload, r_hat, z_hat)

        # Altitude PD: hold absolute z_cmd
        z_err    = drone.z - z_cmd
        F_altitude = drone.mass * (
            self.params["g"]
            - self.params["kp_z"] * z_err
            - self.params["kd_z"] * drone.vz
        ) * z_hat

        return (F_centripetal
                + F_radial
                + F_tangential
                + F_cancel_cable_inward_pull
                + F_cancel_cable_downward_pull
                + F_altitude)

    # ------------------------------------------------------------------
    # Mode B — TAKEOFF  (altitude climb, no spin)
    # ------------------------------------------------------------------

    def _takeoff_thrust(self, drone, payload, z_cmd: float) -> np.ndarray:
        """Climb vertically to z_cmd; hold starting xy position; no spin."""
        dx = drone.x - payload.x
        dy = drone.y - payload.y
        r  = np.hypot(dx, dy)

        z_hat = np.array([0.0, 0.0, 1.0])

        if r < 1e-9:
            r_hat = np.zeros(3)
        else:
            r_hat = np.array([dx / r, dy / r, 0.0])

        dv    = drone.v - payload.v
        r_dot = dv[0] * r_hat[0] + dv[1] * r_hat[1]

        # Radial PD only (no centripetal, no tangential)
        r_err    = r - self.params["R"]
        F_radial = -drone.mass * (
            self.params["kp_alt"] * r_err + self.params["kd_alt"] * r_dot
        ) * r_hat

        _, F_cancel_cable_downward_pull = self._cable_cancel(drone, payload, r_hat, z_hat)

        z_err    = drone.z - z_cmd
        F_altitude = drone.mass * (
            self.params["g"]
            - self.params["kp_z"] * z_err
            - self.params["kd_z"] * drone.vz
        ) * z_hat

        return F_radial + F_cancel_cable_downward_pull + F_altitude

    # ------------------------------------------------------------------
    # Mode C — TRANSITION  (orbit unwinds + forward flight builds)
    # ------------------------------------------------------------------

    def _transition_thrust(self, drone, payload, z_cmd: float, omega_cmd: float,
                           v_forward: float, y_slot: float) -> np.ndarray:
        """
        Keep full orbit control (centripetal + radial PD + tangential P) with
        decaying omega_cmd so the circular motion physically winds down, while
        simultaneously pushing the drone toward v_cruise in +x and guiding it
        gently to its V-formation lateral slot in y.

        This looks realistic: the orbit unwinds as the formation accelerates
        forward, rather than teleporting to a geometric blend target.
        """
        # Orbit control handles: centripetal, radial, tangential, cable cancel, altitude
        F_orbit = self._orbit_thrust(drone, payload, z_cmd, omega_cmd)

        # Forward velocity push — builds v_cruise over the transition duration
        kd = self.params["kd_cruise"]
        F_forward = (drone.mass * kd * (v_forward - drone.vx)
                     * np.array([1.0, 0.0, 0.0]))

        # Gentle y correction toward V-formation lateral slot
        kp_y  = self.params["kp_cruise"] * 0.25   # softer than cruise position gain
        y_err = drone.y - y_slot
        F_y   = -drone.mass * kp_y * y_err * np.array([0.0, 1.0, 0.0])

        return F_orbit + F_forward + F_y

    # ------------------------------------------------------------------
    # Mode D — POSITION PD  (CRUISE)
    # ------------------------------------------------------------------

    def _position_pd_thrust(self, drone, payload, pos_cmd: np.ndarray,
                            vel_cmd: np.ndarray) -> np.ndarray:
        """3-D position PD controller tracking pos_cmd / vel_cmd."""
        z_hat = np.array([0.0, 0.0, 1.0])

        pos_err = drone.position - pos_cmd
        vel_err = drone.v - vel_cmd

        F_pd = drone.mass * (
            -self.params["kp_cruise"] * pos_err
            - self.params["kd_cruise"] * vel_err
        )
        F_grav = drone.mass * self.params["g"] * z_hat

        # Cable cancel (keeps feedforward accurate even during transition)
        dx = drone.x - payload.x
        dy = drone.y - payload.y
        r  = np.hypot(dx, dy)
        r_hat = np.array([dx / r, dy / r, 0.0]) if r > 1e-9 else np.zeros(3)
        F_cancel_inward, F_cancel_down = self._cable_cancel(drone, payload, r_hat, z_hat)

        return F_pd + F_grav + F_cancel_inward + F_cancel_down

    # ------------------------------------------------------------------
    # Shared cable feedforward
    # ------------------------------------------------------------------

    def _cable_cancel(self, drone, payload, r_hat: np.ndarray,
                      z_hat: np.ndarray) -> tuple:
        """
        Return (F_cancel_inward_pull, F_cancel_downward_pull).

        Computes the cable tension from current geometry and returns
        equal-and-opposite forces so PD loops behave as if no cable exists.
        """
        r_vec_3d = drone.position - payload.position
        L_cable  = np.linalg.norm(r_vec_3d)
        L0       = self.params["L0"]

        if L_cable > L0:
            T_cable = self.params["k_cable"] * (L_cable - L0)
            r_xy    = np.hypot(r_vec_3d[0], r_vec_3d[1])
            F_inward = T_cable * (r_xy / L_cable) * r_hat
            F_down   = T_cable * r_vec_3d[2] / L_cable * z_hat
        else:
            F_inward = np.zeros(3)
            F_down   = np.zeros(3)

        return F_inward, F_down