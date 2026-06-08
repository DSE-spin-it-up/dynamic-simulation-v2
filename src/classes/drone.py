import numpy as np
from scipy.spatial.transform import Rotation

from src.utils.default_params import DEFAULT_PARAMS

from .drone_controller import DroneController

class Drone():
    def __init__(
        self,
        id: int,
        mass: float,
        initial_position: np.ndarray,
        initial_velocity: np.ndarray = np.array([0.0, 0.0, 0.0]),
        inertia: np.ndarray = DEFAULT_PARAMS["drone_inertia"],
    ):
        self.id       = id
        self.mass     = mass
        self.position = np.array(initial_position, dtype=float, copy=True)
        self.v        = np.array(initial_velocity if initial_velocity is not None
                                 else [0.0, 0.0, 0.0], dtype=float, copy=True)

        # --- attitude state (new) ---
        self.q      = np.array([0.0, 0.0, 0.0, 1.0])          # quaternion, scalar-last (x,y,z,w), identity = level
        self.omega  = np.zeros(3)                               # angular velocity, body frame [rad/s]
        self.inertia = DEFAULT_PARAMS["drone_inertia"]                      # inertia tensor, body frame [kg·m²]

        self.controller = DroneController()

    # ----------------------------------------------------------------------------------
    # Ignore this, it's just to print the drone object
    def __repr__(self):
        return (
            f"Drone object, id: {self.id}, position [m]: ({self.x}, {self.y}, {self.z}), mass [kg]: {self.mass}, velocity [m/s]: ({self.vx}, {self.vy}, {self.vz})\n"
        )
    
    # ----------------------------------------------------------------------------------
    # Also ignore this, is just for easier access to position and velocity components
    @property
    def vx(self):
        return self.v[0]

    @property
    def vy(self):
        return self.v[1]

    @property
    def vz(self):
        return self.v[2]
    
    @property
    def x(self):
        return self.position[0]
    @property
    def y(self):
        return self.position[1]
    @property
    def z(self):
        return self.position[2]

    # ── attitude derived properties ────────────────────────────
    @property
    def rotation(self) -> Rotation:
        """Current attitude as a scipy Rotation."""
        return Rotation.from_quat(self.q)

    @property
    def euler_angles(self) -> np.ndarray:
        """[roll, pitch, yaw] in radians (ZYX convention)."""
        return self.rotation.as_euler("ZYX")[::-1]

    @property
    def body_z(self) -> np.ndarray:
        """Thrust axis (body +Z) expressed in world frame."""
        return self.rotation.apply(np.array([0.0, 0.0, 1.0]))