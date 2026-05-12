import numpy as np

from .drone_controller import DroneController
from ..utils.default_params import DEFAULT_PARAMS

class Drone():
    def __init__(self, id: int, mass: float, initial_position: np.ndarray, initial_velocity: np.ndarray = np.array([0, 0, 0])):
        self.id = id
        self.mass = mass
        self.position = initial_position
        self.v = initial_velocity
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

    # ----------------------------------------------------------------------------------

    def compute_aero_forces(self) -> np.ndarray:
        """Placeholder for aerodynamic forces. Currently returns zero."""
        return np.zeros(3)
    
    def compute_gravity_force(self) -> np.ndarray:
        """Compute gravitational force vector [0, 0, -mg]."""
        return np.array([0, 0, -self.mass * 9.81])

    def apply_force(self, force: np.ndarray):
        """Update velocity and position based on applied force."""
        # Simple Euler integration for demonstration (not used in actual simulation)
        acceleration = force / self.mass
        self.v += acceleration * DEFAULT_PARAMS["dt"]  # Assume small time step for this update
        self.position += self.v * DEFAULT_PARAMS["dt"]