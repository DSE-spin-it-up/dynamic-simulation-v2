import numpy as np

from .drone_controller import DroneController

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