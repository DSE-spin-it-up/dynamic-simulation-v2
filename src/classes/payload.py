import numpy as np


class Payload():
    def __init__(self, id: int, mass: float, initial_position: np.ndarray, initial_velocity: np.ndarray = np.array([0, 0, 0])):
        """
        Initialize a Payload object.
        
        Parameters
        ----------
        id : int
            The unique identifier for the payload. By convention, the payload ALWAYS uses id=-1.
            This distinguishes it from drone IDs (which are 0, 1, 2, ...).
        """
        self.id = id
        self.mass = mass
        self.position = np.array(initial_position, dtype=float, copy=True)
        self.v = np.array(initial_velocity if initial_velocity is not None else [0, 0, 0], dtype=float, copy=True)
        
    # ----------------------------------------------------------------------------------
    # Ignore this, it's just to print the payload object
    def __repr__(self):
        return (
            f"Payload object, id: {self.id}, position [m]: ({self.x}, {self.y}, {self.z}), mass [kg]: {self.mass}, velocity [m/s]: ({self.vx}, {self.vy}, {self.vz})\n"
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