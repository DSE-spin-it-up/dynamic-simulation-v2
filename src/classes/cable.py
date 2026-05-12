import numpy as np

from .payload import Payload
from .drone import Drone

class Cable():
    def __init__(self, id, length, stiffness, damping, payload: Payload, drone: Drone):
        self.id = id
        self.length = length
        self.stiffness = stiffness
        self.damping = damping
        self.payload = payload
        self.drone = drone

    def __repr__(self):
        return (
            f"Cable object, id: {self.id}, length [m]: {self.length}, stiffness [N/m]: {self.stiffness}, damping [N s/m]: {self.damping}\n"
            f"Assigned nodes\n  n1: {self.payload}\n  n2: {self.drone}\n"
        )
    
    #TODO: in the future, move the cable physics calculation to pyisics.py

    def __relative_pos(self):
        return np.array([self.drone.position - self.payload.position])

    def __relative_vel(self):
        return np.array([self.drone.v - self.payload.v])

    def __calculate_f_spring(self):
        relative_pos = self.__relative_pos()
        norm_pos = np.linalg.norm(relative_pos)

        if norm_pos != 0:
            unit_vector = relative_pos / norm_pos
        else:
            unit_vector = np.array([0, 0, 0])

        f_spring = -self.stiffness * (norm_pos - self.length) * unit_vector
        return np.squeeze(f_spring)

    def __calculate_f_damping(self):
        relative_pos = self.__relative_pos()
        relative_vel = np.squeeze(self.__relative_vel())
        norm_pos = np.linalg.norm(relative_pos)

        if norm_pos != 0:
            unit_vector = np.squeeze(relative_pos / norm_pos)
        else:
            unit_vector = np.squeeze(np.array([0, 0, 0]))

        f_damping = -self.damping * np.dot(relative_vel, unit_vector) * unit_vector
        return np.squeeze(f_damping)

    def _force_vector_drone(self):
        """Returns force on drone for tensile-only cable (no compression)"""
        current_length = np.linalg.norm(self.__relative_pos())
        if current_length >= self.length:
            return self.__calculate_f_spring() + self.__calculate_f_damping()
        else:
            return np.array([0, 0, 0])
        
    def force_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        ''' Returns the force vectors acting on the payload and drone, in that order, in the global frame.'''
        f_drone = self._force_vector_drone()
        f_payload = -f_drone
        return f_payload, f_drone