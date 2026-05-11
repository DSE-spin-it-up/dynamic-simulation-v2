import numpy as np

from .payload import Payload
from .drone import Drone

class Cable():
    def __init__(self, id, length, stiffness, damping, conection_A: Payload, conection_B: Drone):
        self.id = id
        self.length = length
        self.stiffness = stiffness
        self.damping = damping
        self.conection_A = conection_A
        self.conection_B = conection_B

    def __repr__(self):
        return (
            f"Cable object, id: {self.id}, length [m]: {self.length}, stiffness [N/m]: {self.stiffness}, damping [N s/m]: {self.damping}\n"
            f"Assigned nodes\n  n1: {self.conection_A}\n  n2: {self.conection_B}\n"
        )

    def __relative_pos(self):
        return np.array([self.conection_A.position - self.conection_B.position])

    def __relative_vel(self):
        return np.array([self.conection_A.v - self.conection_B.v])

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

    def force_value(self):
        """Returns force for tensile-only cable (no compression)"""
        current_leght = np.linalg.norm(self.__relative_pos())
        if current_leght >= self.length:
            return self.__calculate_f_spring() + self.__calculate_f_damping()
        else:
            return np.array([0, 0, 0])