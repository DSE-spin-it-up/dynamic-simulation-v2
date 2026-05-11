import numpy as np
import pytest
import sys
from pathlib import Path

from src.classes.cable import Cable
from src.classes.drone import Drone
from src.classes.payload import Payload


@pytest.mark.parametrize("test_name,drone_position,drone_velocity,expected_force_payload,expected_force_drone", [
    ("cable_at_rest", [0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]),
    ("cable_stretched", [0, 0, 1.5], [0, 0, 0], [0, 0, 50], [0, 0, -50]),
    ("cable_compressed", [0, 0, 0.5], [0, 0, 0], [0, 0, 0], [0, 0, 0]),
    ("cable_damping", [0, 0, 1], [0, 0, 1], [0, 0, 10], [0, 0, -10]),
    ("rotated_cable", [1, 0, 1], [0, 0, 0], [29.28932188, 0, 29.28932188], [-29.28932188, 0, -29.28932188]),
    ("combinated_case", [1, 0, 1], [1, 0, 1], [39.28932188, 0, 39.28932188], [-39.28932188, 0, -39.28932188]),
    ("misaligned_velocity", [1, 0, 1], [0, 0, 1], [34.28932188, 0, 34.28932188], [-34.28932188, 0, -34.28932188]),
])
def test_cable_tension(test_name, drone_position, drone_velocity, expected_force_payload, expected_force_drone):
    """Test cable tension with various configurations."""
    dummyPayload = Payload(mass=1, initial_position=np.array([0, 0, 0]), initial_velocity=np.array([0, 0, 0]))
    dummyDrone = Drone(id=1, mass=1, initial_position=np.array([0, 0, 1]), initial_velocity=np.array([0, 0, 0]))
    cable = Cable(id=0, length=1, stiffness=100, damping=10, payload=dummyPayload, drone=dummyDrone)
    
    dummyDrone.position = np.array(drone_position)
    dummyDrone.v = np.array(drone_velocity)
    
    forces = cable.force_vectors()
    assert np.allclose(forces[0], expected_force_payload), f"[{test_name}] Expected {expected_force_payload}, got {forces[0]}"
    assert np.allclose(forces[1], expected_force_drone), f"[{test_name}] Expected {expected_force_drone}, got {forces[1]}"


if __name__ == "__main__":
    # To allow running the test direcly from the file
    sys.path.insert(0, str(Path(__file__).parent.parent))
    pytest.main(["-v", __file__])
