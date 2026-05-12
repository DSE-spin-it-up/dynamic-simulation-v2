import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.initial_states import get_initial_states
from src.utils.initialise_objects import initialise_objects
from src.utils.default_params import DEFAULT_PARAMS

def test_initial_positions():
    # Setup
    for n_drones in range(2, 10):
        initial_states = get_initial_states(n_drones, R=DEFAULT_PARAMS["R"], L0=DEFAULT_PARAMS["L0"], payload_pos=np.zeros(3))
        z_drone = DEFAULT_PARAMS["z_target"]
        expected_drone_positions = np.array([
            [DEFAULT_PARAMS["R"] * np.cos(2 * np.pi * i / n_drones), DEFAULT_PARAMS["R"] * np.sin(2 * np.pi * i / n_drones), z_drone]
            for i in range(n_drones)
        ])
        # Verify
        assert initial_states["payload"]["position"].shape == (3,)
        assert np.allclose(initial_states["payload"]["position"], np.zeros(3))
        for i in range(n_drones):
            assert initial_states[i]["position"].shape == (3,)
            assert np.allclose(initial_states[i]["position"], expected_drone_positions[i])

def test_initial_velocities():
    # Setup
    n_drones = 5
    initial_states = get_initial_states(n_drones, R=DEFAULT_PARAMS["R"], L0=DEFAULT_PARAMS["L0"], payload_pos=np.zeros(3))
    
    # Verify
    assert initial_states["payload"]["velocity"].shape == (3,)
    assert np.allclose(initial_states["payload"]["velocity"], np.zeros(3))
    for i in range(n_drones):
        assert initial_states[i]["velocity"].shape == (3,)
        assert np.allclose(initial_states[i]["velocity"], np.zeros(3))

def test_initialise_objects():
    # Setup
    n_drones = 4
    initial_states = get_initial_states(n_drones, R=DEFAULT_PARAMS["R"], L0=DEFAULT_PARAMS["L0"], payload_pos=np.zeros(3))
    
    # Act
    drones, payload, cables = initialise_objects(initial_states)
    
    # Verify
    assert len(drones) == n_drones
    assert all(isinstance(drones[i], type(drones[0])) for i in range(n_drones))
    assert isinstance(payload, type(payload))
    assert len(cables) == n_drones
    assert all(isinstance(cables[i], type(cables[0])) for i in range(n_drones))

if __name__ == "__main__":
    pytest.main(["-v", __file__])