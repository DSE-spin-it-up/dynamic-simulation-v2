import numpy as np
from pathlib import Path
import sys 

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.initialise_objects import initialise_objects
from src.utils.default_params import DEFAULT_PARAMS
from src.utils.initial_states import get_initial_states
from src.simulation.physics import compute_net_forces, compute_forces, update_state
from src.visualizations.plot import animate_trajectories_3d, plot_radius_vs_time, plot_gain_response


def main():
    initial_states = get_initial_states(
        num_drones=DEFAULT_PARAMS["n_drones"],
        payload_pos=np.array([0.0, 0.0, 0.0]),
    )
    drones, payload, cables, trajectory_planner, mission_planner = initialise_objects(initial_states)

    # Main simulation loop
    history = {"t": [], "drones": [[] for _ in drones], -1: []}
    t = DEFAULT_PARAMS["t_start"]
    while t < DEFAULT_PARAMS["t_end"]:
        # ---------------------------------- DATA RECORDING ----------------------------------------
        # Record state at current time
        history["t"].append(t)
        for i, drone in enumerate(drones):
            history["drones"][i].append(np.hstack((drone.position, drone.v)))
        history[-1].append(np.hstack((payload.position, payload.v)))

        # ---------------------------------- MISSION PLANNING UPDATES ----------------------------------------

        # mission_command = mission_planner.update(t, drones, payload, cables)
        trajectory_planner.calculate_traj_step(t, drones=drones, payload=payload, mission_phase=0)

        # ---------------------------------- CONTROL UPDATES ----------------------------------------

        # Run low level DroneControllers (currently just returning a force)
        controller_forces = {}
        for drone in drones:
            thrust = drone.controller.compute_thrust(drone, payload)
            controller_forces[drone.id] = thrust

        # ---------------------------------- PHYSICS UPDATES ----------------------------------------
        # Forces

        forces = compute_forces(drones, cables, payload)
        # for V1, add controller forces
        for drone in drones:
            forces[drone.id]["thrust"] = controller_forces[drone.id]
        # currently separated because V1 controller returns a force, and there is no aero modelling
        net_forces = compute_net_forces(forces)

        # Apply forces to update drone and payload states
        for drone in drones:
            update_state(drone, net_forces[drone.id])
        update_state(payload, net_forces[-1])

        # -------------------------------------------------- VISUALIZATION UPDATES ----------------------------------------

        # Real time plotting

        # Time update
        t += DEFAULT_PARAMS["dt"]

    history["t"] = np.asarray(history["t"])
    history["drones"] = [np.asarray(traj) for traj in history["drones"]]
    history[-1] = np.asarray(history[-1])

    plot_gain_response(DEFAULT_PARAMS)
    plot_radius_vs_time(history, R=DEFAULT_PARAMS["R"], L0=DEFAULT_PARAMS["L0"])
    animate_trajectories_3d(history, params=DEFAULT_PARAMS)


if __name__ == "__main__":
    main()