import numpy as np
from pathlib import Path
import sys 

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.initialise_objects import initialise_objects
from src.utils.default_params import DEFAULT_PARAMS
from src.utils.initial_states import get_initial_states
from src.utils.import_csv import load_drone_trajectories
from src.simulation.physics import compute_net_forces, compute_forces, update_state, compute_moments, compute_net_moments
from src.visualizations.plot import animate_trajectories_3d


def main():
    t_start, t_end, trajectories_dt, trajectories = load_drone_trajectories(DEFAULT_PARAMS["trajectories_path"])

    initial_states = get_initial_states(trajectories=trajectories, dt=trajectories_dt)
    drones, payload, cables = initialise_objects(initial_states)

    # Main simulation loop
    history = {"t": [], "drones": [[] for _ in drones], -1: []}
    history["trajectories"] = trajectories
    t = t_start

    while t < t_end:

        # ---------------------------------- DATA RECORDING ----------------------------------------
        # Record state at current time
        history["t"].append(t)
        for i, drone in enumerate(drones):
            history["drones"][i].append(np.hstack((drone.position, drone.v)))
        history[-1].append(np.hstack((payload.position, payload.v)))    
 

        # ---------------------------------- CONTROL UPDATES ----------------------------------------

        # Run low level DroneControllers (currently just returning a force)
        controller_forces = {}
        controller_moments = {}
        for drone in drones:
            thrust = drone.controller.compute_thrust(drone, trajectories, t, trajectories_dt)
            controller_forces[drone.id] = thrust
            controller_moments[drone.id] = drone.controller.compute_moments(drone, trajectories, t, trajectories_dt)

        # ---------------------------------- PHYSICS UPDATES ----------------------------------------

        forces = compute_forces(drones, cables, payload)
        moments = compute_moments(drones, cables, payload) # currently returns zero moments
        # for V1, add controller forces
        for drone in drones:
            forces[drone.id]["thrust"] = controller_forces[drone.id]
            moments[drone.id]["control"] = controller_moments[drone.id]
        # currently separated because V1 controller returns a force, and there is no aero modelling
        net_forces = compute_net_forces(forces)
        net_moments = compute_net_moments(moments) # currently zero moments

        # Apply forces to update drone and payload states
        for drone in drones:
            update_state(drone, net_forces[drone.id], net_moments[drone.id])
        update_state(payload, net_forces[-1], net_moments[-1])

        # -------------------------------------------------- VISUALIZATION UPDATES ----------------------------------------

        # Time update
        t += DEFAULT_PARAMS["simulation_dt"]

    history["t"] = np.asarray(history["t"])
    history["drones"] = [np.asarray(traj) for traj in history["drones"]]
    history[-1] = np.asarray(history[-1])
    # plot_gain_response(DEFAULT_PARAMS)
    # plot_radius_vs_time(history, R=DEFAULT_PARAMS["R"], L0=DEFAULT_PARAMS["L0"])
    animate_trajectories_3d(history)


if __name__ == "__main__":
    main()