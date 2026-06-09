import numpy as np
from pathlib import Path
import sys 

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.initialise_objects import initialise_objects
from src.utils.default_params import DEFAULT_PARAMS
from src.utils.initial_states import get_initial_states
from src.utils.import_csv import load_drone_trajectories
from src.simulation.physics import compute_net_forces, compute_forces, update_state
from src.visualizations.plot import animate_trajectories_3d, plot_drone_distances


def main():
    t_start, t_end, trajectories_dt, trajectories = load_drone_trajectories(DEFAULT_PARAMS["trajectories_path"])

    initial_states = get_initial_states(trajectories=trajectories, dt=trajectories_dt)
    drones, payload, cables = initialise_objects(initial_states)
    last_gust_t = None
    last_wind_vector = None

    # Main simulation loop
    history = {"t": [], "drones": [[] for _ in drones], -1: []}
    history["trajectories"] = trajectories
    # Pre-compute pair labels once
    drone_ids = [drone.id for drone in drones]
    pairs = [(drone_ids[i], drone_ids[j]) for i in range(len(drones)) for j in range(i+1, len(drones))]
    history["distance_pairs"] = pairs          # e.g. [(0,1), (0,2), (1,2)]
    history["distances"] = [[] for _ in pairs] # one list per pair

    t = t_start

    while t < t_end:

        # ---------------------------------- DATA RECORDING ----------------------------------------
        # Record state at current time
        history["t"].append(t)
        for i, drone in enumerate(drones):
            history["drones"][i].append(np.hstack((drone.position, drone.v)))
        history[-1].append(np.hstack((payload.position, payload.v))) 

            # ── DISTANCE RECORDING ────────────────────────────────────────
        for k, (i, j) in enumerate(pairs):
            dist = np.linalg.norm(drones[i].position - drones[j].position)
            history["distances"][k].append(dist)   
 

        # ---------------------------------- CONTROL UPDATES ----------------------------------------

        # Run low level DroneControllers (currently just returning a force)
        controller_forces = {}
        for drone in drones:
            thrust = drone.controller.compute_thrust(drone, trajectories, t, trajectories_dt)
            controller_forces[drone.id] = thrust

        # ---------------------------------- PHYSICS UPDATES ----------------------------------------

        forces, last_gust_t, last_wind_vector = compute_forces(drones, cables, payload, last_gust_t, last_wind_vector, t=t)
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
        # store distance between drones
        for i in range(len(drones)):
            for j in range(i+1, len(drones)):
                dist = np.linalg.norm(drones[i].position - drones[j].position)
                history["drones"][i][-1] = np.hstack((history["drones"][i][-1], dist))
                history["drones"][j][-1] = np.hstack((history["drones"][j][-1], dist))

        # Time update
        t += DEFAULT_PARAMS["simulation_dt"]

    history["t"] = np.asarray(history["t"])
    history["drones"] = [np.asarray(traj) for traj in history["drones"]]
    history[-1] = np.asarray(history[-1])
    history["distances"] = [np.asarray(d) for d in history["distances"]]
    # plot_gain_response(DEFAULT_PARAMS)
    # plot_radius_vs_time(history, R=DEFAULT_PARAMS["R"], L0=DEFAULT_PARAMS["L0"])
    animate_trajectories_3d(history)
    plot_drone_distances(history)


if __name__ == "__main__":
    main()