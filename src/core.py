import numpy as np
from pathlib import Path
import sys 

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.initialise_objects import initialise_objects
from src.utils.default_params import DEFAULT_PARAMS
from src.utils.initial_states import get_initial_states
from src.utils.import_csv import load_drone_trajectories
from src.simulation.physics import compute_net_forces, compute_forces, update_state
from src.visualizations.plot import animate_trajectories_3d, plot_drone_distances, plot_trajectory_errors, plot_drone_forces


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
    history["trajectory_errors"] = [[] for _ in drones]  # one list per drone
    history["forces"] = {"aero": [[] for _ in drones], "thrust": [[] for _ in drones]}

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
            # ── TRAJECTORY ERROR RECORDING ────────────────────────────────
        idx = min(int(round(t / trajectories_dt)), len(list(trajectories.values())[0]) - 1)
        for i, drone in enumerate(drones):
            target = trajectories[drone.id][idx]
            error = np.linalg.norm(drone.position - target)
            history["trajectory_errors"][i].append(error)
 

        # ---------------------------------- CONTROL UPDATES ----------------------------------------

        # Run low level DroneControllers (currently just returning a force)
        controller_forces = {}
        for drone in drones:
            thrust, record_thrust = drone.controller.compute_thrust(drone, trajectories, t, trajectories_dt)
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

        # ── FORCE RECORDING ───────────────────────────────────────────
        for i, drone in enumerate(drones):
            history["forces"]["aero"][i].append(forces[drone.id]["aero"].copy())
            history["forces"]["thrust"][i].append(list(record_thrust).copy())

        # Time update
        t += DEFAULT_PARAMS["simulation_dt"]
        # DEBUG
        # if not np.isnan(drone.position).any():

        #     print(f"drone positions at t={t:.2f}s: {[drone.position for drone in drones]} payload position: {payload.position}")

    # Post-simulation processing for visualization
    history["t"] = np.asarray(history["t"])
    history["drones"] = [np.asarray(traj) for traj in history["drones"]]
    history[-1] = np.asarray(history[-1])
    history["distances"] = [np.asarray(d) for d in history["distances"]]
    history["trajectory_errors"] = [np.asarray(e) for e in history["trajectory_errors"]]
    history["forces"]["aero"]   = [np.asarray(f) for f in history["forces"]["aero"]] # type: ignore
    history["forces"]["thrust"] = [np.asarray(f) for f in history["forces"]["thrust"]] # type: ignore

    # Nominal distances — computed once from trajectory data, resampled to sim time axis
    traj_t = np.linspace(t_start, t_end, len(list(trajectories.values())[0]))
    history["nominal_distances"] = []
    for i, j in pairs:
        nominal = np.array([
            np.linalg.norm(trajectories[i][k] - trajectories[j][k])
            for k in range(len(traj_t))
        ])
        history["nominal_distances"].append(np.interp(history["t"], traj_t, nominal))

    animate_trajectories_3d(history)
    plot_drone_distances(history)
    plot_trajectory_errors(history)
    plot_drone_forces(history)


if __name__ == "__main__":
    main()