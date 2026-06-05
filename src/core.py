import numpy as np
from pathlib import Path
import sys 

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.initialise_objects import initialise_objects
from src.utils.default_params import DEFAULT_PARAMS, VehicleParams, StateLimits
from src.utils.initial_states import get_cruise_initial_states, get_initial_states
from src.simulation.physics import compute_net_forces, compute_forces, update_state
from src.visualizations.plot import animate_trajectories_3d


def main():
    initial_states = get_initial_states()
    drones, payload, cables, trajectory_planner, mission_planner = initialise_objects(initial_states)

    # Integration term for PID controller
    integral_error = {drone.id: np.zeros(3) for drone in drones}

    # Main simulation loop
    history = {"t": [], "drones": [[] for _ in drones], -1: [], "projected_trajectories": [[]for _ in drones]}
    history["plan_time"] = []
    history["payload_ref"] = []
    t = DEFAULT_PARAMS["t_start"]
    n_loops_hold_waypoint = DEFAULT_PARAMS["opti_N_apply"] * (DEFAULT_PARAMS["opti_dt"] / DEFAULT_PARAMS["dt"])  # number of time steps to hold each planned waypoint

    n_sim_loops = 0 # counter to track how many time steps have been taken, used to determine when to update the planned trajectory
    planned_positions = None

    while t < DEFAULT_PARAMS["t_end"]:
        # ---------------------------------- DATA RECORDING ----------------------------------------
        # Record state at current time
        history["t"].append(t)
        for i, drone in enumerate(drones):
            history["drones"][i].append(np.hstack((drone.position, drone.v)))
        history[-1].append(np.hstack((payload.position, payload.v)))

        # ---------------------------------- MISSION PLANNING UPDATES ----------------------------------------

        mission_command = mission_planner.update(t, drones, payload, cables)

        # Run the planner after the previous trajectory N_apply has been used
        if planned_positions is None or n_sim_loops % (n_loops_hold_waypoint * DEFAULT_PARAMS["opti_N_apply"]) == 0:
            planned_positions, planned_time = trajectory_planner.calculate_traj_step(
                t,
                drones=drones,
                payload=payload,
                mission_phase=mission_command.phase,
            )
            if planned_positions is None:
                t = 100000
                break
            for drone_n in range(DEFAULT_PARAMS["n_drones"]):
                history["projected_trajectories"][drone_n].append(planned_positions[drone_n])
            history["plan_time"].append(t)
            history["payload_ref"].append(trajectory_planner.ref_window)

        # ---------------------------------- CONTROL UPDATES ----------------------------------------

        # Use the first waypoint of the current horizon for the whole hold window.
        controller_forces = {}

        for drone in drones:
            # PID controller to track the planned trajectory (V1)
            # find ref pos with the drone id and the window time
            k = int(round((t - history["plan_time"][-1]) / DEFAULT_PARAMS["dt"]))
            k = np.clip(k, 0, planned_positions[drone.id].shape[1] - 1)
            ref_state = planned_positions[drone.id][:, k]
            ref_pos = ref_state[3:6]
            error = ref_pos - drone.position
            integral_error[drone.id] += error * DEFAULT_PARAMS["dt"]
            force = DEFAULT_PARAMS["prop_gain"] * error - DEFAULT_PARAMS["deriv_gain"] * drone.v + DEFAULT_PARAMS["int_gain"] * integral_error[drone.id]  # PID control
            thrust = force
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
        n_sim_loops += 1


    history["t"] = np.asarray(history["t"])
    history["drones"] = [np.asarray(traj) for traj in history["drones"]]
    history[-1] = np.asarray(history[-1])

    # plot_gain_response(DEFAULT_PARAMS)
    # plot_radius_vs_time(history, R=DEFAULT_PARAMS["R"], L0=DEFAULT_PARAMS["L0"])
    animate_trajectories_3d(history, params=DEFAULT_PARAMS, stride=2)


if __name__ == "__main__":
    main()