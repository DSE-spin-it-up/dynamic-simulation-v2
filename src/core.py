import numpy as np
from pathlib import Path
import sys 

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.initialise_objects import initialise_objects
from src.utils.default_params import DEFAULT_PARAMS
from src.utils.initial_states import get_initial_states
from src.simulation.physics import compute_net_forces, compute_forces, update_state
from src.visualizations.plot import animate_trajectories_3d


def main():
    initial_states = get_initial_states(
        num_drones=DEFAULT_PARAMS["n_drones"],
        payload_pos=np.array([0.0, 0.0, 0.0]),
    )
    drones, payload, cables, trajectory_planner, mission_planner = initialise_objects(initial_states)

    # Integration term for PID controller
    integral_error = {drone.id: np.zeros(3) for drone in drones}

    # Main simulation loop
    history = {"t": [], "drones": [[] for _ in drones], -1: [], "projected_trajectories": [[]for _ in drones]}
    history["plan_time"] = []
    t = DEFAULT_PARAMS["t_start"]
    waypoints_hold = int(round(DEFAULT_PARAMS["opti_dt"] / DEFAULT_PARAMS["dt"])) * DEFAULT_PARAMS["opti_timepstep_N"]

    counter_waypoint = 0
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
        # Run the planner only after the previous trajectory has been used
        if planned_positions is None or counter_waypoint % waypoints_hold == 0:
            planned_positions, _ = trajectory_planner.calculate_traj_step(
                t,
                drones=drones,
                payload=payload,
                mission_phase=mission_command.phase,
            )
            for drone_n in range(DEFAULT_PARAMS["n_drones"]):
                history["projected_trajectories"][drone_n].append(planned_positions[drone_n])
            history["plan_time"].append(t)

        # ---------------------------------- CONTROL UPDATES ----------------------------------------

        # Use the first waypoint of the current horizon for the whole hold window.
        controller_forces = {}

        for drone in drones:
            # PID controller to track the planned trajectory (V1)
            ref_pos = planned_positions[drone.id][:, 0]
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
        counter_waypoint += 1


    history["t"] = np.asarray(history["t"])
    history["drones"] = [np.asarray(traj) for traj in history["drones"]]
    history[-1] = np.asarray(history[-1])

    # plot_gain_response(DEFAULT_PARAMS)
    # plot_radius_vs_time(history, R=DEFAULT_PARAMS["R"], L0=DEFAULT_PARAMS["L0"])
    animate_trajectories_3d(history, params=DEFAULT_PARAMS, stride=2)


if __name__ == "__main__":
    main()