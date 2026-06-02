import numpy as np
from pathlib import Path
import sys 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.initialise_objects import initialise_objects
from src.utils.default_params import DEFAULT_PARAMS
from src.utils.initial_states import get_initial_states
from src.simulation.physics import compute_net_forces, compute_forces, update_state
from src.visualizations.plot import animate_trajectories_3d, plot_radius_vs_time, plot_gain_response


def main():
    initial_states = get_initial_states(
        num_drones=DEFAULT_PARAMS["n_drones"],
        payload_pos=np.array([0.0, 0.0, 95.0]),
    )
    drones, payload, cables, path_planner = initialise_objects(initial_states)

    # Main simulation loop
    history = {"t": [], "drones": [[] for _ in drones], -1: []}
    t = DEFAULT_PARAMS["t_start"]

    # Read in optimal trajectories
    files = ["src/pos_sol_0.csv", "src/pos_sol_1.csv", "src/pos_sol_2.csv"]

    drones_pos = []
    for f in files:
        drones_pos.append(np.loadtxt(f, delimiter=",", skiprows=40))
    drones_pos = np.array(drones_pos)

    files2 = ["src/V_sol_0.csv", "src/V_sol_1.csv", "src/V_sol_2.csv"]

    drones_vel = []
    for f in files2:
        drones_vel.append(np.loadtxt(f, delimiter=",", skiprows=40))
    drones_vel = np.array(drones_vel)

    print(drones_vel.shape)
    
    counter_waypoint = 0
    while t < 600-40:
        # ---------------------------------- DATA RECORDING ----------------------------------------
        # Record state at current time
        history["t"].append(t)
        for i, drone in enumerate(drones):
            history["drones"][i].append(np.hstack((drone.position, drone.v)))
        history[-1].append(np.hstack((payload.position, payload.v)))

        # ---------------------------------- CONTROLLER UPDATES ----------------------------------------

        # Poll TrajectoryPlanner (currently empty)
        path_planner.update(t, drones, payload, cables)

        # Run low level DroneControllers (currently just returning a force)
        controller_forces = {}
        
        i = 0
        while i < len(drones):
            force = 5*(drones_pos[i][t] - history["drones"][i][-1][:3]) + 0.1*(drones_vel[i][t] - history["drones"][i][-1][3:])
            thrust = force
            controller_forces[i] = thrust
            i += 1
        
        # for drone in drones:
        #     thrust = drone.controller.compute_thrust(drone, payload)
        #     controller_forces[drone.id] = thrust

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
        t += 1

    history["t"] = np.asarray(history["t"])
    history["drones"] = [np.asarray(traj) for traj in history["drones"]]
    history[-1] = np.asarray(history[-1])
    # plot_gain_response(DEFAULT_PARAMS)
    # plot_radius_vs_time(history, R=DEFAULT_PARAMS["R"], L0=DEFAULT_PARAMS["L0"])
    animate_trajectories_3d(history, params=DEFAULT_PARAMS, stride=2)


    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    n_drones = len(drones_pos)

    # -----------------------------
    # Drones: reference vs actual
    # -----------------------------
    for i in range(n_drones):

        ref = drones_pos[i]                 # (T, 3)
        act = history["drones"][i][:, :3]   # (T, 3)

        ax.plot(ref[:, 0], ref[:, 1], ref[:, 2],
                "--", color="blue", alpha=0.6,
                label="Reference" if i == 0 else "")

        ax.plot(act[:, 0], act[:, 1], act[:, 2],
                "-", color="red", alpha=0.7,
                label="Actual" if i == 0 else "")

        # START MARKERS
        ax.scatter(*ref[0], color="blue", s=40, marker="o")
        ax.scatter(*act[0], color="red", s=40, marker="o")
    # -----------------------------
    # Payload trajectory
    # -----------------------------
    payload = history[-1][:, :3]

    ax.plot(payload[:, 0], payload[:, 1], payload[:, 2],
            "k-", linewidth=2.0, label="Payload")

    # Optional start/end markers
    ax.scatter(*payload[0], color="k", s=40)
    ax.scatter(*payload[-1], color="k", s=80, marker="x")

    # -----------------------------
    # Labels
    # -----------------------------
    ax.set_title("Reference vs Actual vs Payload Trajectories")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.legend()
    plt.show()

    fig = plt.figure(figsize=(10, 8))

    plt.subplot(3, 1, 1)
    plt.plot(history["t"], drones_pos[0][:, 0])
    plt.plot(history["t"], history["drones"][i][:, :3][:, 0])
    plt.xlabel('Time')
    plt.ylabel('x pos')

    plt.subplot(3, 1, 2)
    plt.plot(history["t"], drones_pos[0][:, 1])
    plt.plot(history["t"], history["drones"][i][:, :3][:, 1])
    plt.xlabel('Time')
    plt.ylabel('y pos')

    plt.subplot(3, 1, 3)
    plt.plot(history["t"], drones_pos[0][:, 2])
    plt.plot(history["t"], history["drones"][i][:, :3][:, 2])
    plt.xlabel('Time')
    plt.ylabel('z pos')

    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(10, 8))

    plt.subplot(3, 1, 1)
    plt.plot(history["t"], payload[:, 0])
    plt.xlabel('Time')
    plt.ylabel('x pos')

    plt.subplot(3, 1, 2)
    plt.plot(history["t"], payload[:, 1])
    plt.xlabel('Time')
    plt.ylabel('y pos')

    plt.subplot(3, 1, 3)
    plt.plot(history["t"], payload[:, 2])
    plt.xlabel('Time')
    plt.ylabel('z pos')

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()