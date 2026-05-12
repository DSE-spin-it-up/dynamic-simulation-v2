import numpy as np
from pathlib import Path
import sys 

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.initialise_objects import initialise_objects
from src.utils.default_params import DEFAULT_PARAMS
from src.utils.initial_states import get_initial_states
from src.simulation.physics import compute_net_forces, simulate
from src.visualizations.plot import animate_trajectories_3d, plot_radius_vs_time, plot_gain_response


def main():
    initial_states = get_initial_states(
        num_drones=DEFAULT_PARAMS["n_drones"],
        R=DEFAULT_PARAMS["R"],
        L0=DEFAULT_PARAMS["L0"],
        payload_pos=np.array([0.0, 0.0, 0.0]),
        z_target=DEFAULT_PARAMS["z_target"],
    )
    drones, payload, cables, high_level_controller = initialise_objects(initial_states)

    # Main simulation loop
    history = {"t": [], "drones": [[] for _ in drones], -1: []}
    t = DEFAULT_PARAMS["t_start"]
    while t < DEFAULT_PARAMS["t_end"]:
        # Record state at current time
        history["t"].append(t)
        for i, drone in enumerate(drones):
            history["drones"][i].append(np.hstack((drone.position, drone.v)))
        history[-1].append(np.hstack((payload.position, payload.v)))

        # Poll HighLevelController (currently empty)
        high_level_controller.update(t, drones, payload, cables)

        # Run low level DroneControllers (currently just returning a force)
        controller_forces = {}
        for drone in drones:
            thrust = drone.controller.update(t, drone, payload)
            controller_forces[drone.id] = thrust

        # Compute forces
        forces_snapshot = {id: {} for id in [drone.id for drone in drones]}
        forces_snapshot[-1] = {} # Payload forces accessed with ID -1

        # Add controller forces to the snapshot
        for drone_id, thrust in controller_forces.items():
            forces_snapshot[drone_id]["controller"] = thrust
        
        # For each cable, compute forces and apply to payload and drone
        forces_snapshot[-1]["cable_tension"] = np.zeros(3) # Initialize payload cable tension
        for cable in cables:
            # There is only one cable per drone, several for the payload
            force_payload, force_drone = cable.force_vectors()
            forces_snapshot[-1]["cable_tension"] += force_payload
            forces_snapshot[cable.drone.id]["cable_tension"] = force_drone

        # Aero forces for drones and payload
        for drone in drones:
            aero_force = drone.compute_aero_forces()
            forces_snapshot[drone.id]["aero"] = aero_force
        aero_force_payload = payload.compute_aero_forces()
        forces_snapshot[-1]["aero"] = aero_force_payload

        # Gravity forces for drones and payload
        for drone in drones:
            gravity_force = drone.compute_gravity_force()
            forces_snapshot[drone.id]["gravity"] = gravity_force
        gravity_force_payload = payload.compute_gravity_force()
        forces_snapshot[-1]["gravity"] = gravity_force_payload

        # Apply forces and update states

        # Sum forces
        net_forces = compute_net_forces(forces_snapshot)
        # Apply net forces to update drone and payload states
        for drone in drones:
            drone.apply_force(net_forces[drone.id])
        payload.apply_force(net_forces[-1])

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