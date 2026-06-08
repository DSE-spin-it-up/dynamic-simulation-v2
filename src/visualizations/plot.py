import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from typing import cast, Any
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R


def plot_trajectories(history):
    """
    2-D top-down view of drone and payload trajectories in the x-y horizontal plane.

    Parameters
    ----------
    history : dict returned by simulate()
        Keys: 't', 'drones' (list of N_times x 6 arrays), 'payload' (N_times x 6 array)
    """
    n_drones = len(history["drones"])
    colors = plt.get_cmap('tab10')(np.linspace(0, 0.9, n_drones))

    _, ax = plt.subplots(figsize=(7, 7))

    for i, (traj, color) in enumerate(zip(history["drones"], colors)):
        x, y = traj[:, 0], traj[:, 1]
        ax.plot(x, y, color=color, linewidth=0.8, label=f"Drone {i}")
        ax.plot(x[0], y[0], "o", color=color, markersize=6)   # start
        ax.plot(x[-1], y[-1], "s", color=color, markersize=6)  # end

    px, py = history[-1][:, 0], history[-1][:, 1]
    ax.plot(px, py, "k-", linewidth=1.5, label="Payload")
    ax.plot(px[0], py[0], "ko", markersize=8)
    ax.plot(px[-1], py[-1], "ks", markersize=8)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Drone trajectories — horizontal plane (x-y)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def _plane_polys(pos, R, size):
    """
    pos : (3,)
    R   : (3,3) body->world rotation matrix
    """

    fuse = [(1.0, 0, 0), (0.0, 0.12, 0), (-0.9, 0, 0), (0.0, -0.12, 0)]
    wing = [(0.25, -1.0, 0), (0.45, 0, 0), (0.25, 1.0, 0),
            (-0.05, 1.0, 0), (-0.25, 0, 0), (-0.05, -1.0, 0)]
    tail = [(-0.6, -0.45, 0), (-0.5, 0, 0), (-0.6, 0.45, 0),
            (-0.8, 0.45, 0), (-0.75, 0, 0), (-0.8, -0.45, 0)]
    fin  = [(-0.55, 0, 0), (-0.5, 0, 0.45), (-0.85, 0, 0.45), (-0.9, 0, 0)]

    polys = []
    pos = np.asarray(pos)

    for loc in (fuse, wing, tail, fin):
        pts = np.asarray(loc).T
        polys.append((pos[:, None] + size * (R @ pts)).T)

    return polys

def quat_to_rot(q):
    # q is [x, y, z, w]
    return R.from_quat(q).as_matrix()

def animate_trajectories_3d(
    history,
    stride: int = 10,
    trail_length: int = 10,
    params: dict = {},
    window_size: float = 30.0,
):
    """
    3D animation with:
      - UAV aircraft glyphs oriented from recorded quaternions
      - Payload marker
      - Cables
      - Short UAV trails
      - Sliding camera window

    Expects each drone history row to contain:
        [x, y, z, vx, vy, vz, qx, qy, qz, qw]  (Scalar-last format from simulation)
    """

    from typing import Any, cast
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import animation
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.spatial.transform import Rotation as R

    n_drones = len(history["drones"])
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.9, n_drones))

    # ------------------------------------------------------------------
    # Extract arrays once
    # ------------------------------------------------------------------
    drones_xyz = [
        np.ascontiguousarray(history["drones"][i][:, :3])
        for i in range(n_drones)
    ]

    drone_quats = [
        np.ascontiguousarray(history["drones"][i][:, 6:10])
        for i in range(n_drones)
    ]

    payload_xyz = np.ascontiguousarray(history[-1][:, :3])
    has_phases = "phase" in history

    # ------------------------------------------------------------------
    # Figure Configuration
    # ------------------------------------------------------------------
    try:
        plt.switch_backend("TkAgg")
    except Exception:
        pass

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Tracking Trajectory")
    ax.grid(True)

    # ------------------------------------------------------------------
    # Initialize Artists
    # ------------------------------------------------------------------
    drone_trails = []
    drone_markers = []
    cable_lines = []
    plane_artists = []

    plane_size = 3.0  # meters

    for i, color in enumerate(colors):
        trail = ax.plot([], [], [], linewidth=1.5, color=color, antialiased=True)[0]
        marker = ax.plot([], [], [], "o", color=color, markersize=4, antialiased=True, label=f"Drone {i}")[0]
        cable = ax.plot([], [], [], "-", linewidth=1.2, color=color, antialiased=True)[0]
        
        plane = Poly3DCollection(
            [],
            facecolor=color,
            edgecolor="k",
            linewidths=0.3,
            alpha=0.95,
        )
        ax.add_collection3d(plane)

        if "trajectories" in history:
            ax.plot(
                history["trajectories"][i][:, 0],
                history["trajectories"][i][:, 1],
                history["trajectories"][i][:, 2],
                "--",
                linewidth=0.5,
                color=color,
                alpha=0.3,
            )

        drone_trails.append(trail)
        drone_markers.append(marker)
        cable_lines.append(cable)
        plane_artists.append(plane)

    payload_marker = ax.plot([], [], [], "ks", markersize=7, label="Payload")[0]

    time_text = ax.text2D(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        fontsize=10,
        family="monospace",
    )

    ax.legend(loc="upper right")

    # ------------------------------------------------------------------
    # Animation update loop
    # ------------------------------------------------------------------
    n_frames = (len(history["t"]) - 1) // stride + 1

    def _update(frame):
        k = min(frame * stride, len(history["t"]) - 1)
        trail_start = max(0, k - trail_length)

        px, py, pz = payload_xyz[k]
        current_pts = [[px, py, pz]]

        for i in range(n_drones):
            xyz = drones_xyz[i]
            quat = drone_quats[i]

            pos = xyz[k]
            q = quat[k]  # [qx, qy, qz, qw] from simulation data

            current_pts.append(pos)

            # 1. Update Trails
            drone_trails[i].set_data(xyz[trail_start:k + 1, 0], xyz[trail_start:k + 1, 1])
            cast(Any, drone_trails[i]).set_3d_properties(xyz[trail_start:k + 1, 2])

            # 2. Update Drone Markers
            drone_markers[i].set_data([pos[0]], [pos[1]])
            cast(Any, drone_markers[i]).set_3d_properties([pos[2]])

            # 3. Update Cable Lines
            cable_lines[i].set_data([px, pos[0]], [py, pos[1]])
            cast(Any, cable_lines[i]).set_3d_properties([pz, pos[2]])

            # 4. Update Aircraft Glyph (With Pitch and Yaw Visual Fix)
            # We invert the y (pitch) and z (yaw) elements of the vector part 
            # to bridge the gap between Matplotlib's 3D axes projection and the simulation frame.
            q_visual = np.array([q[0], -q[1], -q[2], q[3]])
            R_mat = R.from_quat(q_visual).as_matrix()

            polys = _plane_polys(
                pos=pos,
                R=R_mat,
                size=plane_size,
            )
            plane_artists[i].set_verts(polys)

        # 5. Update Payload Marker
        payload_marker.set_data([px], [py])
        cast(Any, payload_marker).set_3d_properties([pz])

        # 6. Sliding Camera Window Configuration
        current_pts = np.asarray(current_pts)
        center = current_pts.mean(axis=0)
        half_win = window_size / 2.0

        ax.set_xlim(center[0] - half_win, center[0] + half_win)
        ax.set_ylim(center[1] - half_win, center[1] + half_win)
        ax.set_zlim(max(0.0, center[2] - half_win), center[2] + half_win)

        try:
            ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass

        label = f"t = {history['t'][k]:6.2f} s"
        if has_phases:
            label += f" | {history['phase'][k]}"
        time_text.set_text(label)

    # ------------------------------------------------------------------
    # Timing and Execution
    # ------------------------------------------------------------------
    from src.utils.default_params import DEFAULT_PARAMS

    dt = DEFAULT_PARAMS.get("simulation_dt", 0.01)
    interval_ms = max(1, int(round(stride * dt * 1000)))

    anim = animation.FuncAnimation(
        fig,
        _update,  # type: ignore
        frames=n_frames,
        interval=interval_ms,
        blit=False,
        cache_frame_data=False,
    )

    plt.tight_layout()
    plt.show()

    return anim