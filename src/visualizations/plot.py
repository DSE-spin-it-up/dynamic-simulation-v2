import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import os


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

def plot_radius_vs_time(history, R: float | None = None , L0: float | None = None):
    """
    Plot orbit radius r(t) = distance from each drone to the payload over time.

    Parameters
    ----------
    history : dict returned by simulate()
    R       : target orbit radius (drawn as a dashed reference line if provided)
    L0      : cable rest length (drawn as a dotted reference line if provided)
    """
    n_drones = len(history["drones"])
    colors = plt.get_cmap('tab10')(np.linspace(0, 0.9, n_drones))
    t = history["t"]
    px = history[-1][:, 0]
    py = history[-1][:, 1]

    _, ax = plt.subplots(figsize=(9, 4))

    for i, (traj, color) in enumerate(zip(history["drones"], colors)):
        r = np.hypot(traj[:, 0] - px, traj[:, 1] - py)
        ax.plot(t, r, color=color, linewidth=1.0, label=f"Drone {i}")

    if R is not None:
        ax.axhline(R, color="k", linestyle="--", linewidth=1.0, label=f"R = {R} m")
    if L0 is not None:
        ax.axhline(L0, color="gray", linestyle=":", linewidth=1.0, label=f"L0 = {L0} m")

    ax.set_xlabel("t [s]")
    ax.set_ylabel("r [m]")
    ax.set_title("Orbit radius vs time")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def animate_trajectories(history, stride: int = 10, trail_length: int = 50):
    """
    Animate drones orbiting the payload with cable lines in the x-y plane.

    Parameters
    ----------
    history      : dict returned by simulate()
    stride       : render every Nth timestep (controls playback speed)
    trail_length : number of past frames shown as a fading tail per drone
    """
    n_drones = len(history["drones"])
    colors = plt.get_cmap('tab10')(np.linspace(0, 0.9, n_drones))

    drones_x = [history["drones"][i][:, 0] for i in range(n_drones)]
    drones_y = [history["drones"][i][:, 1] for i in range(n_drones)]
    px = history[-1][:, 0]
    py = history[-1][:, 1]

    all_x = np.concatenate(drones_x + [px])
    all_y = np.concatenate(drones_y + [py])
    pad = max(np.ptp(all_x), np.ptp(all_y)) * 0.2 + 0.5
    xlim = (all_x.min() - pad, all_x.max() + pad)
    ylim = (all_y.min() - pad, all_y.max() + pad)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Drone simulation — horizontal plane (x-y)")
    ax.grid(True, alpha=0.3)

    drone_trails = []
    drone_markers = []
    cable_lines = []

    for i, color in enumerate(colors):
        (trail,) = ax.plot([], [], color=color, linewidth=0.8, alpha=0.3)
        (marker,) = ax.plot([], [], "o", color=color, markersize=10, label=f"Drone {i}")
        (cable,) = ax.plot([], [], "-", color=color, linewidth=1.5, alpha=0.7)
        drone_trails.append(trail)
        drone_markers.append(marker)
        cable_lines.append(cable)

    (payload_marker,) = ax.plot([], [], "ks", markersize=12, label="Payload")
    time_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=10)
    ax.legend(loc="upper right", fontsize=8)

    n_frames = (len(history["t"]) - 1) // stride + 1

    def _update(frame):
        k = min(frame * stride, len(history["t"]) - 1)
        trail_start = max(0, k - trail_length)

        for i in range(n_drones):
            drone_trails[i].set_data(drones_x[i][trail_start : k + 1], drones_y[i][trail_start : k + 1])
            drone_markers[i].set_data([drones_x[i][k]], [drones_y[i][k]])
            cable_lines[i].set_data([px[k], drones_x[i][k]], [py[k], drones_y[i][k]])

        payload_marker.set_data([px[k]], [py[k]])
        time_text.set_text(f"t = {history['t'][k]:.2f} s")
        return drone_trails + drone_markers + cable_lines + [payload_marker, time_text]

    anim = animation.FuncAnimation(
        fig, _update, frames=n_frames, interval=30, blit=True
    )
    plt.tight_layout()
    plt.show()
    return anim


def animate_trajectories_3d(
    history,
    stride: int = 10,
    trail_length: int = 10,
    params: dict = {},
):
    """
    Real-time 3D animation with sliding window camera and equal axis scaling.
    """

    from matplotlib import animation
    import matplotlib.pyplot as plt
    import numpy as np
    from typing import Any, cast

    n_drones = len(history["drones"])
    colors = plt.get_cmap('tab10')(np.linspace(0, 0.9, n_drones))

    drones_xyz = [
        np.ascontiguousarray(history["drones"][i][:, :3])
        for i in range(n_drones)
    ]
    payload_xyz = np.ascontiguousarray(history[-1][:, :3])

    has_phases = "phase" in history

    # -----------------------------
    # Global bounds (for aspect hint)
    # -----------------------------
    all_pos = np.vstack(drones_xyz + [payload_xyz])
    x_min, x_max = all_pos[:, 0].min(), all_pos[:, 0].max()
    y_min, y_max = all_pos[:, 1].min(), all_pos[:, 1].max()
    z_min, z_max = all_pos[:, 2].min(), all_pos[:, 2].max()

    plt.switch_backend('TkAgg')

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Spin it up!")
    ax.grid(True)

    # initial box aspect (will be re-applied every frame for stability)
    try:
        ax.set_box_aspect([
            x_max - x_min,
            y_max - y_min,
            z_max - z_min + 1e-6,
        ])
    except Exception:
        pass

    # optional: reduce perspective distortion (recommended for control sims)
    try:
        ax.set_proj_type('ortho')
    except Exception:
        pass

    # -----------------------------
    # Artists
    # -----------------------------
    drone_trails = []
    drone_markers = []
    cable_lines = []

    for i, color in enumerate(colors):
        trail = ax.plot([], [], [], linewidth=1.0, color=color)[0]
        marker = ax.plot([], [], [], "o", color=color, markersize=6, label=f"Drone {i}")[0]
        cable = ax.plot([], [], [], "-", linewidth=1.0, color=color)[0]

        ax.plot(
            history["trajectories"][i][:, 0],
            history["trajectories"][i][:, 1],
            history["trajectories"][i][:, 2],
            "--", linewidth=0.5, color=color, alpha=0.5,
        )

        drone_trails.append(trail)
        drone_markers.append(marker)
        cable_lines.append(cable)

    payload_marker = ax.plot([], [], [], "ks", markersize=8, label="Payload")[0]
    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    ax.legend(loc="upper right")

    all_artists = tuple(
        drone_trails + drone_markers + cable_lines + [payload_marker, time_text]
    )

    # -----------------------------
    # Init
    # -----------------------------
    def _init():
        for t in drone_trails:
            t.set_data([], [])
            cast(Any, t).set_3d_properties([])
        for m in drone_markers:
            m.set_data([], [])
            cast(Any, m).set_3d_properties([])
        for c in cable_lines:
            c.set_data([], [])
            cast(Any, c).set_3d_properties([])

        payload_marker.set_data([], [])
        cast(Any, payload_marker).set_3d_properties([])
        time_text.set_text("")
        return all_artists

    # -----------------------------
    # Update
    # -----------------------------
    n_frames = (len(history["t"]) - 1) // stride + 1
    window_size = params.get("window_size", 30.0)
    half_win = window_size / 2.0

    def _update(frame):
        k = min(frame * stride, len(history["t"]) - 1)
        trail_start = max(0, k - trail_length)

        px, py, pz = payload_xyz[k]

        # -----------------------------
        # drones
        # -----------------------------
        for i in range(n_drones):
            xyz = drones_xyz[i]

            drone_trails[i].set_data(
                xyz[trail_start:k+1, 0],
                xyz[trail_start:k+1, 1],
            )
            cast(Any, drone_trails[i]).set_3d_properties(
                xyz[trail_start:k+1, 2]
            )

            drone_markers[i].set_data([xyz[k, 0]], [xyz[k, 1]])
            cast(Any, drone_markers[i]).set_3d_properties([xyz[k, 2]])

            cable_lines[i].set_data([px, xyz[k, 0]], [py, xyz[k, 1]])
            cast(Any, cable_lines[i]).set_3d_properties([pz, xyz[k, 2]])

        # payload
        payload_marker.set_data([px], [py])
        cast(Any, payload_marker).set_3d_properties([pz])

        # -----------------------------
        # SLIDING WINDOW + EQUAL SCALE
        # -----------------------------
        cx, cy, cz = px, py, pz

        xmin, xmax = cx - half_win, cx + half_win
        ymin, ymax = cy - half_win, cy + half_win
        zmin, zmax = cz - half_win, cz + half_win

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)

        # enforce identical scaling in all axes (critical line)
        ax.set_box_aspect((1, 1, 1))

        # -----------------------------
        # time label
        # -----------------------------
        label = f"t = {history['t'][k]:6.2f} s"
        if has_phases:
            label += f" | {history['phase'][k]}"
        time_text.set_text(label)

        return all_artists

    # -----------------------------
    # animation
    # -----------------------------
    from src.utils.default_params import DEFAULT_PARAMS
    dt = DEFAULT_PARAMS.get("dt", 0.01)
    interval_ms = max(1, int(round(stride * dt * 1000)))

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=n_frames,
        init_func=_init,
        interval=interval_ms,
        blit=False,   # required for dynamic axis updates
        cache_frame_data=False,
    )

    plt.tight_layout()
    plt.show()

    return anim


def plot_drone_distances(history: dict, output_path: str = "output/drone_distances.png") -> None:
    t                 = history["t"]
    distances         = history["distances"]
    nominal_distances = history["nominal_distances"]
    pairs             = history["distance_pairs"]
    n                 = len(pairs)

    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for k, (ax, (i, j)) in enumerate(zip(axes, pairs)):
        color = f"C{k}"
        ax.plot(t, distances[k],          linewidth=1.5, color=color, label="Simulated")
        ax.plot(t, nominal_distances[k],  linewidth=1.0, color=color, label="Nominal",
                linestyle="--", alpha=0.7)

        min_sim     = np.min(distances[k])
        min_nominal = np.min(nominal_distances[k])

        textstr = f"Min simulated:  {min_sim:.2f} m\nMin nominal:      {min_nominal:.2f} m"
        ax.text(
            0.02, 0.97, textstr,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="grey", alpha=0.8),
        )

        ax.set_ylabel("Distance [m]")
        ax.set_title(f"Drone {i} — Drone {j}")
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Inter-drone distances: simulated vs nominal", y=1.01)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved drone distance plot to {output_path}")

def plot_trajectory_errors(history: dict, output_path: str = "output/trajectory_errors.png") -> None:
    t      = history["t"]
    errors = history["trajectory_errors"]

    dt     = float(t[1] - t[0])
    window = max(1, int(1.0 / dt))
    kernel = np.ones(window) / window

    fig, ax = plt.subplots(figsize=(10, 4))

    for i, error in enumerate(errors):
        smoothed = np.convolve(error, kernel, mode="same")
        ax.plot(t, smoothed, linewidth=1.5, color=f"C{i}", label=f"Drone {i}")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position error [m]")
    ax.set_title("Trajectory tracking error per drone")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved trajectory error plot to {output_path}")