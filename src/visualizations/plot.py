import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from typing import cast, Any


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
    Faster 3D animation version.

    Main optimisations
    ------------------
    - No dynamic axis updates
    - No plot_surface()
    - Shorter trails
    - Larger stride
    - No transparency
    - Fewer redraw-heavy operations
    """

    from matplotlib import animation
    import matplotlib.pyplot as plt
    import numpy as np

    n_drones = len(history["drones"])
    colors = plt.get_cmap('tab10')(np.linspace(0, 0.9, n_drones))

    drones_xyz = [history["drones"][i][:, :3] for i in range(n_drones)]
    payload_xyz = history[-1][:, :3]

    has_phases = "phase" in history

    # ------------------------------------------------------------------
    # Axis limits
    # ------------------------------------------------------------------

    all_pos = np.vstack(drones_xyz + [payload_xyz])

    x_min, x_max = np.min(all_pos[:, 0]), np.max(all_pos[:, 0])
    y_min, y_max = np.min(all_pos[:, 1]), np.max(all_pos[:, 1])
    z_min, z_max = np.min(all_pos[:, 2]), np.max(all_pos[:, 2])

    pad = 1.0

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_zlim(min(0, z_min - pad), z_max + pad)

    try:
        ax.set_box_aspect(
            [
                x_max - x_min,
                y_max - y_min,
                z_max - z_min + 1e-6,
            ]
        )
    except AttributeError:
        pass

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    ax.set_title("Spin it up!")

    # Keep grid simple
    ax.grid(True)

    # ------------------------------------------------------------------
    # Animated artists
    # ------------------------------------------------------------------

    drone_trails = []
    drone_markers = []
    cable_lines = []
    trajectory_lines = []

    for i, color in enumerate(colors):

        trail = ax.plot(
            [],
            [],
            [],
            linewidth=1.0,
            color=color,
            antialiased=False,
        )[0]

        marker = ax.plot(
            [],
            [],
            [],
            "o",
            color=color,
            markersize=6,
            antialiased=False,
            label=f"Drone {i}",
        )[0]

        cable = ax.plot(
            [],
            [],
            [],
            "-",
            linewidth=1.0,
            color=color,
            antialiased=False,
        )[0]

        traj = ax.plot(
            history["trajectories"][i][:, 0],
            history["trajectories"][i][:, 1],
            history["trajectories"][i][:, 2],
            "--",
            linewidth=0.5,
            color=color,
            alpha=0.5,
            label=f"Drone {i} trajectory",
        )[0]

        drone_trails.append(trail)
        drone_markers.append(marker)
        cable_lines.append(cable)
        trajectory_lines.append(traj)

    payload_marker = ax.plot(
        [],
        [],
        [],
        "ks",
        markersize=8,
        label="Payload",
    )[0]

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
    # Animation update
    # ------------------------------------------------------------------

    n_frames = (len(history["t"]) - 1) // stride + 1

    def _update(frame):

        k = min(frame * stride, len(history["t"]) - 1)

        trail_start = max(0, k - trail_length)

        px, py, pz = payload_xyz[k]

        for i in range(n_drones):

            xyz = drones_xyz[i]

            xs = xyz[trail_start : k + 1, 0]
            ys = xyz[trail_start : k + 1, 1]
            zs = xyz[trail_start : k + 1, 2]

            # Trail
            drone_trails[i].set_data(xs, ys)
            cast(Any, drone_trails[i]).set_3d_properties(zs)

            # Marker
            drone_markers[i].set_data([xyz[k, 0]], [xyz[k, 1]])
            cast(Any, drone_markers[i]).set_3d_properties([xyz[k, 2]])

            # Cable
            cable_lines[i].set_data(
                [px, xyz[k, 0]],
                [py, xyz[k, 1]],
            )

            cast(Any, cable_lines[i]).set_3d_properties(
                [pz, xyz[k, 2]]
            )

        # Payload
        payload_marker.set_data([px], [py])
        cast(Any, payload_marker).set_3d_properties([pz])

        # Time label
        label = f"t = {history['t'][k]:6.2f} s"

        if has_phases:
            label += f" | {history['phase'][k]}"

        time_text.set_text(label)

        return (
            drone_trails
            + drone_markers
            + cable_lines
            + trajectory_lines
            + [payload_marker, time_text]
        )


    from src.utils.default_params import DEFAULT_PARAMS
    dt = DEFAULT_PARAMS.get("dt", 0.01)
    interval_ms = max(1, int(round(stride * dt * 1000)))

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=n_frames,
        interval=interval_ms,
        blit=False,  # othersiwe the computation time goes crazy
    )

    plt.tight_layout()
    plt.show()

    return anim