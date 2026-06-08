from pathlib import Path
import re

import numpy as np
import pandas as pd


def load_drone_trajectories(filepath: str) -> tuple[float, float, float, dict[int, np.ndarray]]:
    # Read CSV
    filepath = Path(filepath) # type: ignore

    if not filepath.is_absolute(): # type: ignore
        filepath = Path(__file__).resolve().parent.parent / filepath # type: ignore

    df = pd.read_csv(filepath)

    # Time vector and dt
    t_start = float(df["t"].iloc[0])
    t_end = float(df["t"].iloc[-1])
    t = df["t"].to_numpy(dtype=float)
    dt = float(np.mean(np.diff(t)))

    # Detect drone IDs automatically
    drone_ids = sorted({
        int(match.group(1))
        for col in df.columns
        if (match := re.match(r"drone_(\d+)_x", col))
    })

    # Build trajectory dictionary
    trajectories = {}

    for (drone_id) in drone_ids:
        trajectories[drone_id] = df[
            [
                f"drone_{drone_id}_x",
                f"drone_{drone_id}_y",
                f"drone_{drone_id}_z",
            ]
        ].to_numpy(dtype=float)

    return t_start, t_end, dt, trajectories # type: ignore