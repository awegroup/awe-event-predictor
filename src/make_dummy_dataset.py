"""
Generate a small synthetic dataset in the raw input format.

The flights are NOT physically accurate. They only mimic the file format and
contain tether-force peaks above 2500 N with a precursor (the kite speeds up a
few seconds before each peak), so the whole pipeline can be run end-to-end
without access to real flight data.

Writes

    <output>/training_data/dummy_train_XX.csv
    <output>/test_data/dummy_test_XX.csv

Usage:
    python src/make_dummy_dataset.py --output data/dummy_dataset
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FS = 10.0                 # Hz
FORCE_GAIN = 2.0          # N / (m/s)^2, tether force = FORCE_GAIN * V^2
V_BASE = 25.0             # m/s, mean kite speed
V_EVENT = 14.0            # m/s, extra speed at the peak of an event
EVENT_SIGMA = 2.5         # s, width of the speed bump
MIN_EVENT_GAP = 60.0      # s
MAX_EVENT_GAP = 110.0     # s
LOOP_PERIOD = 12.0        # s, one figure-eight
START_TIME = 1_700_000_000.0


def _event_times(duration, rng):
    times, t = [], 40.0
    while True:
        t += rng.uniform(MIN_EVENT_GAP, MAX_EVENT_GAP)
        if t > duration - 20.0:
            return np.array(times)
        times.append(t)


def make_flight(duration, seed):
    """Return one synthetic flight as a DataFrame."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration, 1.0 / FS)
    n = len(t)
    w = 2 * np.pi / LOOP_PERIOD

    # Kite speed: figure-eight modulation + noise + one bump per event
    speed = V_BASE + 3.0 * np.sin(2 * w * t) + rng.normal(0, 0.3, n)
    for te in _event_times(duration, rng):
        speed += V_EVENT * rng.uniform(0.9, 1.1) * np.exp(-0.5 * ((t - te) / EVENT_SIGMA) ** 2)

    force = FORCE_GAIN * speed ** 2 + rng.normal(0, 30, n)

    # Figure-eight geometry
    azimuth = 0.35 * np.sin(w * t)
    elevation = 0.55 + 0.12 * np.sin(2 * w * t)
    tether_length = 250.0 + 40.0 * np.sin(2 * np.pi * t / 300.0)
    reelout_speed = np.gradient(tether_length, t)
    distance = tether_length + 2.0
    height = distance * np.sin(elevation)
    east = distance * np.cos(elevation) * np.sin(azimuth)
    north = distance * np.cos(elevation) * np.cos(azimuth)

    course = np.unwrap(np.arctan2(np.gradient(east, t), np.gradient(height, t)))
    heading = course + rng.normal(0, 0.02, n)
    course_rate = np.gradient(course, t)

    vx = speed * np.cos(course)
    vy = speed * np.sin(course)
    vz = np.gradient(height, t) + rng.normal(0, 0.2, n)

    df = pd.DataFrame({
        "time": START_TIME + seed * 86400 + t,
        "kite_0_vx": vx,
        "kite_0_vy": vy,
        "kite_0_vz": vz,
        "kite_0_altitude": height + 5.0,
        # attitude in degrees, wrapped to [-180, 180), as in the real logs
        "kite_0_roll": np.rad2deg(0.3 * np.sin(w * t)),
        "kite_0_pitch": np.rad2deg(0.1 * np.cos(w * t)),
        "kite_0_yaw": (np.rad2deg(heading) + 180.0) % 360.0 - 180.0,
        "ground_tether_reelout_speed": reelout_speed,
        "ground_tether_length": tether_length,
        "ground_tether_force": force,
        "kite_pos_east": east,
        "kite_pos_north": north,
        "kite_height": height,
        "kite_elevation": elevation,
        "kite_azimuth": azimuth,
        "kite_distance": distance,
        "kite_heading": np.angle(np.exp(1j * heading)),  # wrapped to [-pi, pi]
        "kite_course": np.angle(np.exp(1j * course)),
        "kite_actual_steering": 30.0 * np.cos(w * t) + rng.normal(0, 1.0, n),
        "kite_actual_depower": 20.0 + rng.normal(0, 0.5, n),
        "ground_mech_power": force * reelout_speed,
        "kite_turn_rate": course_rate + rng.normal(0, 0.02, n),
        "kite_course_rate": course_rate,
    })

    # Sprinkle a few NaNs (not in the first row) to exercise forward-filling
    nan_rows = rng.choice(np.arange(1, n), size=max(1, n // 1000), replace=False)
    df.loc[nan_rows, "kite_actual_depower"] = np.nan
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="data/dummy_dataset", help="Output dataset folder (default: data/dummy_dataset)")
    parser.add_argument("--n-train", type=int, default=4, help="Number of training flights (default: 4)")
    parser.add_argument("--n-test", type=int, default=2, help="Number of test flights (default: 2)")
    parser.add_argument("--duration", type=float, default=900.0, help="Flight duration in seconds (default: 900)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    args = parser.parse_args()

    out = Path(args.output)
    splits = [("training_data", "dummy_train", args.n_train), ("test_data", "dummy_test", args.n_test)]
    seed = args.seed
    for folder, prefix, count in splits:
        (out / folder).mkdir(parents=True, exist_ok=True)
        for i in range(count):
            path = out / folder / f"{prefix}_{i:02d}.csv"
            # Raw format: space-separated, first column is an unnamed index
            make_flight(args.duration, seed).to_csv(path, sep=" ")
            seed += 1
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
