import os
import pandas as pd


def compute_speed_bands_from_full_csv(
    segment_name: str,
    segment_dir: str,
    fps: int = 24
):
    """
    Computes per-player distance covered in:
      - low speed (<20 km/h)
      - high-speed running (20–25 km/h)
      - sprinting (>=25 km/h)

    Uses full_speed_distance.csv exported by SpeedAndDistance_Estimator
    """

    path = os.path.join(segment_dir, "full_speed_distance.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing full_speed_distance.csv in {segment_dir}")

    df = pd.read_csv(path)

    required = {"track_id", "frame_num", "speed_kmh"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")

    # Distance covered per frame (meters)
    df["distance_m"] = df["speed_kmh"] * 1000 / 3600 / fps

    # Speed bands
    df["band"] = "low"
    df.loc[df["speed_kmh"] >= 20, "band"] = "hsr"
    df.loc[df["speed_kmh"] >= 25, "band"] = "sprint"

    # Aggregate per player
    bands = (
        df.groupby(["track_id", "band"], as_index=False)
          .agg(distance_m=("distance_m", "sum"))
          .pivot(index="track_id", columns="band", values="distance_m")
          .fillna(0)
          .reset_index()
    )

    bands.rename(
        columns={
            "low": "low_speed_distance_m",
            "hsr": "hsr_distance_m",
            "sprint": "sprint_distance_m",
        },
        inplace=True,
    )

    bands["segment"] = segment_name

    out_path = os.path.join(segment_dir, "player_speed_bands.csv")
    bands.to_csv(out_path, index=False)

    print(f"📊 Speed bands saved: {out_path}")
    return out_path
