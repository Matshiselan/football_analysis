import pandas as pd

def compute_speed_bands_from_full_csv(
    segment_name: str,
    segment_dir: str,
    fps: int = 25
):
    """
    Uses player_full_stats.csv produced by SpeedAndDistance_Estimator
    """

    path = f"{segment_dir}/player_full_stats.csv"
    df = pd.read_csv(path)

    # Required columns sanity check
    required = {"track_id", "frame_idx", "speed_kmh"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")

    # Distance per frame (m)
    df["distance_m"] = df["speed_kmh"] * 1000 / 3600 / fps

    # Speed bands
    df["band"] = "low"
    df.loc[df["speed_kmh"] >= 20, "band"] = "hsr"
    df.loc[df["speed_kmh"] >= 25, "band"] = "sprint"

    band_dist = (
        df
        .groupby(["track_id", "band"])
        .agg(distance_m=("distance_m", "sum"))
        .reset_index()
    )

    out = (
        band_dist
        .pivot_table(
            index="track_id",
            columns="band",
            values="distance_m",
            fill_value=0
        )
        .reset_index()
        .rename(columns={
            "low": "low_speed_distance_m",
            "hsr": "hsr_distance_m",
            "sprint": "sprint_distance_m"
        })
    )

    out["segment"] = segment_name

    out_path = f"{segment_dir}/player_speed_bands.csv"
    out.to_csv(out_path, index=False)

    return out
