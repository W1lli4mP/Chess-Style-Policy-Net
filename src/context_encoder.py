import pandas as pd
import torch

#* context dim = 15
def encode_context(row: pd.Series) -> torch.Tensor:
    # one-hot encode time_class
    time_class = row["time_class"]
    is_bullet = float(time_class == "bullet")
    is_blitz = float(time_class == "blitz")
    is_rapid = float(time_class == "rapid")

    # one-hot encode my_colour
    my_colour = row["my_colour"]
    is_white = float(my_colour == "white")

    context = [
        is_bullet,
        is_blitz,
        is_rapid,
        is_white,
        float(row["base_time_seconds"]),
        float(row["increment_seconds"]),
        float(row["my_clock_seconds"]),
        float(row["opponent_clock_seconds"]),
        float(row["my_clock_to_base_ratio"]),
        float(row["opponent_clock_to_base_ratio"]),
        float(row["play"]),
        float(row["fullmove_number"]),
        float(row["rating_diff"]),
        float(row["opponent_rating"]),
        float(row["my_rating"])
    ]

    return torch.tensor(context, dtype=torch.float32)