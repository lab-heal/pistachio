from datetime import datetime, time

import numpy as np
import pandas as pd

MS_IN_MINUTE = 60_000


def process_hr_df(
    hr_df: pd.DataFrame,
    timestamp_col: str = "ActivityTime",
    hr_col: str = "HeartRate",
) -> pd.DataFrame:
    df = hr_df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df.set_index(timestamp_col, inplace=True)
    return df


def daily_hrv_sdann_sleep(
    hr_df: pd.DataFrame,
    sleep_df: pd.DataFrame,
    default_sleep_end: time = time(7, 0, 0),
    default_sleep_duration: pd.Timedelta = pd.Timedelta(hours=9),
) -> pd.DataFrame:
    """
    :param: hr_df: pd.DataFrame with datetime index and "HeartRate" (bpm) column
    :param sleep_df: pd.DataFrame with "ActivityDateTime", "CalendarDate", "DeepSleepDurationInSeconds", "LightSleepDurationInSeconds", "AwakeDurationInSeconds",
        "UnmeasurableSleepDurationInSeconds", "RemSleepInSeconds" columns
    """

    hr_df = process_hr_df(hr_df)
    sleep_start_and_end_df = sleep_start_and_end_times(sleep_df)

    # For each interval, mask hr_df and compute SDANN
    def compute_sdann(row):
        # row.name is the value of the index column, i.e., the date
        default_end = datetime.combine(row.name, default_sleep_end)
        default_start = default_end - default_sleep_duration

        sleep_start = (
            sleep_start_and_end_df.loc[row.name, "sleep_start"]
            if row.name in sleep_start_and_end_df.index
            else default_start
        )
        sleep_end = (
            sleep_start_and_end_df.loc[row.name, "sleep_end"]
            if row.name in sleep_start_and_end_df.index
            else default_end
        )
        mask = (hr_df.index >= sleep_start) & (hr_df.index <= sleep_end)
        hr_slice = hr_df.loc[mask].copy()
        if hr_slice.empty:
            return np.nan
        hr_slice["nn_interval"] = MS_IN_MINUTE / hr_slice["HeartRate"]
        return sdann_from_hr(hr_slice)

    hrv = hr_df.groupby(hr_df.index.date).apply(compute_sdann)
    hrv.name = "overnight_hrv_sdann"
    return hrv


def sdann_from_hr(hr_df: pd.DataFrame) -> float:
    """
    :param: hr_df: pd.DataFrame with timestamp index and "HeartRate" (bpm) column
    """
    nn_intervals = MS_IN_MINUTE / hr_df["HeartRate"]
    five_min_averages = nn_intervals.resample("5min").mean()
    # ddof=1 for sample standard deviation
    sdann = np.std(five_min_averages.dropna(), ddof=1)
    return sdann


def sleep_start_and_end_times(sleep_df: pd.DataFrame) -> pd.DataFrame:
    """
    :param sleep_df: pd.DataFrame with "ActivityDateTime", "CalendarDate", "DeepSleepDurationInSeconds", "LightSleepDurationInSeconds", "AwakeDurationInSeconds",
        "UnmeasurableSleepDurationInSeconds", "RemSleepInSeconds" columns
    :return: pd.DataFrame with date index, "sleep_start", "sleep_end" columns
    """
    df = sleep_df.copy()

    df["sleep_start"] = pd.to_datetime(df["ActivityDateTime"])
    duration_cols = [
        "DeepSleepDurationInSeconds",
        "LightSleepDurationInSeconds",
        "AwakeDurationInSeconds",
        "UnmeasurableSleepDurationInSeconds",
        "RemSleepInSeconds",
    ]
    df["total_sleep_seconds"] = df[duration_cols].sum(axis=1)
    df["sleep_end"] = df["sleep_start"] + pd.to_timedelta(
        df["total_sleep_seconds"], unit="s"
    )
    df["CalendarDate"] = df["CalendarDate"].dt.date
    result = df.set_index("CalendarDate")[["sleep_start", "sleep_end"]]
    return result
