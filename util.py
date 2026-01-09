import numpy as np
import pandas as pd


def engineer_features(
    df: pd.DataFrame,
    stress_lookback_days=0,
    sleep_lookback_range=(0, 0),
    extra_cols_to_drop=[],
) -> pd.DataFrame:
    ## Feature engineering
    df["day_of_week"] = pd.to_datetime(df["ActivityDateTime"]).dt.dayofweek

    # Cyclical encoding for month
    df["month_sin"] = np.sin(
        2 * np.pi * pd.to_datetime(df["ActivityDateTime"]).dt.month / 12
    )
    df["month_cos"] = np.cos(
        2 * np.pi * pd.to_datetime(df["ActivityDateTime"]).dt.month / 12
    )

    df["therapy_length_days"] = (
        pd.to_datetime(df["ActivityDateTime"]) - pd.to_datetime(df["Therapy Start"])
    ).dt.days
    df["therapy_week"] = df["therapy_length_days"] // 7

    # Useful for indexing but will not be in the final model
    df["ActivityDateTime"] = pd.to_datetime(df["ActivityDateTime"], format=FORMAT_MIN)

    # Steps
    df["steps_0_to_15m"] = df["Steps"]
    df["steps_15_to_30m"] = df["Steps"].shift(1)
    df["steps_30_to_45m"] = df["Steps"].shift(2)
    df["steps_45_to_60m"] = df["Steps"].shift(3)
    df = df.drop("Steps", axis=1)

    # Hrv
    df["hrv_sdann_overnight_diff"] = df["hrv_sdann_overnight"] - df["hrv_sdann_avg_7d"]
    # df = df.drop(["hrv_sdann_overnight", "hrv_sdann_avg_7d"], axis=1)

    # StressLevelValueAverage
    df["StressLevelValueAverage_15m"] = df["StressLevelValueAverage"]
    df["StressLevelValueAverage_30m"] = (
        df["StressLevelValueAverage"].rolling(window=2, min_periods=1).sum()
    )
    df["StressLevelValueAverage_45m"] = (
        df["StressLevelValueAverage"].rolling(window=3, min_periods=1).sum()
    )
    df["StressLevelValueAverage_60m"] = (
        df["StressLevelValueAverage"].rolling(window=4, min_periods=1).sum()
    )
    df = df.drop("StressLevelValueAverage", axis=1)

    # Drop features
    df = df.drop(
        [
            # "Diagnosis",
            # "Medication ",
            "CDI start date",
            "PDI start date",
            "PDI end date",
            "Type of medication",
            "Medication start date",
            "Week",
            "Therapy session",
            "Therapy Start",
            "Therapy End",
            "Education Status",
            "Parental Status",
            "Employment Status",
            # "Pre.ECBI",
            # "Pre.ECBI.Prob",
            "Post.ECBI",
            "Post.ECBI.Prob",
            "QuitStudy",
            "ParticipatingParent.Sex",
            "Parent-PhoneType",
            # Data that is only available as "real time data" (more battery use?) in Companion SDK
            # NOTE: Removing these actually improves model accuracy?
            "DurationInSeconds",  # total active time
            "DistanceInMeters",
            "ActiveKilocalories",
            "METmins",
            "METavg",
            "activity_seconds_sedentary",
            "activity_seconds_active",
            "activity_seconds_highly_active",
        ]
        # Sleep
        + [
            f"{prefix}_T-{day}"
            for day in range(*sleep_lookback_range)
            for prefix in [
                "awake",
                "deep",
                "light",
                "rem",
                "unmeasurable",
            ]
        ]
        + [
            f"{prefix}_T-{day}"
            for day in range(stress_lookback_days + 1, 6)
            for prefix in [
                "AverageStressLevel",
                "MaxStressLevel",
                "StressDurationInSeconds",
                "RestStressDurationInSeconds",
                "ActivityStressDurationInSeconds",
                "LowStressDurationInSeconds",
                "MediumStressDurationInSeconds",
                "HighStressDurationInSeconds",
                "StressQualifier",
            ]
        ]
        # moving window stats
        + [
            # "hr_moving_avg_10m",
            # "hr_moving_std_10m",
            # "hr_moving_min_10m",
            # "hr_moving_max_10m",
            "hr_moving_avg_30m",
            "hr_moving_std_30m",
            "hr_moving_min_30m",
            "hr_moving_max_30m",
            "hr_moving_avg_60m",
            "hr_moving_std_60m",
            "hr_moving_min_60m",
            "hr_moving_max_60m",
        ]
        + extra_cols_to_drop,
        axis=1,
    )
    df = df.drop([col for col in df.columns if col.startswith("hr-prev")], axis=1)

    def yn_to_bool(df):
        """
        Convert columns with 'Y'/'N' strings to boolean True/False.
        """
        for col in df.columns:
            if df[col].nunique() == 2 and set(df[col].dropna().unique()) == {"Y", "N"}:
                df[col] = df[col].map({"Y": True, "N": False})
        return df

    df = yn_to_bool(df)

    # Convert categorical columns to dummy variables
    df = pd.get_dummies(df, drop_first=True)

    return df


def prep_X_y(df: pd.DataFrame, response_column: str) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(
        [
            "ActivityDateTime",
            "tantrum_within_60m",
            "tantrum_within_45m",
            "tantrum_within_30m",
            "tantrum_within_15m",
            # Useful for indexing
            "Arm_Sham",
            "dyad",
            "therapy_week",
        ],
        axis=1,
    )
    y = df[response_column].astype(int)
    return X, y


FORMAT_DAY = "%m/%d/%Y"
FORMAT_MIN = "%m/%d/%Y %I:%M %p"
FORMAT_SEC = "%m/%d/%Y %I:%M:%S %p"
