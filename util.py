from typing import TypedDict

import numpy as np
import pandas as pd

FEATURE_SETS = {
    "index": [
        "ActivityDateTime",
        "Arm",
        "dyad",
        "therapy_week",
    ],
    "response": [
        "tantrum_within_15m",
        "tantrum_within_30m",
        "tantrum_within_45m",
        "tantrum_within_60m",
    ],
    "hr": [f"hr_moving_{stat}_10m" for stat in ["avg", "std", "min", "max"]],
    "activity": [
        "steps_0_to_15m",
        "steps_15_to_30m",
        "steps_30_to_45m",
        "steps_45_to_60m",
    ],
    "stress": [
        "stress_avg_garmin_0_to_15m",
        "stress_avg_garmin_15_to_30m",
        "stress_avg_garmin_30_to_45m",
        "stress_avg_garmin_45_to_60m",
    ],
    "overnight_hrv": [
        "hrv_sdann_overnight",
        "hrv_sdann_avg_7d",
        "hrv_sdann_overnight_diff",
    ],
    "medical": [
        "Diag.ADHD",
        "Diag.ASD",
        "Diag.Anxiety",
        "Diag.SAD",
        "Child.On.Antidepressants",
        "Child.On.Stimulants",
        "Child.On.Non.Stimulants",
    ],
    "therapy": [
        "Pre.ECBI",
        "Pre.ECBI.Prob",
        "therapy_length_days",
    ],
    "child_demo": [
        "Child sex",
        "Child.Age",
    ],
    "parent_demo": [
        "Education Status",
        "Parent-PhoneType",
        "Parental Status",
        "Parent.Age",
        "BothParentsInStudy",
    ],
    "temporal": [
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
    ],
}


SLEEP_FEAT_PREFIXES = (
    "awake",
    "deep",
    "light",
    "rem",
    "unmeasurable",
)


SLEEP_LOOKBACK_DAYS = list(range(0, 5))


class FeatureSetDataFrames(TypedDict):
    # useful indexing columns, removed in "prep_X_y"
    index: pd.DataFrame
    # response columns
    response: pd.DataFrame
    # features
    hr: pd.DataFrame
    activity: pd.DataFrame
    sleep: pd.DataFrame
    stress: pd.DataFrame
    overnight_hrv: pd.DataFrame
    medical: pd.DataFrame
    therapy: pd.DataFrame
    child_demo: pd.DataFrame
    parent_demo: pd.DataFrame
    temporal: pd.DataFrame


def sleep_features(sleep_days_to_keep) -> list[str]:
    return [
        f"{prefix}_T-{day}"
        for day in sleep_days_to_keep
        for prefix in SLEEP_FEAT_PREFIXES
    ]


def engineer_features(
    df: pd.DataFrame,
    stress_lookback_days=0,
    sleep_days_to_keep=[1, 2],  # Numbers from 0 to 4
    extra_cols_to_drop=[],
) -> FeatureSetDataFrames:
    # Useful for indexing but will not be in the final model
    df["ActivityDateTime"] = pd.to_datetime(df["ActivityDateTime"], format=FORMAT_MIN)

    # Steps
    df["steps_0_to_15m"] = df["Steps"]
    df["steps_15_to_30m"] = df["Steps"].shift(1)
    df["steps_30_to_45m"] = df["Steps"].shift(2)
    df["steps_45_to_60m"] = df["Steps"].shift(3)
    df = df.drop("Steps", axis=1)

    # Cyclical encodings
    day_of_week = pd.to_datetime(df["ActivityDateTime"]).dt.dayofweek
    df["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)

    month = pd.to_datetime(df["ActivityDateTime"]).dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    df["therapy_length_days"] = (
        pd.to_datetime(df["ActivityDateTime"]) - pd.to_datetime(df["Therapy Start"])
    ).dt.days
    df["therapy_week"] = df["therapy_length_days"] // 7

    # Overnight HRV
    df["hrv_sdann_overnight_diff"] = df["hrv_sdann_overnight"] - df["hrv_sdann_avg_7d"]

    # Stress level (based on momentary HRV)
    df["stress_avg_garmin_0_to_15m"] = df.pop("StressLevelValueAverage")
    df["stress_avg_garmin_15_to_30m"] = df["stress_avg_garmin_0_to_15m"].shift(1)
    df["stress_avg_garmin_30_to_45m"] = df["stress_avg_garmin_0_to_15m"].shift(2)
    df["stress_avg_garmin_45_to_60m"] = df["stress_avg_garmin_0_to_15m"].shift(3)

    # Including both "Medication " and "Type of medication" reduces accuracy bc of collinearity?
    df = df.drop(
        ["Type of medication", "Medication ", "Diagnosis"],
        axis=1,
    )

    df = df.drop(
        [
            # A list of careers that is too diverse to be useful
            "Employment Status",
            # Collinear with "Parental Status"
            "Parent.WidowedSingle",
            "Parent.Married",
            "Parent.SeparatedDivorced",
            "Parent.Engaged.Together",
            "ParticipatingParent.Sex",
        ],
        axis=1,
    )
    # Drop features
    df = df.drop(
        [
            "CDI start date",
            "PDI start date",
            "Medication start date",
            "Week",
            "Therapy session",
            "Therapy Start",
            # Not known when the study begins
            "Therapy End",
            "Post.ECBI",
            "Post.ECBI.Prob",
            "PDI end date",
            "QuitStudy",
            # Data that is only available as "real time data" (more battery use?) in Companion SDK
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
            for day in [
                day for day in SLEEP_LOOKBACK_DAYS if day not in sleep_days_to_keep
            ]
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

    sleep_feature_names = sleep_features(sleep_days_to_keep)
    return {
        "index": pd.get_dummies(df[FEATURE_SETS["index"]], drop_first=True),
        "response": pd.get_dummies(df[FEATURE_SETS["response"]]),
        "hr": pd.get_dummies(df[FEATURE_SETS["hr"]], drop_first=True),
        "activity": pd.get_dummies(df[FEATURE_SETS["activity"]], drop_first=True),
        "sleep": pd.get_dummies(df[sleep_feature_names], drop_first=True),
        "stress": pd.get_dummies(df[FEATURE_SETS["stress"]], drop_first=True),
        "overnight_hrv": pd.get_dummies(
            df[FEATURE_SETS["overnight_hrv"]], drop_first=True
        ),
        "medical": pd.get_dummies(df[FEATURE_SETS["medical"]], drop_first=True),
        "therapy": pd.get_dummies(df[FEATURE_SETS["therapy"]], drop_first=True),
        "child_demo": pd.get_dummies(df[FEATURE_SETS["child_demo"]], drop_first=True),
        "parent_demo": pd.get_dummies(df[FEATURE_SETS["parent_demo"]], drop_first=True),
        "temporal": pd.get_dummies(df[FEATURE_SETS["temporal"]], drop_first=True),
    }


def prep_X_y(df: pd.DataFrame, response_column: str) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(
        [
            "ActivityDateTime",
            # response columns
            "tantrum_within_60m",
            "tantrum_within_45m",
            "tantrum_within_30m",
            "tantrum_within_15m",
            # These were useful for indexing but should not be in the model
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
