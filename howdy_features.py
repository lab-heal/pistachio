"""Feature engineering shared by inference.ipynb and retraining.ipynb.

This is a focused re-implementation of the pistachio tantrum-prediction feature
pipeline (../pistachio/util.py + process_csvs.ipynb), adapted to MyDataHelps (MDH)
data and to the "watch" + "medical" feature superset only.

Differences from the reference, by design:
  * HRV is NOT used (too brittle to compute weekly in production) -- the
    `hrv_sdann_*` columns are dropped entirely from the feature set.
  * Temporal and demographic features are excluded (they are not part of the
    watch+medical superset).
  * Categorical (medical bool) columns are encoded with a scikit-learn
    `OneHotEncoder(handle_unknown="ignore")` *inside* the model pipeline rather
    than `pd.get_dummies`, so the saved artifact is a self-contained production
    pipeline (encoders + model + tuned decision threshold) that tolerates unseen
    categories at inference. `engineer_features` therefore leaves medical columns
    raw; the pipeline does the encoding.

Each row is a 15-minute interval. The label `tantrum_within_60m` is 1 if a
tantrum *starts* within 60 minutes of the interval's timestamp.
"""

from __future__ import annotations

import glob
import os
from datetime import date, datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import (
    PredefinedSplit,
    StratifiedKFold,
    TunedThresholdClassifierCV,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------------
# Feature definitions (watch + medical), ported from pistachio/util.py
# ---------------------------------------------------------------------------

HR_FEATURES = [f"hr_moving_{stat}_10m" for stat in ("avg", "std", "min", "max")]

ACTIVITY_FEATURES = [
    "steps_0_to_15m",
    "steps_15_to_30m",
    "steps_30_to_45m",
    "steps_45_to_60m",
]

STRESS_FEATURES = [
    "stress_avg_garmin_0_to_15m",
    "stress_avg_garmin_15_to_30m",
    "stress_avg_garmin_30_to_45m",
    "stress_avg_garmin_45_to_60m",
]

SLEEP_PREFIXES = ("awake", "deep", "light", "rem", "unmeasurable")
SLEEP_DAYS_TO_KEEP = [1, 2]  # previous 1-2 nights (T-1, T-2)

# Static per-participant medical metadata; one-hot encoded at train time.
MEDICAL_FEATURES = [
    "Diag.ADHD",
    "Diag.ASD",
    "Diag.Anxiety",
    "Diag.SAD",
    "Child.On.Antidepressants",
    "Child.On.Stimulants",
    "Child.On.Non.Stimulants",
]

THERAPY_FEATURES = ["Pre.ECBI", "Pre.ECBI.Prob", "therapy_length_days"]

INDEX_COLS = ["ActivityDateTime", "dyad", "therapy_week"]

RESPONSE_COLS = [
    "tantrum_within_15m",
    "tantrum_within_30m",
    "tantrum_within_45m",
    "tantrum_within_60m",
    "time_before_next_tantrum",
]

TANTRUM_INTERVAL_MINUTES = [15, 30, 45, 60]

# ResultIdentifier of the tantrum onset question in SurveyQuestionResults. The
# export's two tantrum surveys (tantrum-log-always-on / tantrum-log-post-predict)
# share this identifier; unrelated surveys use other identifiers and are ignored.
TANTRUM_START_IDENTIFIER = "tantrum_start_time"


# The model consumes a re-encoded sleep block: a total-sleep amount plus the
# deep/light/rem composition (shares of true sleep), with awake/unmeasurable kept
# as raw durations. All raw durations stay in SECONDS (no minute conversion).
# `transform_sleep_features` builds these from the raw `{prefix}_T-{day}` seconds
# that the parsers (`sleep_features_from_by_date`, `_sleep_by_calendar_date`) emit.
SLEEP_FEATURE_NAMES = (
    "total_sleep",
    "deep_pct",
    "light_pct",
    "rem_pct",
    "awake",
    "unmeasurable",
)


def sleep_features(days_to_keep=SLEEP_DAYS_TO_KEEP) -> list[str]:
    return [f"{name}_T-{day}" for day in days_to_keep for name in SLEEP_FEATURE_NAMES]


def transform_sleep_features(df: pd.DataFrame, days=SLEEP_DAYS_TO_KEEP) -> pd.DataFrame:
    """Re-encode raw per-stage sleep seconds into a total-sleep amount + stage
    composition, for each previous night (T-1, T-2).

    true sleep = deep+light+rem; `total_sleep` is that sum in SECONDS, and
    deep/light/rem each become their share of it (`*_pct`, summing to 1). awake and
    unmeasurable are kept as raw seconds, outside the sleep total. Where true sleep
    is 0 (or any stage is missing) the share is NaN, which XGBoost handles natively
    -- matching how the raw second columns already behaved.

    Per-night transform is skipped when its raw `deep/light/rem_T-{day}` columns
    are absent, so this is a safe no-op on frames that already carry the re-encoded
    columns (e.g. an inference row aligned straight to `FEATURE_COLUMNS`).
    """
    df = df.copy()
    for day in days:
        raw_cols = [f"{s}_T-{day}" for s in ("deep", "light", "rem")]
        if not all(c in df.columns for c in raw_cols):
            continue
        deep, light, rem = (df[c] for c in raw_cols)
        total = deep + light + rem  # seconds
        df[f"total_sleep_T-{day}"] = total
        df[f"deep_pct_T-{day}"] = deep / total  # 0/0 -> NaN (intended)
        df[f"light_pct_T-{day}"] = light / total
        df[f"rem_pct_T-{day}"] = rem / total
        df = df.drop(columns=raw_cols)
    return df


# Numeric model inputs (HRV is intentionally excluded -- see module docstring).
NUMERIC_FEATURE_COLUMNS = (
    HR_FEATURES
    + ACTIVITY_FEATURES
    + sleep_features()
    + STRESS_FEATURES
    + THERAPY_FEATURES
)

# The exact set and order of columns the model pipeline consumes: numeric features
# followed by the raw (bool) medical columns, which the pipeline's OneHotEncoder
# encodes. `engineer_features` emits feature blocks in this order and
# `align_features` reindexes inference rows to match, so the ColumnTransformer's
# `feature_names_in_` is stable across train / retrain / inference.
FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + MEDICAL_FEATURES


# ---------------------------------------------------------------------------
# Feature engineering (raw 15-min intervals -> model-ready frame)
# ---------------------------------------------------------------------------


def canonicalize_interval_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Pin the raw-interval frame's date/time columns to parquet-stable dtypes.

    The retrain job persists this frame to parquet, then on later runs reads it
    back and concatenates it with a freshly built frame. pandas demotes a
    datetime/timedelta column to `object` (holding Timestamp/Timedelta values)
    when concatenating two frames whose columns differ in resolution -- and
    pyarrow cannot serialize an object column of Timestamps. Pinning one
    resolution (microseconds, pyarrow's native round-trip unit) on every producer,
    and on anything read back from parquet, keeps every concat uniform and every
    write serializable. Idempotent; only touches columns that exist.
    """
    df = df.copy()
    if "ActivityDateTime" in df.columns:
        df["ActivityDateTime"] = pd.to_datetime(
            df["ActivityDateTime"], utc=True
        ).astype("datetime64[us, UTC]")
    if "time_before_next_tantrum" in df.columns:
        df["time_before_next_tantrum"] = pd.to_timedelta(
            df["time_before_next_tantrum"]
        ).astype("timedelta64[us]")
    for minutes in TANTRUM_INTERVAL_MINUTES:
        col = f"tantrum_within_{minutes}m"
        if col in df.columns:
            df[col] = df[col].astype(bool)
    return df


def engineer_features(
    df: pd.DataFrame, sleep_days_to_keep=SLEEP_DAYS_TO_KEEP
) -> pd.DataFrame:
    """Turn a raw 15-minute interval frame into the watch+medical feature frame.

    Input columns expected (see `build_intervals`): ActivityDateTime, dyad,
    Steps, StressLevelValueAverage, hr_moving_*_10m, the sleep `{prefix}_T-{day}`
    columns, the MEDICAL_FEATURES (bool), Pre.ECBI, Pre.ECBI.Prob, "Therapy Start",
    and the RESPONSE_COLS. Any HRV columns present in the input are ignored.

    Returns a single frame = index cols + response cols + features, with the
    medical columns kept *raw* (bool) for the pipeline's OneHotEncoder, ready for
    `prep_X_y`.
    """
    df = canonicalize_interval_dtypes(df)
    df = df.sort_values(["dyad", "ActivityDateTime"])

    # Lag features are computed *within* each participant so windows never bleed
    # across participants (the reference shifts the whole frame -- a latent bug).
    grp = df.groupby("dyad", sort=False)
    df["steps_0_to_15m"] = df["Steps"]
    df["steps_15_to_30m"] = grp["Steps"].shift(1)
    df["steps_30_to_45m"] = grp["Steps"].shift(2)
    df["steps_45_to_60m"] = grp["Steps"].shift(3)

    df["stress_avg_garmin_0_to_15m"] = df["StressLevelValueAverage"]
    s = grp["StressLevelValueAverage"]
    df["stress_avg_garmin_15_to_30m"] = s.shift(1)
    df["stress_avg_garmin_30_to_45m"] = s.shift(2)
    df["stress_avg_garmin_45_to_60m"] = s.shift(3)

    # Therapy progression.
    df["therapy_length_days"] = (
        df["ActivityDateTime"]
        - pd.to_datetime(df["Therapy Start"]).dt.tz_localize("America/Chicago")
    ).dt.days
    df["therapy_week"] = df["therapy_length_days"] // 7

    # Re-encode raw per-stage sleep seconds into the total-sleep + composition
    # block the model consumes (mirrors `align_features` for inference).
    df = transform_sleep_features(df, sleep_days_to_keep)

    sleep_cols = [c for c in sleep_features(sleep_days_to_keep) if c in df.columns]
    medical_present = [c for c in MEDICAL_FEATURES if c in df.columns]

    index_df = df[[c for c in INDEX_COLS if c in df.columns]]
    response_df = df[[c for c in RESPONSE_COLS if c in df.columns]]
    # Feature blocks are emitted in FEATURE_COLUMNS order; medical columns stay
    # raw (bool) -- the model pipeline's OneHotEncoder encodes them.
    feature_blocks = [
        df[HR_FEATURES],
        df[ACTIVITY_FEATURES],
        df[sleep_cols],
        df[STRESS_FEATURES],
        df[THERAPY_FEATURES],
        df[medical_present],
    ]
    return pd.concat([index_df, response_df, *feature_blocks], axis=1)


def prep_X_y(
    df: pd.DataFrame, response_column: str = "tantrum_within_60m"
) -> tuple[pd.DataFrame, pd.Series]:
    """Split an engineered frame into features X and integer label y."""
    drop = [c for c in INDEX_COLS + RESPONSE_COLS if c in df.columns]
    X = df.drop(columns=drop)
    y = df[response_column].astype(int)
    return X, y


def bootstrap(
    df: pd.DataFrame, n_samples: int, random_state: int | None = None
) -> pd.DataFrame:
    """Resample `df` with replacement `n_samples` times and concatenate.

    Used to oversample one participant's data so it forms a larger representation
    of a combined training set (mirrors pistachio's `modeling.bootstrap`). Each
    iteration draws a full-size resample; with `n_samples` iterations the result
    has roughly `len(df) * n_samples` rows. A non-None `random_state` makes the
    draw reproducible while still varying across iterations (so replicates are not
    identical). `n_samples <= 0` returns an empty frame with the same columns.
    """
    frames = []
    for i in range(n_samples):
        seed = None if random_state is None else random_state + i
        frames.append(df.sample(frac=1, replace=True, random_state=seed))
    return pd.concat(frames) if frames else df.iloc[0:0]


def inject_random_labels(
    df: pd.DataFrame,
    response_column: str = "tantrum_within_60m",
    positive_rate: float = 0.005,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Overwrite `response_column` with random labels (TEST ONLY).

    Returns a copy of `df` whose response column is randomly set to 1 for
    ~`positive_rate` of rows (at least one positive when there are >=1 rows), the
    rest 0. Lets the retraining notebook run end-to-end when the MDH export has no
    real tantrum survey data. NOT for production training.
    """
    n = len(df)
    out = df.copy()
    if n == 0:
        out[response_column] = pd.Series(dtype=int)
        return out
    n_pos = min(n, max(1, round(n * positive_rate)))
    rng = np.random.default_rng(random_state)
    labels = np.zeros(n, dtype=int)
    labels[rng.choice(n, size=n_pos, replace=False)] = 1
    out[response_column] = labels
    return out


def align_features(X: pd.DataFrame) -> pd.DataFrame:
    """Reindex a (single- or multi-row) inference frame to FEATURE_COLUMNS.

    Guarantees the exact columns and order the model pipeline was fit on; any
    missing numeric column is filled with NaN (XGBoost handles it) and any extra
    column is dropped. Unseen medical category *values* are handled downstream by
    the pipeline's `OneHotEncoder(handle_unknown="ignore")`.

    The raw `{prefix}_T-{day}` sleep seconds assembled at inference are first
    re-encoded into the total-sleep + composition block (mirrors the training-time
    transform in `engineer_features`) so train and inference share one feature set.
    """
    return transform_sleep_features(X).reindex(columns=FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Model pipeline: encoders + XGBoost + tuned decision threshold
# ---------------------------------------------------------------------------
#
# The saved artifact is a single fitted `TunedThresholdClassifierCV` wrapping a
# `Pipeline([OneHotEncoder via ColumnTransformer, XGBClassifier])`. It carries
# all three pieces production needs -- encoders, model and the balanced-accuracy
# decision threshold (`best_threshold_`) -- and is shared by train.ipynb,
# retraining.ipynb and inference.ipynb so the format never diverges.


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """One-hot the medical columns present in X; pass numeric features through.

    `handle_unknown="ignore"` lets a medical value unseen in training (or a whole
    category absent at inference) encode to all-zeros instead of erroring -- this
    is what makes the saved pipeline usable in production.
    """
    medical = [c for c in MEDICAL_FEATURES if c in X.columns]
    return ColumnTransformer(
        [
            (
                "medical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                medical,
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def make_cv(groups=None, n_splits: int = 5, random_state: int = 0):
    """Cross-validation splitter for HPO and threshold tuning.

    With more than one dyad, fold by dyad (a `PredefinedSplit` keyed on each
    row's dyad, so no `groups` need be threaded through `fit`). With a single
    dyad (e.g. per-participant retraining) dyad grouping is degenerate, so fall
    back to a stratified KFold.
    """
    if groups is not None:
        unique = pd.unique(pd.Series(groups))
        if len(unique) > 1:
            fold_of = {g: i % n_splits for i, g in enumerate(unique)}
            test_fold = pd.Series(groups).map(fold_of).to_numpy()
            return PredefinedSplit(test_fold)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def make_xgb(xgb_params: dict | None = None) -> xgb.XGBClassifier:
    """An XGBoost classifier for the (sparse, imbalanced) feature set.

    `missing=np.nan` keeps XGBoost's native handling of the missing watch/sleep
    values; FLAML's `best_config` (or any hand-supplied params) is passed through.
    """
    params = dict(xgb_params or {})
    params.setdefault("eval_metric", "logloss")
    params.setdefault("missing", np.nan)
    return xgb.XGBClassifier(**params)


def build_pipeline(X: pd.DataFrame, xgb_params: dict | None = None) -> Pipeline:
    """Encoder + XGBoost pipeline (no threshold tuning)."""
    return Pipeline([("pre", make_preprocessor(X)), ("clf", make_xgb(xgb_params))])


XGB_FIXED_PARAMS = {
    # Some sources recommend sqrt(n_neg / n_pos) for imbalanced binary classification as going too high can be unstable
    # Empirically this performs better than 200
    "scale_pos_weight": 15,
    # Makes training more stable for imbalanced data (https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html)
    "max_delta_step": 1,
}


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    groups=None,
    xgb_params: dict | None = None,
    n_splits: int = 5,
    random_state: int = 0,
) -> TunedThresholdClassifierCV:
    """Fit the full production estimator on all of (X, y).

    Wraps the encoder+XGBoost pipeline in a `TunedThresholdClassifierCV` that
    picks the decision threshold maximizing balanced accuracy over dyad-grouped
    CV. The fitted object's `.predict()` applies that threshold and
    `.best_threshold_` exposes it; serialize it with `save_model`.
    """
    # pipeline = build_pipeline(X, xgb_params)
    pipeline = build_pipeline(X, xgb_params)
    tuned = TunedThresholdClassifierCV(
        pipeline,
        scoring="balanced_accuracy",
        thresholds=100,
        cv=make_cv(groups, n_splits=n_splits, random_state=random_state),
        refit=True,
        random_state=random_state,
    )
    tuned.fit(X, y)
    return tuned


def cross_val_metrics(
    X: pd.DataFrame,
    y: pd.Series,
    groups=None,
    xgb_params: dict | None = None,
    n_splits: int = 5,
    random_state: int = 0,
) -> dict:
    """Dyad-grouped CV metrics for the encoder+XGBoost pipeline.

    Reports balanced accuracy / sensitivity / specificity at the model's default
    0.5 cut (a tuned threshold is reported separately by `train_model`) plus
    AUROC, which is threshold-independent.
    """
    cv = make_cv(groups, n_splits=n_splits, random_state=random_state)
    pipeline = build_pipeline(X, xgb_params)
    y_arr = np.asarray(y)
    proba = cross_val_predict(pipeline, X, y_arr, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn = int(((y_arr == 0) & (pred == 0)).sum())
    fp = int(((y_arr == 0) & (pred == 1)).sum())
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_arr, pred)),
        "sensitivity": float(
            ((y_arr == 1) & (pred == 1)).sum() / max((y_arr == 1).sum(), 1)
        ),
        "specificity": float(tn / max(tn + fp, 1)),
        "auroc": float(roc_auc_score(y_arr, proba)),
    }


def save_model(model, path: str) -> None:
    """Serialize the full fitted pipeline (encoders + model + threshold)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str):
    """Load a pipeline saved by `save_model`."""
    return joblib.load(path)


# ---------------------------------------------------------------------------
# MDH export CSV -> raw 15-min interval frame
# ---------------------------------------------------------------------------


def _load_csv(
    export_dir: str, prefix: str, participant_id: str | None = None
) -> pd.DataFrame:
    """Load the one export CSV whose name is `{prefix}_<digits>...csv`.

    The digit anchor prevents e.g. `GarminSleepSummary` from also matching
    `GarminSleepSummary_Samples`.

    When `participant_id` is given and the CSV has a `ParticipantID` column, rows
    are filtered to that participant. (The `_Samples` CSVs lack `ParticipantID`;
    they filter implicitly via their `SummaryId` join to a filtered parent CSV.)
    """
    matches = glob.glob(os.path.join(export_dir, f"{prefix}_[0-9]*.csv"))
    if not matches:
        return pd.DataFrame()
    df = pd.read_csv(matches[0])
    if participant_id is not None and "ParticipantID" in df.columns:
        df = df[df["ParticipantID"] == participant_id]
    return df


def _local_dt(start_seconds, offset_seconds) -> pd.Series:
    """Garmin local wall-clock time = UTC epoch + offset"""
    return pd.to_datetime(start_seconds + offset_seconds, unit="s", utc=True)


def build_intervals(
    export_dir: str,
    participant_metadata: dict,
    participant_id: str | None = None,
) -> pd.DataFrame:
    """Build a raw 15-minute interval frame from an MDH export directory.

    `participant_metadata` supplies the static study fields that are not in the
    Garmin data: the MEDICAL_FEATURES (as bools), Pre.ECBI, Pre.ECBI.Prob and
    "Therapy Start" (date string). The result is the input expected by
    `engineer_features` (labels added separately via `compute_tantrum_labels`).

    `participant_id` is the MDH `ParticipantID` to filter the (multi-participant)
    export to. When `None`, the whole export is used (notebook/single-participant
    behavior). The `dyad` label written onto each row comes from
    `participant_metadata["dyad"]`, falling back to the export's first ParticipantID.
    """
    epochs = _load_csv(export_dir, "GarminEpochSummary", participant_id)
    if epochs.empty:
        return pd.DataFrame()

    dyad = participant_metadata.get("dyad") or epochs["ParticipantID"].iloc[0]

    # --- 15-minute grid spanning the participant's epoch data -----------------
    epochs = epochs.copy()
    epochs["dt"] = _local_dt(
        epochs["StartTimeInSeconds"], epochs["StartTimeOffsetInSeconds"]
    )
    epochs["bin"] = epochs["dt"].dt.floor("15min")
    grid = pd.date_range(epochs["bin"].min(), epochs["bin"].max(), freq="15min")
    out = pd.DataFrame({"ActivityDateTime": grid})

    # --- Steps per 15-min bin (NaN where no epoch exists at all) --------------
    steps = epochs.groupby("bin")["Steps"].sum()
    out["Steps"] = out["ActivityDateTime"].map(steps)

    # --- Garmin stress level, ~3-min samples -> 15-min mean (ignore -1) -------
    out["StressLevelValueAverage"] = _stress_per_bin(export_dir, grid, participant_id)

    # --- HR moving-window stats over the trailing 10 minutes ------------------
    hr = _hr_local_series(export_dir, participant_id)
    for stat in ("avg", "std", "min", "max"):
        out[f"hr_moving_{stat}_10m"] = _hr_moving_stat(hr, grid, stat)

    # --- Sleep features for the previous nights (T-1, T-2) --------------------
    # Garmin labels a sleep summary with the *wake-morning* CalendarDate, so the
    # night that ended this morning (T-1) carries today's date -> shift by
    # day - 1, not day. (`sleep_features_from_points` mirrors this for inference.)
    sleep_by_date = _sleep_by_calendar_date(export_dir, participant_id)
    for day in SLEEP_DAYS_TO_KEEP:
        night = (
            out["ActivityDateTime"].dt.normalize() - pd.Timedelta(days=day - 1)
        ).dt.date
        for prefix in SLEEP_PREFIXES:
            out[f"{prefix}_T-{day}"] = night.map(
                lambda d: sleep_by_date.get(d, {}).get(prefix, np.nan)
            )

    # --- Static study metadata broadcast to every interval --------------------
    out["dyad"] = dyad
    for col in MEDICAL_FEATURES + ["Pre.ECBI", "Pre.ECBI.Prob", "Therapy Start"]:
        out[col] = participant_metadata.get(col)

    return canonicalize_interval_dtypes(out)


def _stress_per_bin(
    export_dir: str, grid: pd.DatetimeIndex, participant_id: str | None = None
) -> pd.Series:
    samples = _load_csv(export_dir, "GarminStressDetailSummary_Samples")
    parents = _load_csv(export_dir, "GarminStressDetailSummary", participant_id)
    if samples.empty or parents.empty:
        return pd.Series(np.nan, index=range(len(grid)))

    starts = parents.set_index("SummaryId")[
        ["StartTimeInSeconds", "StartTimeOffsetInSeconds"]
    ]
    s = samples[samples["SampleType"] == "Stress"].join(starts, on="SummaryId")
    s = s[s["Value"] >= 0]  # -1 = unmeasured
    s["dt"] = _local_dt(
        s["StartTimeInSeconds"] + s["OffsetInSeconds"], s["StartTimeOffsetInSeconds"]
    )
    per_bin = s.groupby(s["dt"].dt.floor("15min"))["Value"].mean()
    return pd.Series(grid, index=range(len(grid))).map(per_bin).reset_index(drop=True)


def _hr_local_series(export_dir: str, participant_id: str | None = None) -> pd.Series:
    """All heart-rate samples as a time-sorted Series indexed by local time."""
    samples = _load_csv(export_dir, "GarminDailySummary_Samples")
    parents = _load_csv(export_dir, "GarminDailySummary", participant_id)
    if samples.empty or parents.empty:
        return pd.Series(dtype=float)

    starts = parents.set_index("SummaryId")[
        ["StartTimeInSeconds", "StartTimeOffsetInSeconds"]
    ]
    h = samples.join(starts, on="SummaryId").dropna(subset=["StartTimeInSeconds"])
    dt = _local_dt(
        h["StartTimeInSeconds"] + h["OffsetInSeconds"], h["StartTimeOffsetInSeconds"]
    )
    return pd.Series(h["HeartRate"].to_numpy(), index=dt).sort_index()


def _hr_moving_stat(hr: pd.Series, grid: pd.DatetimeIndex, stat: str) -> np.ndarray:
    """For each grid time t, a stat over HR samples in (t - 10min, t]."""
    if hr.empty:
        return np.full(len(grid), np.nan)
    times = hr.index.to_numpy()
    values = hr.to_numpy(dtype=float)
    window = np.timedelta64(10, "m")
    out = np.full(len(grid), np.nan)
    grid_np = grid.to_numpy()
    for i, t in enumerate(grid_np):
        lo = np.searchsorted(times, t - window, side="right")
        hi = np.searchsorted(times, t, side="right")
        if hi > lo:
            w = values[lo:hi]
            out[i] = {
                "avg": np.mean,
                "std": np.std,
                "min": np.min,
                "max": np.max,
            }[stat](w)
    return out


def _sleep_by_calendar_date(
    export_dir: str, participant_id: str | None = None
) -> dict[date, dict[str, float]]:
    sleep = _load_csv(export_dir, "GarminSleepSummary", participant_id)
    if sleep.empty:
        return {}
    cols = {
        "awake": "AwakeDurationInSeconds",
        "deep": "DeepSleepDurationInSeconds",
        "light": "LightSleepDurationInSeconds",
        "rem": "RemSleepInSeconds",
        "unmeasurable": "UnmeasurableSleepInSeconds",
    }
    result: dict[date, dict[str, float]] = {}
    for _, row in sleep.iterrows():
        d = pd.to_datetime(row["CalendarDate"]).date()
        result[d] = {
            prefix: row.get(src) for prefix, src in cols.items()
        }  # ty: ignore[invalid-assignment]
    return result


# ---------------------------------------------------------------------------
# MDH API device data -> sleep features (inference)
# ---------------------------------------------------------------------------

_SLEEP_SOURCE_KEYS = {
    "awake": "AwakeDurationInSeconds",
    "deep": "DeepSleepDurationInSeconds",
    "light": "LightSleepDurationInSeconds",
    "rem": "RemSleepInSeconds",
    "unmeasurable": "UnmeasurableSleepInSeconds",
}


def sleep_points_to_by_date(points: list[dict]) -> dict[date, dict[str, float]]:
    """Parse MDH SLEEP device-data points into a `{calendar_date: {prefix: seconds}}` map.

    Stage durations and the calendar date are read from each point's top-level /
    `properties` / `value` fields (matched case-insensitively) using the same
    field names as the export CSV (`DeepSleepDurationInSeconds`, `CalendarDate`,
    ...). The live API nests the durations under `properties` and returns them as
    *strings*, so values are coerced to float.
    """
    by_date: dict[date, dict[str, float]] = {}
    for point in points:
        flat = _flatten_point(point)
        raw_date = (
            _get_field(flat, "CalendarDate")
            or _get_field(flat, "observationDate")
            or _get_field(flat, "startDate")
        )
        cal = pd.to_datetime(raw_date, errors="coerce")
        if pd.isna(cal):
            continue
        by_date[cal.date()] = {
            prefix: _to_float(_get_field(flat, src))
            for prefix, src in _SLEEP_SOURCE_KEYS.items()
        }
    return by_date


def sleep_features_from_by_date(
    by_date: dict[date, dict[str, float]],
    target_dt: datetime,
    sleep_days_to_keep=SLEEP_DAYS_TO_KEEP,
) -> dict:
    """Build the re-encoded `{name}_T-1` / `_T-2` sleep features from a
    by-calendar-date map (the `sleep_features()` block: `total_sleep` +
    `deep/light/rem_pct`, with `awake`/`unmeasurable` raw seconds).

    Garmin labels a sleep summary with the *wake-morning* `CalendarDate`, so the
    night before the scored day (T-1) carries `target_dt`'s own date; T-2 carries
    the day before. This mirrors the offset in `build_intervals`/training.

    The raw per-stage seconds are re-encoded here via `transform_sleep_features`,
    so the inference path (and its logs) carry the same pct features the model
    consumes; the downstream `align_features` transform is then a no-op.
    """
    feats: dict = {}
    for day in sleep_days_to_keep:
        cal_date = (target_dt - timedelta(days=day - 1)).date()
        night = by_date.get(cal_date, {})
        for prefix in SLEEP_PREFIXES:
            feats[f"{prefix}_T-{day}"] = night.get(prefix, np.nan)
    return (
        transform_sleep_features(pd.DataFrame([feats]), sleep_days_to_keep)
        .iloc[0]
        .to_dict()
    )


def sleep_features_from_points(
    points: list[dict], target_dt: datetime, sleep_days_to_keep=SLEEP_DAYS_TO_KEEP
) -> dict:
    """Build the re-encoded `_T-1` / `_T-2` sleep features from MDH SLEEP points."""
    by_date = sleep_points_to_by_date(points)
    return sleep_features_from_by_date(by_date, target_dt, sleep_days_to_keep)


def _flatten_point(point: dict) -> dict:
    """Merge a device-data point's nested dicts so lookups are field-name based."""
    flat = dict(point)
    for nested in ("properties", "value"):
        val = point.get(nested)
        if isinstance(val, dict):
            flat.update(val)
    return flat


def _get_field(flat: dict, name: str):
    """Case-insensitive lookup tolerant of camelCase/PascalCase keys."""
    for key, value in flat.items():
        if key.lower() == name.lower():
            return value
    return None


def _to_float(value) -> float:
    """Coerce a raw stage duration (str/int/float/None) to float; blanks -> NaN.

    The live MDH API returns durations as strings (e.g. '3180'); XGBoost rejects
    object-dtype columns, so missing/unparseable values become NaN.
    """
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


# ---------------------------------------------------------------------------
# Tantrum labels from survey responses
# ---------------------------------------------------------------------------


def compute_tantrum_labels(
    intervals_df: pd.DataFrame, survey_df: pd.DataFrame, participant_id: str
) -> pd.DataFrame:
    """Attach forward-looking tantrum labels to a raw interval frame.

    Adds `tantrum_within_{15,30,45,60}m` (bool) and `time_before_next_tantrum`.
    A label is 1 if any tantrum *starts* in [t, t + N minutes). `survey_df` is the
    (multi-participant) export; onsets are filtered to `participant_id`.
    """
    df = intervals_df.copy()
    starts = _tantrum_onsets(survey_df, participant_id)
    times = pd.to_datetime(df["ActivityDateTime"])

    starts_np = np.sort(np.array(starts, dtype="datetime64[ns]"))
    for minutes in TANTRUM_INTERVAL_MINUTES:
        window = np.timedelta64(minutes, "m")
        df[f"tantrum_within_{minutes}m"] = times.map(
            lambda t: _starts_within(starts_np, t.to_datetime64(), window)
        )
    df["time_before_next_tantrum"] = times.map(
        lambda t: _time_to_next(starts_np, t.to_datetime64())
    )
    return canonicalize_interval_dtypes(df)


def _starts_within(
    starts: np.ndarray, t: np.datetime64, window: np.timedelta64
) -> bool:
    if starts.size == 0:
        return False
    return bool(np.any((starts >= t) & (starts < t + window)))


def _time_to_next(starts: np.ndarray, t: np.datetime64) -> pd.Timedelta:
    future = starts[starts >= t]
    if future.size == 0:
        return pd.Timedelta.max  # sentinel: no upcoming tantrum (matches reference)
    return pd.Timedelta(future[0] - t)


def _tantrum_onsets(
    survey_df: pd.DataFrame, participant_id: str
) -> list[np.datetime64]:
    """Extract one participant's tantrum onset timestamps from SurveyQuestionResults.

    Onsets are the rows whose `ResultIdentifier` is `tantrum_start_time` — the
    identifier shared by both tantrum surveys (tantrum-log-always-on and
    tantrum-log-post-predict); rows from other surveys in the export are ignored.

    The export is multi-participant; rows are filtered to `participant_id` via the
    `ParticipantID` column (absent in single-participant exports, where the filter
    is a no-op). `Answers` holds the participant-entered onset, typically a
    time-of-day like "6:18 PM"; `StartDate`/`EndDate` are when the question was
    answered and supply the (local) date to anchor that time to.
    """
    if survey_df is None or survey_df.empty:
        return []

    if "ParticipantID" in survey_df.columns:
        survey_df = survey_df[
            survey_df["ParticipantID"].astype(str) == str(participant_id)
        ]

    onset = survey_df[
        survey_df["ResultIdentifier"].astype(str) == TANTRUM_START_IDENTIFIER
    ]
    starts: list[np.datetime64] = []
    for _, row in onset.iterrows():
        submitted = pd.to_datetime(  # ty: ignore[no-matching-overload]
            row.get("EndDate") or row.get("StartDate"), errors="coerce"
        )
        answer = row.get("Answers")
        if answer is None:
            continue
        answer_time = pd.to_datetime(answer, errors="coerce").time()
        answer_dt = pd.Timestamp.combine(submitted.date(), answer_time)
        if pd.isna(answer_dt):
            continue
        starts.append(answer_dt.to_datetime64())
    return starts
