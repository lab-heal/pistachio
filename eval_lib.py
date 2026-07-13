"""Evaluation helpers for the upweighting / feature-set retraining study.

This module ports `howdy-server/notebooks/eval_retrain_upweight.ipynb` into the
pistachio research project and broadens it the way the research project's
`eval_retrain_feature_sets.ipynb` did:

- the **production** modeling stack from `howdy_features` (vendored copy of
  `howdy_lib.features`): no HRV, per-dyad **upweighting** via `sample_weight`
  instead of bootstrapping;
- evaluated over **both** estimators (XGBoost and L2 logistic regression) and over
  the feature-set combinations `watch` / `+demographic` / `+medical` / `+both`;
- results cached as **parquet** (tidy per-prediction rows) and fitted pipelines
  cached as joblib, so the long training run is never repeated for analysis.

Everything XGBoost-specific is delegated to `howdy_features` (`F`); the only new
code is what `F` doesn't cover — demographics, the logistic-regression pipeline,
the estimator-generic HPO/training, the weekly retrain loop, and result caching.
"""

from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    make_scorer,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm.auto import tqdm

import howdy_features as F

# The per-dyad upweighting routes an XGBoost / LR `sample_weight` through the
# TunedThresholdClassifierCV -> Pipeline -> classifier stack and a weight-aware
# threshold scorer. sklearn requires metadata routing enabled globally for that;
# it is a no-op for the unweighted population / base fits (no sample_weight passed).
set_config(enable_metadata_routing=True)

RESPONSE_COLUMN = "tantrum_within_60m"
RANDOM_STATE = 0


# --------------------------------------------------------------------------- #
# Feature sets (production features, NO HRV).
# --------------------------------------------------------------------------- #
# "watch" = the production physio block (HR / activity / sleep / stress), exactly
# F.FEATURE_COLUMNS minus the medical + therapy columns. HRV is excluded by
# construction (it is not part of F.FEATURE_COLUMNS).
WATCH = F.HR_FEATURES + F.ACTIVITY_FEATURES + F.sleep_features() + F.STRESS_FEATURES

# "medical" superset mirrors pistachio's grouping (clinical diagnoses/medications
# + therapy intake), all from the production feature set.
MEDICAL = F.MEDICAL_FEATURES + F.THERAPY_FEATURES

# "demographic" features as defined in pistachio's util.py (child_demo + parent_demo).
# These are NOT in F.FEATURE_COLUMNS; attach_demographics() joins them on from the
# raw frame (they are static per dyad).
DEMOGRAPHIC = [
    "Child sex",
    "Child.Age",
    "Education Status",
    "Parent-PhoneType",
    "Parental Status",
    "Parent.Age",
    "BothParentsInStudy",
]


# The full feature set, used for the closing population SHAP analysis.
FULL_FEATURE_SET = "watch_demographic_medical"

# Categorical columns to one-hot encode (medical Y/N flags + the categorical
# demographics). Child.Age / Parent.Age are numeric and pass through.
DEMOGRAPHIC_CATEGORICAL = [
    "Child sex",
    "Education Status",
    "Parent-PhoneType",
    "Parental Status",
    "BothParentsInStudy",
]
CATEGORICAL_FEATURES = F.MEDICAL_FEATURES + DEMOGRAPHIC_CATEGORICAL


def attach_demographics(engineered: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Join the per-dyad demographic columns onto the engineered frame.

    `F.engineer_features` keeps only INDEX + RESPONSE + F.FEATURE_COLUMNS, so the
    demographics are pulled from the raw frame. They are static per dyad, so we
    take the first value per dyad and join on `dyad`.
    """
    demo = raw_df.groupby("dyad")[DEMOGRAPHIC].first()
    return engineered.join(demo, on="dyad")


def select_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Slice an engineered+demographics frame to INDEX + RESPONSE + `feature_cols`.

    `F.prep_X_y` then drops INDEX + RESPONSE, leaving exactly `feature_cols` as X.
    """
    cols = F.INDEX_COLS + F.RESPONSE_COLS + feature_cols
    return df[[c for c in cols if c in df.columns]].copy()


# --------------------------------------------------------------------------- #
# Estimator-generic pipeline (XGBoost reuses F.*; logistic regression is new).
# --------------------------------------------------------------------------- #
def make_preprocessor(X: pd.DataFrame, estimator: str) -> ColumnTransformer:
    """Preprocessor for the given estimator.

    xgboost: one-hot the categorical columns present in X (unseen categories ->
    all-zeros), pass numerics through untouched (XGBoost handles NaN natively).
    Mirrors `F.make_preprocessor` but also covers the demographic categoricals.

    lrl2: logistic regression cannot take NaN and is scale-sensitive, so impute +
    one-hot the categoricals and impute (median) + standard-scale the numerics
    (mirrors pistachio's modeling.py LR path).
    """
    cat = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    num = [c for c in X.columns if c not in cat]
    if estimator == "xgboost":
        return ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
    # OneHotEncoder handles NaN natively (encodes it as a category), so no categorical
    # imputer is needed -- and SimpleImputer rejects the bool medical columns anyway.
    cat_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    # StandardScaler.fit accepts sample_weight, so with metadata routing enabled it
    # must explicitly opt out -- only the classifier consumes the upweighting.
    num_pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler().set_fit_request(sample_weight=False)),
        ]
    )
    return ColumnTransformer(
        [("cat", cat_enc, cat), ("num", num_pipe, num)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_estimator(estimator: str, params: dict | None = None) -> BaseEstimator:
    """The classifier step for the given estimator.

    xgboost reuses `F.make_xgb` with `F.XGB_FIXED_PARAMS` (scale_pos_weight=15,
    max_delta_step=1). lrl2 is an L2 LogisticRegression with class_weight="balanced"
    for the imbalance; only LogisticRegression-valid keys from a FLAML config are
    forwarded (FLAML's lrl2 tunes `C`).
    """
    params = dict(params or {})
    if estimator == "xgboost":
        return F.make_xgb({**params, **F.XGB_FIXED_PARAMS})
    valid = set(LogisticRegression().get_params())
    lr_params = {k: v for k, v in params.items() if k in valid}
    # lr_params.setdefault("max_iter", 2000)
    # lr_params.setdefault("class_weight", "balanced")
    return LogisticRegression(penalty="l2", **lr_params)


def build_pipeline(
    X: pd.DataFrame, estimator: str, params: dict | None = None
) -> Pipeline:
    """Preprocessor + classifier pipeline (no threshold tuning)."""
    return Pipeline(
        [
            ("pre", make_preprocessor(X, estimator)),
            ("clf", make_estimator(estimator, params)),
        ]
    )


def flaml_hpo(
    X: pd.DataFrame,
    y: pd.Series,
    estimator: str,
    groups=None,
    max_iter: int = 100,
    time_budget: int | None = None,
    metric: str = "roc_auc",
    random_state: int = RANDOM_STATE,
    verbose: int = 1,
) -> tuple[dict, object]:
    """Estimator-generic FLAML HPO (generalizes `F.flaml_hpo` to LR as well).

    FLAML needs a numeric matrix, so it searches on the preprocessed features.
    For xgboost the fixed scale_pos_weight / max_delta_step are pinned via custom_hp
    (matching `F.flaml_hpo`); for lrl2 FLAML tunes the regularization. Returns
    `(best_config, automl)`.
    """
    from flaml import AutoML

    pre = make_preprocessor(X, estimator)
    X_enc = pre.fit_transform(X)

    fit_kwargs: dict = dict(
        X_train=X_enc,
        y_train=np.asarray(y),
        task="classification",
        metric=metric,
        estimator_list=[estimator],
        eval_method="cv",
        max_iter=max_iter,
        early_stop=True,
        retrain_full=False,
        seed=random_state,
        verbose=verbose,
    )
    if estimator == "xgboost":
        fit_kwargs["custom_hp"] = {
            "xgboost": {
                "scale_pos_weight": {"domain": 15.0, "init_value": 15.0},
                "max_delta_step": {"domain": 1.0, "init_value": 1.0},
            }
        }
    if time_budget is not None:
        fit_kwargs["time_budget"] = time_budget
    if groups is not None and len(pd.unique(pd.Series(groups))) > 1:
        fit_kwargs["split_type"] = "group"
        fit_kwargs["groups"] = np.asarray(groups)

    automl = AutoML()
    automl.fit(**fit_kwargs)
    return dict(automl.best_config), automl


# Balanced-accuracy scorer that consumes sample_weight, so the threshold search in
# TunedThresholdClassifierCV weights the upweighted dyad rows exactly as training
# does. Needs set_config(enable_metadata_routing=True) (set at import).
WEIGHTED_BALANCED_ACCURACY = make_scorer(balanced_accuracy_score).set_score_request(
    sample_weight=True
)


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    estimator: str,
    groups=None,
    params: dict | None = None,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> TunedThresholdClassifierCV:
    """Fit the full estimator (preprocessor + classifier + tuned threshold).

    Wraps the pipeline in a `TunedThresholdClassifierCV` picking the decision
    threshold that maximizes balanced accuracy over dyad-grouped CV (`F.make_cv`).
    """
    pipeline = build_pipeline(X, estimator, params)
    tuned = TunedThresholdClassifierCV(
        pipeline,
        scoring="balanced_accuracy",
        thresholds=100,
        cv=F.make_cv(groups, n_splits=n_splits, random_state=random_state),
        refit=True,
        random_state=random_state,
    )
    tuned.fit(X, y)
    return tuned


def train_model_weighted(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight,
    estimator: str,
    groups=None,
    params: dict | None = None,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> TunedThresholdClassifierCV:
    """`train_model`, but training rows carry a `sample_weight` (the upweight path).

    The classifier step is fit with per-row weights and the threshold scorer is
    weight-aware; sample_weight is routed through the stack via sklearn metadata
    routing. Same total weight mass as bootstrapping the dyad's rows, no resampling.
    """
    pipeline = build_pipeline(X, estimator, params)
    pipeline.named_steps["clf"].set_fit_request(sample_weight=True)
    tuned = TunedThresholdClassifierCV(
        pipeline,
        scoring=WEIGHTED_BALANCED_ACCURACY,
        thresholds=100,
        cv=F.make_cv(groups, n_splits=n_splits, random_state=random_state),
        refit=True,
        random_state=random_state,
    )
    tuned.fit(X, y, sample_weight=np.asarray(sample_weight, dtype=float))
    return tuned


def fit_model(
    train_df: pd.DataFrame,
    estimator: str,
    params: dict,
    cache_path: Path | None = None,
    sample_weight=None,
    response: str = RESPONSE_COLUMN,
    random_state: int = RANDOM_STATE,
) -> TunedThresholdClassifierCV:
    """Fit (or load from cache) the estimator with a fixed (population) FLAML config.

    HPO is NOT re-run here -- the hyperparameters come from a single population
    search, so a retrain is just a refit + threshold tune. With `sample_weight` the
    fit goes through `train_model_weighted` (the per-dyad upweight); otherwise the
    standard `train_model` (population / base model). If `cache_path` is given the
    fitted pipeline is serialized there and reused on later runs.
    """
    if cache_path is not None and Path(cache_path).exists():
        return F.load_model(str(cache_path))
    X, y = F.prep_X_y(train_df, response)
    groups = train_df["dyad"].to_numpy()
    if sample_weight is None:
        model = train_model(
            X, y, estimator, groups=groups, params=params, random_state=random_state
        )
    else:
        model = train_model_weighted(
            X,
            y,
            sample_weight,
            estimator,
            groups=groups,
            params=params,
            random_state=random_state,
        )
    if cache_path is not None:
        F.save_model(model, str(cache_path))
    return model


# --------------------------------------------------------------------------- #
# Metrics.
# --------------------------------------------------------------------------- #
def specificity(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    return tn / max(tn + fp, 1)


def metrics(true, proba, pred) -> dict:
    """Pooled metrics over a set of predictions (AUROC is NaN for a single class)."""
    true = np.asarray(true)
    auroc = roc_auc_score(true, proba) if len(np.unique(true)) > 1 else np.nan
    return {
        "n": int(len(true)),
        "positives": int(true.sum()),
        "auroc": auroc,
        "sensitivity": recall_score(true, pred, zero_division=0),
        "specificity": specificity(true, pred),
        "balanced_accuracy": balanced_accuracy_score(true, pred),
    }


def cumulative_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Per-week CUMULATIVE metrics from tidy predictions (rows tagged by `week`).

    Through week W pools every prediction made in weeks <= W (each made by whatever
    model was in force that week), matching the source notebook's cumulative curves.
    """
    weeks = sorted(int(w) for w in pred_df["week"].unique())
    rows = []
    for w in weeks:
        sub = pred_df[pred_df["week"] <= w]
        rows.append(
            {"week": w, **metrics(sub["y_true"], sub["y_proba"], sub["y_pred"])}
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Result + model caching.
# --------------------------------------------------------------------------- #
def _safe(name) -> str:
    """Make a dyad id safe to use as a path component."""
    return re.sub(r"[^0-9A-Za-z._-]", "_", str(name))


def predictions_path(
    intermediate_dir, estimator: str, feature_set: str, mode: str
) -> Path:
    return Path(intermediate_dir) / f"{estimator}_{feature_set}_{mode}.parquet"


def save_predictions(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_predictions(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def get_config(
    estimator: str,
    df_pop_fs: pd.DataFrame,
    cache_path: Path,
    response: str = RESPONSE_COLUMN,
    max_iter: int = 100,
    time_budget: int | None = 30,
    random_state: int = RANDOM_STATE,
    verbose: int = 0,
) -> dict:
    """Get (or compute + cache) the population FLAML config for (estimator, feature set).

    HPO runs once per (estimator, feature set) on the population; every fit for that
    combo reuses the config and only re-tunes the threshold.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        return joblib.load(cache_path)
    X, y = F.prep_X_y(df_pop_fs, response)
    config, _ = flaml_hpo(
        X,
        y,
        estimator,
        groups=df_pop_fs["dyad"].to_numpy(),
        max_iter=max_iter,
        time_budget=time_budget,
        random_state=random_state,
        verbose=verbose,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(config, cache_path)
    return config


ACTIVE_HOURS = (7, 20)  # therapy hours, inclusive


# --------------------------------------------------------------------------- #
# Weekly retrain / scoring loop (upweighting variant), estimator-generic.
# --------------------------------------------------------------------------- #
def eval_weekly(
    estimator: str,
    feature_set: str,
    mode: str,
    df_pop: pd.DataFrame,
    df_test: pd.DataFrame,
    params: dict,
    models_dir: Path,
    response: str = RESPONSE_COLUMN,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Walk test weeks in order, scoring each dyad with its current model.

    `mode="no_retrain"` scores every week with the single population model.
    `mode="retrain_dyad"` rebuilds each dyad's model before scoring week w on the
    population + that dyad's accumulated history (therapy_week < w), with the dyad's
    rows UPWEIGHTED by `level` (= number of test dyads; population rows weight 1) --
    same total weight mass as bootstrapping, no resampling.

    All fits reuse `params` (the population FLAML config); only the threshold is
    re-tuned. Every fitted pipeline is cached under `models_dir/<estimator>/
    <feature_set>/...`, so re-runs and the SHAP cells load weights instead of
    retraining. Returns tidy per-prediction rows:
    [estimator, feature_set, mode, week, y_true, y_proba, y_pred].
    """
    models_dir = Path(models_dir)
    base_cache = models_dir / estimator / feature_set / "population.joblib"
    base_model = fit_model(
        df_pop,
        estimator,
        params,
        cache_path=base_cache,
        response=response,
        random_state=random_state,
    )

    test_dyads = df_test["dyad"].unique()
    dyad_models = {d: base_model for d in test_dyads}
    level = len(test_dyads)

    weeks = sorted(int(w) for w in df_test["therapy_week"].unique() if w >= 0)
    first_week = weeks[0] if weeks else None

    records = []
    for week in tqdm(weeks, desc=f"{estimator}/{feature_set}/{mode}: weeks"):
        if mode == "retrain_dyad" and week != first_week:
            new_models = {}
            for dyad, dyad_df in tqdm(
                df_test.groupby("dyad"),
                total=len(test_dyads),
                desc=f"week {week:>2}: refit dyads",
                leave=False,
            ):
                history = dyad_df[dyad_df["therapy_week"] < week]
                if history.empty:
                    new_models[dyad] = dyad_models[dyad]
                    continue
                cache_path = (
                    models_dir
                    / estimator
                    / feature_set
                    / mode
                    / f"week_{week}"
                    / f"dyad_{_safe(dyad)}.joblib"
                )
                if cache_path.exists():
                    new_models[dyad] = F.load_model(str(cache_path))
                    continue
                # Upweight the dyad's history (one copy, weight `level`); weights
                # line up with combined's row order (pop first, then the dyad).
                combined = pd.concat([df_pop, history], ignore_index=True)
                weights = np.concatenate(
                    [np.ones(len(df_pop)), np.full(len(history), level)]
                )
                new_models[dyad] = fit_model(
                    combined,
                    estimator,
                    params,
                    cache_path=cache_path,
                    sample_weight=weights,
                    response=response,
                    random_state=random_state,
                )
            dyad_models = new_models

        week_df = df_test[df_test["therapy_week"] == week]
        for dyad, week_dyad_df in week_df.groupby("dyad"):
            g = week_dyad_df[
                week_dyad_df["ActivityDateTime"]
                .dt.tz_convert("America/Chicago")
                .dt.hour.between(*ACTIVE_HOURS)
            ]
            X, y = F.prep_X_y(g, response)
            model = dyad_models[dyad]
            if X.empty:
                continue
            records.append(
                pd.DataFrame(
                    {
                        "estimator": estimator,
                        "feature_set": feature_set,
                        "mode": mode,
                        "week": week,
                        "y_true": y.to_numpy(),
                        "y_proba": model.predict_proba(X)[:, 1],
                        "y_pred": np.asarray(model.predict(X)),
                    }
                )
            )

        # Print AUROC for the week
        week_records = pd.concat(
            records[len(records) - len(week_df.groupby("dyad")) :], ignore_index=True
        )
        auroc = roc_auc_score(week_records["y_true"], week_records["y_proba"])
        print(f"Week {week}: AUROC = {auroc:.4f}")

    return pd.concat(records, ignore_index=True)


# --------------------------------------------------------------------------- #
# HR-range reference baseline.
# --------------------------------------------------------------------------- #
class HrModel(BaseEstimator):
    """Predict a tantrum when hr_moving_avg_10m is in (low, high)."""

    def __init__(self, low=105, high=129):
        self.low, self.high = low, high

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        hr = X["hr_moving_avg_10m"]
        return ((hr > self.low) & (hr < self.high)).astype(int).to_numpy()

    def predict_proba(self, X):
        p = self.predict(X).astype(float)
        return np.column_stack([1 - p, p])


def eval_baseline(
    df_test: pd.DataFrame, response: str = RESPONSE_COLUMN
) -> pd.DataFrame:
    """Tidy per-prediction rows for the static HR-range model (same schema as eval_weekly)."""
    model = HrModel()
    records = []
    for week in sorted(int(w) for w in df_test["therapy_week"].unique() if w >= 0):
        g = df_test[
            (df_test["therapy_week"] == week)
            & (
                df_test["ActivityDateTime"]
                .dt.tz_convert("America/Chicago")
                .dt.hour.between(*ACTIVE_HOURS)
            )
        ]
        X, y = F.prep_X_y(g, response)
        records.append(
            pd.DataFrame(
                {
                    "estimator": "hr_baseline",
                    "feature_set": "hr_baseline",
                    "mode": "hr_baseline",
                    "week": week,
                    "y_true": y.to_numpy(),
                    "y_proba": model.predict_proba(X)[:, 1],
                    "y_pred": model.predict(X),
                }
            )
        )
    return pd.concat(records, ignore_index=True)


# --------------------------------------------------------------------------- #
# SHAP helpers (tree models only; used for the population XGBoost model).
# --------------------------------------------------------------------------- #
def shap_for_model(model, X: pd.DataFrame):
    """Positive-class TreeExplainer SHAP values over rows X.

    Reaches into the fitted TunedThresholdClassifierCV for its refitted pipeline,
    encodes X exactly as the pipeline does, and explains the tree step. Returns
    (shap_values, X_encoded) with post-encoder column names.
    """
    import shap

    pipe = model.estimator_  # refitted Pipeline inside TunedThresholdClassifierCV
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    X_enc = pd.DataFrame(pre.transform(X), columns=list(pre.get_feature_names_out()))
    sv = shap.TreeExplainer(clf).shap_values(X_enc)
    return sv, X_enc


def mean_abs_shap(sv, X_enc: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| per (encoded) feature -- the global-importance summary."""
    return pd.Series(np.abs(sv).mean(axis=0), index=X_enc.columns)
