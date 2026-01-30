from pathlib import Path

import numpy as np
import pandas as pd

from util import prep_X_y
import pickle
from typing import Any, Literal

from flaml import AutoML
from sklearn import clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    make_scorer,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    FixedThresholdClassifier,
    KFold,
    PredefinedSplit,
    TunedThresholdClassifierCV,
)
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

from hr_model import HrModel
from util import FeatureSetDataFrames


def specificity_score(y_true, y_pred) -> float:
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    if (tn + fp) == 0:
        return float("nan")
    return tn / (tn + fp)


def find_threshold_ref_specificity(model, X, y, *, verbose=False) -> float:
    hr_model = HrModel()

    ref_y_pred = hr_model.predict(X)
    ref_specificity = specificity_score(y, ref_y_pred)
    ref_recall = recall_score(y, ref_y_pred, zero_division=np.nan)

    y_pred_proba = model.predict_proba(X)
    thresholds = np.logspace(-5, 0, 100)

    for th in thresholds:
        y_pred = (y_pred_proba[:, 1] >= th).astype(int)
        specificity = specificity_score(y, y_pred)
        recall = recall_score(y, y_pred, zero_division=np.nan)
        if specificity >= ref_specificity:
            if verbose:
                print(f"Selected threshold: {th}")
                print(f"Selected specificity: {specificity}, recall: {recall}")
                print(f"Reference specificity: {ref_specificity}, recall: {ref_recall}")
            return th

    raise ValueError("No suitable threshold found!")


def find_threshold_ref_specificity_cv(model, df_train, df_val, cv, verbose=False):
    X, y = prep_X_y(pd.concat([df_train, df_val]), "tantrum_within_60m")
    thresholds = []
    for train_idx, val_idx in cv.split(df_train):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_tr, y_tr)
        th: int | float = find_threshold_ref_specificity(
            model, X_val, y_val, verbose=verbose
        )
        thresholds.append(th)

    best_threshold = float(np.mean(thresholds))
    if verbose:
        print(f"Cross-validated best threshold: {best_threshold:.4f}")

    model.fit(X, y)
    return FixedThresholdClassifier(model, threshold=best_threshold)


def youdens_j_score(y_true, y_pred) -> float:
    sensitivity = recall_score(y_true, y_pred, zero_division=np.nan)
    specificity = specificity_score(y_true, y_pred)
    return sensitivity + specificity - 1


TuningMethod = (
    Literal["balanced_accuracy"]
    | Literal["youden_index"]
    | Literal["cost_sensitive"]
    | Literal["ref_specificity"]
)


def create_dyad_cv(df_train: pd.DataFrame, n_splits: int = 5) -> PredefinedSplit:
    # Create 5-fold CV based on "dyad"
    dyad_labels = df_train["dyad"]
    skf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = np.zeros(len(df_train), dtype=int)
    for fold_idx, (_, val_idx) in enumerate(
        skf.split(np.zeros(len(dyad_labels)), dyad_labels)
    ):
        folds[val_idx] = fold_idx
    return PredefinedSplit(folds)


def cost_sensitive_score(y, y_pred):
    cm = confusion_matrix(y, y_pred, labels=[0, 1])

    gain_matrix = np.array(
        [
            [0, -1],  # gain for false positives
            [-200, 0],  # gain for false negatives
        ]
    )
    return np.sum(cm * gain_matrix)


def train_model(
    df_train,
    df_val,
    model: Any,
    tuning_method: TuningMethod,
):
    model = clone(model)
    X_train, y_train = prep_X_y(pd.concat([df_train, df_val]), "tantrum_within_60m")
    cv = (
        create_dyad_cv(df_train)
        if len(df_val) == 0
        else PredefinedSplit(test_fold=[-1] * len(df_train) + [0] * len(df_val))
    )

    thresholds = np.logspace(-5, 0, 100)
    if tuning_method == "balanced_accuracy":
        tuned_model = TunedThresholdClassifierCV(
            model,
            scoring="balanced_accuracy",
            thresholds=thresholds,
            cv=cv,
            n_jobs=-1,
        )
        tuned_model.fit(X_train, y_train)
        return tuned_model
    elif tuning_method == "youden_index":
        tuned_model = TunedThresholdClassifierCV(
            model,
            scoring=make_scorer(youdens_j_score),
            thresholds=thresholds,
            cv=cv,
            n_jobs=-1,
        )
        tuned_model.fit(X_train, y_train)
        return tuned_model
    elif tuning_method == "cost_sensitive":
        tuned_model = TunedThresholdClassifierCV(
            model,
            scoring=make_scorer(cost_sensitive_score),
            thresholds=thresholds,
            cv=cv,
            n_jobs=-1,
        )
        tuned_model.fit(X_train, y_train)
        return tuned_model
    elif tuning_method == "ref_specificity":
        return find_threshold_ref_specificity_cv(
            model, df_train, df_val, cv, verbose=False
        )
    else:
        raise ValueError(f"Unknown tuning method: {tuning_method}")


def bootstrap(df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    boot_df = pd.DataFrame()
    for _ in range(n_samples):
        boot_df = pd.concat(
            [
                boot_df,
                df.sample(frac=1, replace=True, random_state=None),
            ]
        )
    return boot_df


def train_and_get_dyad_models(
    df_population: pd.DataFrame,
    df_test: pd.DataFrame,
    mode: str,
    week: int,
    dyad_models: dict[str, TunedThresholdClassifierCV],
    tuning_method: TuningMethod,
):
    min_week = df_test["therapy_week"].min()
    if week == min_week:
        return dyad_models

    new_dyad_models = {}
    bootstrap_level = df_test["dyad"].nunique()
    match mode:
        case "no_retrain":
            return dyad_models
        case "retrain_dyad":
            for dyad, dyad_df in tqdm(df_test.groupby("dyad"), leave=False):
                add_df = dyad_df[dyad_df["therapy_week"] < week]
                add_df = bootstrap(add_df, bootstrap_level)

                df_train = pd.concat(
                    [df_population, add_df[add_df["therapy_week"] < week - 1]]
                )
                df_val = add_df[add_df["therapy_week"] == week - 1]
                new_dyad_models[dyad] = train_model(
                    df_train,
                    df_val,
                    dyad_models[dyad].estimator,
                    tuning_method,
                )
        case _:
            raise ValueError(f"Unknown mode: {mode}")

    return new_dyad_models


def retrain_and_predict(
    base_model,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    mode: str,
    tuning_method: TuningMethod,
):
    dyad_models = {d: base_model for d in df_test["dyad"].unique()}
    weekly_results = []
    weekly_models = []
    weeks = df_test["therapy_week"].unique()
    weeks_iter = sorted(weeks[weeks >= 0])
    for week in tqdm(weeks_iter):
        dyad_models = train_and_get_dyad_models(
            df_train,
            df_test,
            mode,
            week,
            dyad_models=dyad_models,
            tuning_method=tuning_method,
        )
        weekly_models.append((week, dyad_models))
        week_df = df_test[df_test["therapy_week"] == week]

        week_pred_proba = np.empty((0, 2))
        week_preds = np.array([])
        week_trues = np.array([])
        week_thresholds = np.array([])

        for dyad, dyad_week_df in week_df.groupby("dyad"):
            X, y = prep_X_y(dyad_week_df, "tantrum_within_60m")
            model = dyad_models[dyad]

            y_pred_proba = model.predict_proba(X)
            y_pred = model.predict(X)

            week_pred_proba = np.concatenate([week_pred_proba, y_pred_proba])
            week_preds = np.concatenate([week_preds, y_pred])
            week_trues = np.concatenate([week_trues, y.values])
            threshold = (
                model.best_threshold_
                if isinstance(model, TunedThresholdClassifierCV)
                else 0.5
            )
            week_thresholds = np.append(week_thresholds, threshold)

        print(
            f"Week {week} AUROC: {roc_auc_score(week_trues, week_pred_proba[:, 1])} sensitivity: {recall_score(week_trues, week_preds)}, specificity: {specificity_score(week_trues, week_preds)}"
        )
        weekly_results.append(
            (week, week_pred_proba, week_preds, week_trues, week_thresholds)
        )

    return weekly_results, weekly_models


supersets_to_test = [
    ["watch"],
    ["watch", "demographic"],
    ["watch", "medical"],
    ["watch", "demographic", "medical"],
]
feature_supersets = {
    "watch": [
        "hr",
        "activity",
        "sleep",
        "stress",
        "overnight_hrv",
    ],
    "demographic": [
        "child_demo",
        "parent_demo",
    ],
    "medical": [
        "medical",
        "therapy",
    ],
}


def eval_model_on_feature_sets(
    supersets_to_test: list[list[str]],
    dfs: FeatureSetDataFrames,
    weeks: tuple[int, int],
    active_hours: tuple[int, int],
    estimator: str,
    mode: str,
    tuning_method: TuningMethod,
    verbose: bool = False,
) -> dict[str, Any]:
    feature_set_results = {}
    for supersets in supersets_to_test:
        name = "_".join(supersets)
        print(f"Feature sets: {name}")

        features = [fs for superset in supersets for fs in feature_supersets[superset]]
        combined_df = pd.concat(
            [
                dfs["index"],
                dfs["response"],
            ]
            + [dfs[fs] for fs in features],
            axis=1,
        )
        combined_df = combined_df[
            (
                combined_df["ActivityDateTime"].dt.hour.between(
                    active_hours[0], active_hours[1]
                )
            )
        ]
        combined_df = combined_df[
            combined_df["therapy_week"].between(weeks[0], weeks[1])
        ]

        df_train = combined_df[combined_df["Arm_Sham"]]
        df_test = combined_df[~combined_df["Arm_Sham"]]

        # Create 5-fold CV based on "dyad"
        X_train, y_train = prep_X_y(df_train, "tantrum_within_60m")
        automl = AutoML()
        automl.fit(
            X_train,
            y_train,
            max_iter=100,
            early_stop=True,
            estimator_list=[estimator],
            eval_method="cv",
            split_type="group",
            groups=df_train["dyad"],
            verbose=verbose,
        )

        if estimator == "lrl2":
            # Impute missing following https://microsoft.github.io/FLAML/docs/FAQ/#how-does-flaml-handle-missing-values
            numeric_transformer = SimpleImputer(strategy="median")
            categorical_transformer = SimpleImputer(
                strategy="constant", fill_value="__NAN__"
            )
            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "num",
                        numeric_transformer,
                        make_column_selector(dtype_include="number"),
                    ),
                    (
                        "cat",
                        categorical_transformer,
                        make_column_selector(dtype_include="object"),
                    ),
                ]
            )
            model = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("classifier", automl.model.estimator),
                ]
            )
        elif estimator == "xgboost":
            model = automl.model.estimator
        else:
            raise ValueError(f"Unknown estimator: {estimator}")

        tuned_model = train_model(
            df_train,
            pd.DataFrame(),
            model,
            tuning_method=tuning_method,
        )
        results, models = retrain_and_predict(
            tuned_model, df_train, df_test, mode=mode, tuning_method=tuning_method
        )

        data_dir = Path("./intermediate_data")
        data_dir.mkdir(parents=True, exist_ok=True)
        with open(
            data_dir / f"{estimator}_{tuning_method}_{mode}_{name}_results.pkl", "wb"
        ) as f:
            pickle.dump(results, f)

        feature_set_results[name] = results
    return feature_set_results
