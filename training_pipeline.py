from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    hdbscan = None


DATA_PATH = Path("social_ecommerce_data.csv")
ARTIFACT_PATH = Path("model_artifacts.pkl")
PROCESSED_PATH = Path("processed_data.csv")
METRICS_PATH = Path("model_metrics.json")

TARGET_COL = "label"
CATEGORY_COL = "category"
USER_ID_COL = "user_id"
ITEM_ID_COL = "item_id"

SELLER_CONTROLLABLE = [
    "title_length",
    "title_emo_score",
    "img_count",
    "has_video",
    "price",
    "discount_rate",
]
STATIC_USER_CONTEXT = [
    "age",
    "gender",
    "user_level",
    "purchase_freq",
    "total_spend",
    "register_days",
    "follow_num",
    "fans_num",
]
BEHAVIORAL_SESSION_FEATURES = [
    "is_follow_author",
    "add2cart",
    "coupon_received",
    "coupon_used",
    "pv_count",
    "last_click_gap",
    "purchase_intent",
    "freshness_score",
]
FORBIDDEN_LEAKY_FEATURES = [
    "like_num",
    "comment_num",
    "share_num",
    "collect_num",
    "interaction_rate",
    "social_influence",
]

LISTING_FEATURES = SELLER_CONTROLLABLE + STATIC_USER_CONTEXT + [CATEGORY_COL]
SESSION_FEATURES = LISTING_FEATURES + BEHAVIORAL_SESSION_FEATURES
PERSONA_CLUSTER_FEATURES = [
    "age",
    "gender",
    "user_level",
    "purchase_freq",
    "total_spend",
    "register_days",
    "follow_num",
    "fans_num",
]
CLUSTER_FEATURE_SETS = {
    "persona_baseline": PERSONA_CLUSTER_FEATURES,
    "persona_value_focus": [
        "purchase_freq",
        "total_spend",
        "user_level",
        "register_days",
        "follow_num",
        "fans_num",
    ],
    "persona_demographic_plus_value": [
        "age",
        "gender",
        "user_level",
        "register_days",
        "purchase_freq",
        "total_spend",
    ],
}
LOG_TRANSFORM_FEATURES = {"purchase_freq", "total_spend", "follow_num", "fans_num", "register_days"}


def log_step(message: str) -> None:
    print(f"[training_pipeline] {message}", flush=True)


@dataclass
class SplitBundle:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    split_name: str
    metadata: dict[str, Any]


def compute_sentiment_calibration(df: pd.DataFrame) -> dict[str, float]:
    scores = df["title_emo_score"].clip(1e-6, 1 - 1e-6).to_numpy()
    target_alpha, target_beta, _, _ = stats.beta.fit(scores, floc=0, fscale=1)
    # SnowNLP scores are often U-shaped around 0 and 1; use a generic source prior.
    source_alpha, source_beta = 0.8, 0.8
    return {
        "source_alpha": float(source_alpha),
        "source_beta": float(source_beta),
        "target_alpha": float(target_alpha),
        "target_beta": float(target_beta),
    }


def calibrate_sentiment(raw_score: float, sent_cal: dict[str, float]) -> float:
    clipped = np.clip(raw_score, 1e-6, 1 - 1e-6)
    percentile = stats.beta.cdf(clipped, sent_cal["source_alpha"], sent_cal["source_beta"])
    calibrated = stats.beta.ppf(percentile, sent_cal["target_alpha"], sent_cal["target_beta"])
    return float(calibrated)


def load_dataset() -> pd.DataFrame:
    log_step(f"Loading dataset from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df["event_index"] = np.arange(len(df))
    log_step(f"Dataset loaded with shape={df.shape}")
    return df


def build_preprocessor(feature_names: list[str]) -> ColumnTransformer:
    numeric_features = [col for col in feature_names if col != CATEGORY_COL]
    categorical_features = [col for col in feature_names if col == CATEGORY_COL]

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def build_model_pipeline(feature_names: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(feature_names)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )


def get_split_bundles(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> list[SplitBundle]:
    log_step("Preparing evaluation splits: stratified_random, time_based, group_based_user")
    bundles: list[SplitBundle] = []

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    strat_train_idx, strat_test_idx = next(sss.split(X, y))
    bundles.append(
        SplitBundle(
            X.iloc[strat_train_idx].copy(),
            X.iloc[strat_test_idx].copy(),
            y.iloc[strat_train_idx].copy(),
            y.iloc[strat_test_idx].copy(),
            "stratified_random",
            {"strategy": "StratifiedShuffleSplit", "test_size": 0.2},
        )
    )

    sorted_idx = X.sort_values("event_index").index
    split_idx = int(len(sorted_idx) * 0.8)
    time_train_idx = sorted_idx[:split_idx]
    time_test_idx = sorted_idx[split_idx:]
    bundles.append(
        SplitBundle(
            X.loc[time_train_idx].copy(),
            X.loc[time_test_idx].copy(),
            y.loc[time_train_idx].copy(),
            y.loc[time_test_idx].copy(),
            "time_based",
            {
                "strategy": "chronological_index",
                "time_col": None,
                "note": "Dataset has no explicit timestamp; row order is used as a temporal proxy.",
            },
        )
    )

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    group_train_idx, group_test_idx = next(gss.split(X, y, groups=groups))
    bundles.append(
        SplitBundle(
            X.iloc[group_train_idx].copy(),
            X.iloc[group_test_idx].copy(),
            y.iloc[group_train_idx].copy(),
            y.iloc[group_test_idx].copy(),
            "group_based_user",
            {"strategy": "GroupShuffleSplit", "group_col": USER_ID_COL, "test_size": 0.2},
        )
    )

    return bundles


def score_predictions(y_true: pd.Series, proba: np.ndarray) -> dict[str, float]:
    return score_predictions_at_threshold(y_true, proba, threshold=0.5)


def score_predictions_at_threshold(
    y_true: pd.Series,
    proba: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    preds = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "f1": float(f1_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "brier": float(brier_score_loss(y_true, proba)),
        "positive_rate": float(np.mean(preds)),
        "threshold": float(threshold),
    }


def tune_threshold(
    y_true: pd.Series,
    proba: np.ndarray,
    objective: str = "f1",
) -> dict[str, float]:
    candidate_thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0.1, 0.9, 33),
                np.quantile(proba, np.linspace(0.05, 0.95, 19)),
            ]
        )
    )
    best_threshold = 0.5
    best_score = -1.0
    best_metrics: dict[str, float] | None = None

    for threshold in candidate_thresholds:
        metrics = score_predictions_at_threshold(y_true, proba, float(threshold))
        score = metrics[objective]
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics

    return {
        "threshold": float(best_threshold),
        "objective": objective,
        "validation_score": float(best_score),
        "validation_metrics": best_metrics or score_predictions_at_threshold(y_true, proba, 0.5),
    }


def fit_and_evaluate_task(
    df: pd.DataFrame,
    task_name: str,
    feature_names: list[str],
) -> dict[str, Any]:
    log_step(f"Starting task `{task_name}` with {len(feature_names)} features")
    X = df[[USER_ID_COL, ITEM_ID_COL, "event_index", *feature_names]].copy()
    y = df[TARGET_COL].astype(int).copy()
    groups = df[USER_ID_COL].copy()
    splits = get_split_bundles(X, y, groups)

    split_results: dict[str, Any] = {}
    for split in splits:
        log_step(f"Task `{task_name}`: fitting split `{split.split_name}`")
        train_features = split.X_train[feature_names]
        test_features = split.X_test[feature_names]
        pipeline = build_model_pipeline(feature_names)
        pipeline.fit(train_features, split.y_train)
        raw_proba = pipeline.predict_proba(test_features)[:, 1]

        calibrated = CalibratedClassifierCV(estimator=clone(pipeline), method="sigmoid", cv=3)
        calibrated.fit(train_features, split.y_train)
        calibrated_proba = calibrated.predict_proba(test_features)[:, 1]

        threshold_sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tune_train_idx, tune_valid_idx = next(threshold_sss.split(train_features, split.y_train))
        tune_train_features = train_features.iloc[tune_train_idx]
        tune_valid_features = train_features.iloc[tune_valid_idx]
        tune_train_y = split.y_train.iloc[tune_train_idx]
        tune_valid_y = split.y_train.iloc[tune_valid_idx]

        threshold_pipeline = build_model_pipeline(feature_names)
        threshold_pipeline.fit(tune_train_features, tune_train_y)
        threshold_calibrated = CalibratedClassifierCV(
            estimator=clone(threshold_pipeline), method="sigmoid", cv=3
        )
        threshold_calibrated.fit(tune_train_features, tune_train_y)
        tune_valid_proba = threshold_calibrated.predict_proba(tune_valid_features)[:, 1]
        threshold_info = tune_threshold(tune_valid_y, tune_valid_proba, objective="f1")
        tuned_threshold = threshold_info["threshold"]
        tuned_metrics = score_predictions_at_threshold(split.y_test, calibrated_proba, tuned_threshold)

        split_results[split.split_name] = {
            "metadata": split.metadata,
            "raw_metrics": score_predictions(split.y_test, raw_proba),
            "calibrated_metrics": score_predictions(split.y_test, calibrated_proba),
            "threshold_tuning": threshold_info,
            "tuned_metrics": tuned_metrics,
        }
        log_step(
            "Task `{}` split `{}` complete: raw_auc={:.4f}, calibrated_auc={:.4f}, "
            "calibrated_brier={:.4f}, tuned_threshold={:.3f}, tuned_f1={:.4f}".format(
                task_name,
                split.split_name,
                split_results[split.split_name]["raw_metrics"]["roc_auc"],
                split_results[split.split_name]["calibrated_metrics"]["roc_auc"],
                split_results[split.split_name]["calibrated_metrics"]["brier"],
                tuned_threshold,
                tuned_metrics["f1"],
            )
        )

    log_step(f"Task `{task_name}`: fitting full-data pipeline")
    final_pipeline = build_model_pipeline(feature_names)
    final_pipeline.fit(df[feature_names], y)

    log_step(f"Task `{task_name}`: fitting calibrated full-data model")
    final_calibrated = CalibratedClassifierCV(estimator=clone(final_pipeline), method="sigmoid", cv=3)
    final_calibrated.fit(df[feature_names], y)
    log_step(f"Task `{task_name}` complete")

    return {
        "task_name": task_name,
        "feature_names": feature_names,
        "split_results": split_results,
        "pipeline": final_pipeline,
        "calibrated_model": final_calibrated,
    }


def compute_category_benchmarks(df: pd.DataFrame, feature_names: list[str]) -> dict[str, dict[str, float]]:
    purchased = df[df[TARGET_COL] == 1]
    benchmarks: dict[str, dict[str, float]] = {}
    for category, group in purchased.groupby(CATEGORY_COL):
        benchmarks[category] = {}
        for feature in feature_names:
            if feature == CATEGORY_COL:
                continue
            if feature == "has_video":
                benchmarks[category][feature] = float(group[feature].mode().iloc[0])
            else:
                benchmarks[category][feature] = float(group[feature].median())
    return benchmarks


def fit_clusterers(df: pd.DataFrame) -> dict[str, Any]:
    log_step("Starting persona clustering")
    evaluations: list[dict[str, Any]] = []
    best_score = -1.0
    best_size_balance = -1.0
    best_labels: np.ndarray | None = None
    best_model: Any = None
    best_feature_set_name = "persona_baseline"
    best_feature_set = PERSONA_CLUSTER_FEATURES
    best_scaler = None

    def transform_cluster_frame(frame: pd.DataFrame, feature_set: list[str], use_log: bool) -> pd.DataFrame:
        transformed = frame[feature_set].copy()
        if use_log:
            for col in feature_set:
                if col in LOG_TRANSFORM_FEATURES:
                    transformed[col] = np.log1p(transformed[col].clip(lower=0))
        return transformed

    def compute_size_balance(labels: np.ndarray) -> float:
        counts = pd.Series(labels).value_counts(normalize=True)
        return float(counts.min() / counts.max()) if len(counts) > 0 else 0.0

    def is_better_candidate(score: float, size_balance: float, algorithm: str, k: int | None) -> bool:
        nonlocal best_score, best_size_balance
        if score > best_score + 0.02:
            return True
        if abs(score - best_score) <= 0.02:
            if 4 <= (k or 0) <= 5 and size_balance >= best_size_balance - 0.05:
                return True
            if size_balance > best_size_balance + 0.05:
                return True
            if algorithm == "kmeans" and best_model and best_model["algorithm"] != "kmeans":
                return True
        return False

    for feature_set_name, feature_set in CLUSTER_FEATURE_SETS.items():
        for use_log in (False, True):
            log_step(
                f"Clustering feature set `{feature_set_name}` with "
                f"{'log1p + scaling' if use_log else 'raw scaling'}"
            )
            X_persona = transform_cluster_frame(df, feature_set, use_log)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_persona)

            for use_pca in (False, True):
                log_step(
                    f"Clustering branch: {feature_set_name} / "
                    f"{'PCA + KMeans' if use_pca else 'KMeans'}"
                )
                transformed = X_scaled
                pca_model = None
                explained_variance = None
                if use_pca:
                    from sklearn.decomposition import PCA

                    pca_model = PCA(n_components=0.9, random_state=42)
                    transformed = pca_model.fit_transform(X_scaled)
                    explained_variance = float(np.sum(pca_model.explained_variance_ratio_))
                    log_step(
                        f"PCA retained {transformed.shape[1]} components with explained_variance={explained_variance:.4f}"
                    )

                for k in range(2, 7):
                    log_step(
                        f"Evaluating {feature_set_name} / "
                        f"{'PCA + ' if use_pca else ''}KMeans with k={k}"
                    )
                    model = KMeans(n_clusters=k, random_state=42, n_init=20)
                    labels = model.fit_predict(transformed)
                    silhouette = silhouette_score(transformed, labels)
                    calinski = calinski_harabasz_score(transformed, labels)
                    davies = davies_bouldin_score(transformed, labels)
                    size_balance = compute_size_balance(labels)
                    cluster_count = int(pd.Series(labels).nunique())
                    evaluations.append(
                        {
                            "algorithm": "kmeans",
                            "feature_set": feature_set_name,
                            "use_log": use_log,
                            "use_pca": use_pca,
                            "k": k,
                            "silhouette": float(silhouette),
                            "calinski_harabasz": float(calinski),
                            "davies_bouldin": float(davies),
                            "size_balance": float(size_balance),
                            "cluster_count": cluster_count,
                            "inertia": float(model.inertia_),
                            "explained_variance": explained_variance,
                        }
                    )
                    if is_better_candidate(float(silhouette), float(size_balance), "kmeans", k):
                        best_score = float(silhouette)
                        best_size_balance = float(size_balance)
                        best_labels = labels
                        best_model = {
                            "algorithm": "kmeans",
                            "model": model,
                            "use_pca": use_pca,
                            "pca_model": pca_model,
                            "feature_set_name": feature_set_name,
                            "use_log": use_log,
                        }
                        best_feature_set_name = feature_set_name
                        best_feature_set = feature_set
                        best_scaler = scaler
                        log_step(
                            "New best clustering candidate: feature_set={}, algorithm=kmeans, "
                            "use_log={}, use_pca={}, k={}, silhouette={:.4f}, size_balance={:.4f}".format(
                                feature_set_name, use_log, use_pca, k, best_score, best_size_balance
                            )
                        )

            if hdbscan is not None:
                log_step(f"Evaluating HDBSCAN fallback candidate for feature set `{feature_set_name}`")
                min_cluster_size = max(100, int(len(df) * 0.02))
                hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
                hdb_labels = hdb.fit_predict(X_scaled)
                valid_mask = hdb_labels != -1
                unique_valid = np.unique(hdb_labels[valid_mask]) if valid_mask.sum() > 1 else []
                if valid_mask.sum() > 1 and len(unique_valid) > 1:
                    hdb_silhouette = silhouette_score(X_scaled[valid_mask], hdb_labels[valid_mask])
                    hdb_calinski = calinski_harabasz_score(X_scaled[valid_mask], hdb_labels[valid_mask])
                    hdb_davies = davies_bouldin_score(X_scaled[valid_mask], hdb_labels[valid_mask])
                else:
                    hdb_silhouette = -1.0
                    hdb_calinski = -1.0
                    hdb_davies = np.inf
                size_balance = compute_size_balance(hdb_labels[valid_mask]) if valid_mask.any() else 0.0
                evaluations.append(
                    {
                        "algorithm": "hdbscan",
                        "feature_set": feature_set_name,
                        "use_log": use_log,
                        "use_pca": False,
                        "k": None,
                        "silhouette": float(hdb_silhouette),
                        "calinski_harabasz": float(hdb_calinski),
                        "davies_bouldin": float(hdb_davies),
                        "size_balance": float(size_balance),
                        "cluster_count": int(len(unique_valid)),
                        "noise_rate": float(np.mean(hdb_labels == -1)),
                    }
                )
                if is_better_candidate(float(hdb_silhouette), float(size_balance), "hdbscan", None):
                    best_score = float(hdb_silhouette)
                    best_size_balance = float(size_balance)
                    best_labels = hdb_labels
                    best_model = {
                        "algorithm": "hdbscan",
                        "model": hdb,
                        "use_pca": False,
                        "pca_model": None,
                        "feature_set_name": feature_set_name,
                        "use_log": use_log,
                    }
                    best_feature_set_name = feature_set_name
                    best_feature_set = feature_set
                    best_scaler = scaler
                    log_step(
                        "HDBSCAN selected as best clustering candidate: feature_set={}, "
                        "use_log={}, silhouette={:.4f}, size_balance={:.4f}".format(
                            feature_set_name, use_log, best_score, best_size_balance
                        )
                    )
            else:
                log_step("HDBSCAN not installed; skipping density-based clustering candidate")

    if best_labels is None or best_model is None:
        raise RuntimeError("Failed to fit a clustering model")

    log_step(
        f"Selected clustering algorithm={best_model['algorithm']}, feature_set={best_feature_set_name}, "
        f"use_log={best_model['use_log']}, use_pca={best_model['use_pca']}, "
        f"clusters={len(pd.Series(best_labels)[pd.Series(best_labels) >= 0].unique())}"
    )

    output_labels = pd.Series(best_labels, index=df.index, name="user_cluster")
    clusterable = output_labels[output_labels >= 0]
    profile_df = (
        df.loc[clusterable.index]
        .assign(user_cluster=clusterable)
        .groupby("user_cluster")[best_feature_set + [TARGET_COL]]
        .mean()
    )
    counts = clusterable.value_counts().sort_index()
    profile_df["count"] = counts
    profile_df["purchase_rate"] = (
        df.loc[clusterable.index].assign(user_cluster=clusterable).groupby("user_cluster")[TARGET_COL].mean()
    )

    overall = df[best_feature_set].mean(numeric_only=True)
    if "total_spend" in profile_df.columns and "total_spend" in overall.index and overall["total_spend"] != 0:
        profile_df["avg_spend_index"] = profile_df["total_spend"] / overall["total_spend"] * 100
    else:
        profile_df["avg_spend_index"] = 100.0
    if "purchase_freq" in profile_df.columns and "purchase_freq" in overall.index and overall["purchase_freq"] != 0:
        profile_df["purchase_freq_index"] = profile_df["purchase_freq"] / overall["purchase_freq"] * 100
    else:
        profile_df["purchase_freq_index"] = 100.0

    cluster_names: dict[int, str] = {}
    cluster_strategies: dict[int, list[str]] = {}
    cluster_medians: dict[int, dict[str, float]] = {}
    cluster_weights: dict[int, float] = {}
    for cluster_id, row in profile_df.iterrows():
        spend_idx = row["avg_spend_index"]
        freq_idx = row["purchase_freq_index"]
        if spend_idx >= 180 and freq_idx >= 130:
            name = "Champions"
            strategy = [
                "Prioritize premium positioning and richer visuals.",
                "Use modest discounts; avoid eroding willingness to pay.",
            ]
        elif spend_idx >= 180:
            name = "Big Spenders"
            strategy = [
                "Lead with quality, exclusivity, and trust signals.",
                "Test bundles instead of aggressive coupons.",
            ]
        elif freq_idx >= 120:
            name = "Loyal Regulars"
            strategy = [
                "Highlight consistency and repeat-purchase convenience.",
                "Use loyalty nudges and light couponing.",
            ]
        elif row["purchase_rate"] < df[TARGET_COL].mean():
            name = "At Risk"
            strategy = [
                "Use stronger discount framing and simpler titles.",
                "Reduce friction with clearer benefits and more images.",
            ]
        else:
            name = "New / Active"
            strategy = [
                "Optimize first-impression elements such as title and hero images.",
                "Use onboarding-style coupons selectively.",
            ]
        cluster_names[int(cluster_id)] = name
        cluster_strategies[int(cluster_id)] = strategy
        cluster_medians[int(cluster_id)] = {
            col: float(df.loc[clusterable.index]
                       .assign(user_cluster=clusterable)
                       .groupby("user_cluster")[col]
                       .median()
                       .loc[cluster_id])
            for col in STATIC_USER_CONTEXT + BEHAVIORAL_SESSION_FEATURES
            if col in df.columns
        }
        cluster_weights[int(cluster_id)] = float(counts.loc[cluster_id] / len(clusterable))

    return {
        "labels": output_labels,
        "profiles": profile_df,
        "names": cluster_names,
        "strategies": cluster_strategies,
        "medians": cluster_medians,
        "weights": cluster_weights,
        "evaluations": evaluations,
        "model_bundle": best_model,
        "scaler": best_scaler,
        "features": best_feature_set,
        "feature_set_name": best_feature_set_name,
        "available_feature_sets": CLUSTER_FEATURE_SETS,
    }


def build_processed_dataframe(df: pd.DataFrame, cluster_labels: pd.Series) -> pd.DataFrame:
    processed = df.copy()
    category_mapping = {cat: idx for idx, cat in enumerate(sorted(processed[CATEGORY_COL].unique()))}
    processed["category_encoded"] = processed[CATEGORY_COL].map(category_mapping)
    processed["user_cluster"] = cluster_labels
    return processed


def summarize_comparison(tasks: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for task_name, task in tasks.items():
        for split_name, result in task["split_results"].items():
            rows.append(
                {
                    "task": task_name,
                    "split": split_name,
                    "raw_roc_auc": result["raw_metrics"]["roc_auc"],
                    "calibrated_roc_auc": result["calibrated_metrics"]["roc_auc"],
                    "raw_f1": result["raw_metrics"]["f1"],
                    "calibrated_f1": result["calibrated_metrics"]["f1"],
                    "tuned_f1": result["tuned_metrics"]["f1"],
                    "tuned_precision": result["tuned_metrics"]["precision"],
                    "tuned_recall": result["tuned_metrics"]["recall"],
                    "tuned_positive_rate": result["tuned_metrics"]["positive_rate"],
                    "tuned_threshold": result["tuned_metrics"]["threshold"],
                    "raw_brier": result["raw_metrics"]["brier"],
                    "calibrated_brier": result["calibrated_metrics"]["brier"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    log_step("Pipeline run started")
    df = load_dataset()
    log_step("Computing sentiment calibration")
    sentiment_calibration = compute_sentiment_calibration(df)

    listing_task = fit_and_evaluate_task(df, "listing_time", LISTING_FEATURES)
    session_task = fit_and_evaluate_task(df, "session_time", SESSION_FEATURES)
    tasks = {
        "listing_time": listing_task,
        "session_time": session_task,
    }

    clustering = fit_clusterers(df)
    log_step("Building processed dataframe and summary artifacts")
    processed = build_processed_dataframe(df, clustering["labels"])
    category_mapping = {
        cat: idx for idx, cat in enumerate(sorted(df[CATEGORY_COL].unique()))
    }
    benchmarks = compute_category_benchmarks(df, SELLER_CONTROLLABLE)
    comparison_df = summarize_comparison(tasks)

    artifacts = {
        # Legacy aliases for current app behavior.
        "model": listing_task["calibrated_model"],
        "model_name": "Listing-Time Random Forest (Calibrated)",
        "feature_cols": LISTING_FEATURES.copy(),
        "benchmarks": benchmarks,
        "comparison_df": comparison_df,
        "SELLER_CONTROLLABLE": SELLER_CONTROLLABLE,
        "category_mapping": category_mapping,
        "cluster_profiles": clustering["profiles"],
        "cluster_names": clustering["names"],
        "cluster_medians": clustering["medians"],
        "cluster_weights": clustering["weights"],
        "OPTIMAL_K": int(len(clustering["profiles"])),
        "scaler_kmeans": clustering["scaler"],
        "user_features_for_cluster": clustering["features"],
        "kmeans": clustering["model_bundle"]["model"]
        if clustering["model_bundle"]["algorithm"] == "kmeans"
        else None,
        "sentiment_calibration": sentiment_calibration,
        "task_definitions": {
            "listing_time": {
                "available_features": LISTING_FEATURES,
                "forbidden_features": FORBIDDEN_LEAKY_FEATURES + [
                    feat for feat in BEHAVIORAL_SESSION_FEATURES if feat not in ["coupon_received"]
                ],
                "description": "Prediction at listing creation time with deployable features only.",
            },
            "session_time": {
                "available_features": SESSION_FEATURES,
                "forbidden_features": FORBIDDEN_LEAKY_FEATURES,
                "description": "Prediction during an active user session with behavioral context.",
            },
        },
        "feature_groups": {
            "seller_controllable": SELLER_CONTROLLABLE,
            "static_user_context": STATIC_USER_CONTEXT,
            "behavioral_session_features": BEHAVIORAL_SESSION_FEATURES,
            "forbidden_leaky_features": FORBIDDEN_LEAKY_FEATURES,
        },
        "models": {
            "listing_time": {
                "pipeline": listing_task["pipeline"],
                "calibrated_model": listing_task["calibrated_model"],
                "feature_names": LISTING_FEATURES,
                "evaluation": listing_task["split_results"],
            },
            "session_time": {
                "pipeline": session_task["pipeline"],
                "calibrated_model": session_task["calibrated_model"],
                "feature_names": SESSION_FEATURES,
                "evaluation": session_task["split_results"],
            },
        },
        "clustering": {
            "evaluations": clustering["evaluations"],
            "algorithm": clustering["model_bundle"]["algorithm"],
            "use_pca": clustering["model_bundle"]["use_pca"],
            "use_log": clustering["model_bundle"]["use_log"],
            "feature_set_name": clustering["feature_set_name"],
            "selected_features": clustering["features"],
            "available_feature_sets": clustering["available_feature_sets"],
            "pca_model": clustering["model_bundle"]["pca_model"],
            "strategies": clustering["strategies"],
        },
    }

    log_step(f"Saving artifacts to {ARTIFACT_PATH}")
    with ARTIFACT_PATH.open("wb") as f:
        pickle.dump(artifacts, f)
    log_step(f"Saving processed data to {PROCESSED_PATH}")
    processed.to_csv(PROCESSED_PATH, index=False)
    log_step(f"Saving metrics summary to {METRICS_PATH}")
    METRICS_PATH.write_text(
        json.dumps(
            {
                "tasks": {
                    name: task["split_results"] for name, task in tasks.items()
                },
                "clustering": {
                    "evaluations": clustering["evaluations"],
                    "selected_algorithm": clustering["model_bundle"]["algorithm"],
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    log_step("Pipeline run complete")


if __name__ == "__main__":
    main()
