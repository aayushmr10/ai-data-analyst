"""
Automatic ML engine.

Given a dataframe and optional target column:
  - No target             -> clustering (KMeans)
  - Target is numeric      -> regression (Linear Regression + Random Forest, best picked by R^2)
  - Target is categorical  -> classification (Logistic Regression + Random Forest, best picked by accuracy)

Deliberately uses scikit-learn only (no XGBoost) to keep the install
lightweight and avoid platform-specific build issues; RandomForest
covers the "non-linear, robust default" role XGBoost would have played.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    mean_squared_error, r2_score, precision_score, recall_score,
    confusion_matrix, silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from app.utils.serialize import safe_json


def _prepare_features(df: pd.DataFrame, target: str | None) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Encode categoricals, drop unusable columns. Returns (X, used_columns, dropped_columns)."""
    feature_df = df.drop(columns=[target]) if target else df.copy()
    dropped = []
    used = []
    X = pd.DataFrame(index=feature_df.index)

    for col in feature_df.columns:
        s = feature_df[col]
        if pd.api.types.is_numeric_dtype(s):
            if s.isna().all():
                dropped.append(col)
                continue
            X[col] = s.fillna(s.median())
            used.append(col)
        elif pd.api.types.is_bool_dtype(s):
            X[col] = s.fillna(False).astype(int)
            used.append(col)
        else:
            nunique = s.nunique(dropna=True)
            if nunique == 0 or nunique > 50:
                dropped.append(col)  # too high-cardinality (e.g. free text, IDs) to be a useful feature
                continue
            X[col] = LabelEncoder().fit_transform(s.fillna("__missing__").astype(str))
            used.append(col)

    return X, used, dropped


def detect_problem_type(df: pd.DataFrame, target: str | None) -> str:
    if target is None:
        return "clustering"
    s = df[target].dropna()
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 15:
        return "regression"
    return "classification"


def run_regression(df: pd.DataFrame, target: str) -> dict:
    y = df[target].dropna()
    X_full, used_cols, dropped_cols = _prepare_features(df, target)
    X = X_full.loc[y.index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    }

    results = []
    best_name, best_r2, best_model = None, -np.inf, None
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        results.append({"model": name, "r2_score": round(float(r2), 4), "mae": round(float(mae), 4), "rmse": round(rmse, 4)})
        if r2 > best_r2:
            best_name, best_r2, best_model = name, r2, model

    feature_importance = []
    if hasattr(best_model, "feature_importances_"):
        imp = best_model.feature_importances_
        feature_importance = sorted(
            [{"feature": f, "importance": round(float(i), 4)} for f, i in zip(used_cols, imp)],
            key=lambda d: d["importance"], reverse=True,
        )
    elif hasattr(best_model, "coef_"):
        # Raw linear-regression coefficients are scale-dependent (e.g. a
        # column ranging 0-30 will get a much larger coefficient than one
        # ranging 0-5000 even if it matters less). Standardize by each
        # feature's train-set std so importances are comparable.
        coefs = np.atleast_1d(best_model.coef_)
        feature_stds = X_train.std().replace(0, 1)
        standardized = [abs(c) * feature_stds[f] for f, c in zip(used_cols, coefs)]
        feature_importance = sorted(
            [{"feature": f, "importance": round(float(v), 4)} for f, v in zip(used_cols, standardized)],
            key=lambda d: d["importance"], reverse=True,
        )

    quality = "excellent" if best_r2 > 0.8 else "good" if best_r2 > 0.5 else "moderate" if best_r2 > 0.2 else "weak"
    article = "an" if quality == "excellent" else "a"

    return safe_json({
        "problem_type": "regression",
        "target": target,
        "features_used": used_cols,
        "features_dropped": dropped_cols,
        "models_compared": results,
        "best_model": best_name,
        "best_r2": round(float(best_r2), 4),
        "feature_importance": feature_importance[:15],
        "summary": f"The {best_name} model explains {best_r2*100:.1f}% of the variance in {target} (R² = {best_r2:.3f}), {article} {quality} fit.",
    })


def run_classification(df: pd.DataFrame, target: str) -> dict:
    y_raw = df[target].dropna()
    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(y_raw.astype(str)), index=y_raw.index)
    class_names = list(label_encoder.classes_)

    X_full, used_cols, dropped_cols = _prepare_features(df, target)
    X = X_full.loc[y.index]

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }

    results = []
    best_name, best_acc, best_model, best_preds = None, -1, None, None
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        avg = "binary" if len(class_names) == 2 else "weighted"
        f1 = f1_score(y_test, preds, average=avg, zero_division=0)
        precision = precision_score(y_test, preds, average=avg, zero_division=0)
        recall = recall_score(y_test, preds, average=avg, zero_division=0)
        results.append({
            "model": name, "accuracy": round(float(acc), 4), "f1_score": round(float(f1), 4),
            "precision": round(float(precision), 4), "recall": round(float(recall), 4),
        })
        if acc > best_acc:
            best_name, best_acc, best_model, best_preds = name, acc, model, preds

    cm = confusion_matrix(y_test, best_preds).tolist()

    feature_importance = []
    if hasattr(best_model, "feature_importances_"):
        imp = best_model.feature_importances_
        feature_importance = sorted(
            [{"feature": f, "importance": round(float(i), 4)} for f, i in zip(used_cols, imp)],
            key=lambda d: d["importance"], reverse=True,
        )

    quality = "excellent" if best_acc > 0.9 else "good" if best_acc > 0.75 else "moderate" if best_acc > 0.6 else "weak"

    return safe_json({
        "problem_type": "classification",
        "target": target,
        "class_names": class_names,
        "features_used": used_cols,
        "features_dropped": dropped_cols,
        "models_compared": results,
        "best_model": best_name,
        "best_accuracy": round(float(best_acc), 4),
        "confusion_matrix": cm,
        "feature_importance": feature_importance[:15],
        "summary": f"The {best_name} model correctly classifies {best_acc*100:.1f}% of cases, a {quality} result.",
    })


def run_clustering(df: pd.DataFrame, n_clusters: int | None = None) -> dict:
    X, used_cols, dropped_cols = _prepare_features(df, target=None)
    X = X.dropna()
    if len(X) < 10:
        return safe_json({"problem_type": "clustering", "error": "Not enough complete rows to cluster (need at least 10)."})

    X_scaled = StandardScaler().fit_transform(X)

    if n_clusters is None:
        # simple elbow heuristic: try 2-6 clusters, pick best silhouette score
        best_k, best_score = 2, -1
        scores = []
        for k in range(2, min(7, len(X))):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            try:
                score = silhouette_score(X_scaled, labels)
            except ValueError:
                score = -1
            scores.append({"k": k, "silhouette_score": round(float(score), 4)})
            if score > best_score:
                best_k, best_score = k, score
        n_clusters = best_k
    else:
        scores = []

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels) if n_clusters > 1 else 0.0

    cluster_summary = []
    X_with_labels = X.copy()
    X_with_labels["_cluster"] = labels
    for c in sorted(set(labels)):
        cluster_df = X_with_labels[X_with_labels["_cluster"] == c].drop(columns=["_cluster"])
        cluster_summary.append({
            "cluster": int(c),
            "size": int(len(cluster_df)),
            "pct_of_data": round(len(cluster_df) / len(X) * 100, 1),
            "mean_values": {col: round(float(cluster_df[col].mean()), 3) for col in used_cols[:10]},
        })

    quality = "well-separated" if sil > 0.5 else "reasonably distinct" if sil > 0.25 else "overlapping"

    return safe_json({
        "problem_type": "clustering",
        "n_clusters": n_clusters,
        "features_used": used_cols,
        "features_dropped": dropped_cols,
        "silhouette_score": round(float(sil), 4),
        "k_search": scores,
        "cluster_summary": cluster_summary,
        "summary": f"Data was grouped into {n_clusters} clusters (silhouette score {sil:.3f}) -- clusters are {quality}.",
    })


def run_auto_ml(df: pd.DataFrame, target: str | None) -> dict:
    problem_type = detect_problem_type(df, target)
    if problem_type == "regression":
        return run_regression(df, target)
    if problem_type == "classification":
        return run_classification(df, target)
    return run_clustering(df, n_clusters=None)
