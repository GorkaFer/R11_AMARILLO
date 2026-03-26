
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit


META_COLS = ["file_id", "is_anomaly", "fault_type", "severity"]
DEFAULT_DOMAINS = ("time", "frequency", "stft")
BASE_METRIC_COLS = [
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "precision_macro",
    "recall_macro",
    "recall_negative",
    "recall_positive",
    "recall_normal",
    "recall_anomaly",
]


# -----------------------------------------------------------------------------
# Style and plotting
# -----------------------------------------------------------------------------
def mondragon_palette() -> List[str]:
    return [
        "#113f48",
        "#0d626a",
        "#10a3a7",
        "#7cc8c9",
        "#bce4e5",
        "#dceff0",
    ]


def mondragon_cmap(dark_high: bool = True) -> LinearSegmentedColormap:
    palette = mondragon_palette()
    colors = list(reversed(palette)) if dark_high else palette
    return LinearSegmentedColormap.from_list("mondragon_blue_green", colors)


def _safe_sorted_unique(values: Sequence[Any]) -> List[Any]:
    values = list(values)
    str_like = all(isinstance(v, str) for v in values)
    return sorted(values) if str_like else sorted(values, key=lambda x: str(x))


def primary_metric_for_level(level: int) -> str:
    if level == 1:
        return "recall_anomaly"
    if level in (2, 3):
        return "f1_macro"
    raise ValueError("level must be 1, 2 or 3")


def default_class_weight(level: int) -> Optional[str]:
    return "balanced_subsample" if level == 1 else None


def expected_metric_columns(level: Optional[int] = None) -> List[str]:
    cols = list(BASE_METRIC_COLS)
    if level in (2, 3):
        cols = [c for c in cols if c not in {"recall_negative", "recall_positive", "recall_normal", "recall_anomaly"}]
    return cols


def ensure_metric_columns(df: pd.DataFrame, level: Optional[int] = None) -> pd.DataFrame:
    out = df.copy()
    for col in expected_metric_columns(level):
        if col not in out.columns:
            out[col] = np.nan
    return out


def select_metric_column(summary_df: pd.DataFrame, level: int, requested: Optional[str] = None) -> str:
    candidates = []
    if requested is not None:
        candidates.append(requested)
    candidates.extend([primary_metric_for_level(level), "f1_macro", "balanced_accuracy", "accuracy"])

    seen = set()
    for col in candidates:
        if col in seen:
            continue
        seen.add(col)
        if col in summary_df.columns:
            values = pd.to_numeric(summary_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if values.notna().any():
                return col

    numeric_cols = summary_df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        values = pd.to_numeric(summary_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.notna().any():
            return col

    raise ValueError("No sortable metric column with finite values found.")


def plot_confusion_matrix_mondragon(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Optional[Sequence[Any]] = None,
    title: str = "Confusion matrix",
    normalize: bool = True,
    figsize: Tuple[int, int] = (6, 5),
    dark_high: bool = True,
) -> None:
    if labels is None:
        labels = _safe_sorted_unique(list(y_true) + list(y_pred))
    labels = list(labels)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_display = cm.astype(float)

    if normalize:
        row_sums = cm_display.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_display = cm_display / row_sums
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = 0.0, float(cm_display.max()) if cm_display.size else 1.0

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        cm_display,
        cmap=mondragon_cmap(dark_high=dark_high),
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row proportion" if normalize else "Count")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels([str(x) for x in labels], rotation=35, ha="right")
    ax.set_yticklabels([str(x) for x in labels])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    threshold = (vmax - vmin) * 0.55 + vmin if cm_display.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm_display[i, j]
            count = int(cm[i, j])
            text = f"{count}\n{100 * value:.1f}%" if normalize else f"{count}"
            use_white = value >= threshold
            txt_color = "white" if use_white else "#0a0a0a"
            bbox_face = (0, 0, 0, 0.32) if use_white else (1, 1, 1, 0.72)
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color=txt_color,
                fontsize=9,
                fontweight="semibold",
                bbox=dict(boxstyle="round,pad=0.20", facecolor=bbox_face, edgecolor="none"),
            )

    plt.tight_layout()
    plt.show()


def plot_metric_comparison(
    summary_df: pd.DataFrame,
    metric: str = "f1_macro",
    group_col: str = "domain",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (9, 4),
) -> None:
    if metric not in summary_df.columns:
        print(f"[Skip plot] Metric '{metric}' not found.")
        return
    if group_col not in summary_df.columns:
        print(f"[Skip plot] Group column '{group_col}' not found.")
        return

    plot_df = summary_df[[group_col, metric]].copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce").replace([np.inf, -np.inf], np.nan)
    plot_df = plot_df.dropna(subset=[metric]).sort_values(metric, ascending=False)

    if plot_df.empty:
        print(f"[Skip plot] No finite values available for metric '{metric}'.")
        return

    palette = mondragon_palette()
    colors = [palette[i % len(palette)] for i in range(len(plot_df))]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(plot_df[group_col].astype(str), plot_df[metric], color=colors, edgecolor="none")
    ax.set_ylabel(metric)
    ax.set_xlabel(group_col)
    ax.set_title(title or f"{metric} comparison")
    ax.tick_params(axis="x", rotation=35)

    ymin = min(0.0, float(plot_df[metric].min()) - 0.05)
    ymax = float(plot_df[metric].max()) + 0.08

    if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin == ymax:
        ymin, ymax = 0.0, 1.0

    ax.set_ylim(ymin, ymax)

    for bar, val in zip(bars, plot_df[metric]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="semibold",
        )

    plt.tight_layout()
    plt.show()


def plot_experiment_leaderboard(
    summary_df: pd.DataFrame,
    metric: str,
    label_col: str = "experiment",
    title: str = "Experiment leaderboard",
    figsize: Tuple[int, int] = (11, 4),
) -> None:
    plot_metric_comparison(summary_df, metric=metric, group_col=label_col, title=title, figsize=figsize)


def plot_top_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 10,
    title: str = "Top feature importances",
    figsize: Tuple[int, int] = (8, 5),
) -> None:
    top_df = importance_df.head(top_n).iloc[::-1].copy()
    palette = mondragon_palette()
    colors = [palette[i % len(palette)] for i in range(len(top_df))]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(top_df["feature"], top_df["importance"], color=colors, edgecolor="none")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title(title)

    max_val = float(top_df["importance"].max()) if not top_df.empty else 0.0
    offset = max_val * 0.015 if max_val > 0 else 0.01
    for bar, val in zip(bars, top_df["importance"]):
        ax.text(
            bar.get_width() + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.show()


def plot_top_features_over_time(
    feature_values_df: pd.DataFrame,
    file_id: Optional[Any] = None,
    feature_names: Optional[Sequence[str]] = None,
    highlight_source: str = "prediction",
    highlight_class: Optional[Any] = None,
    normal_label: Any = "normal",
    title_prefix: str = "",
    figsize: Tuple[int, int] = (12, 7),
) -> None:
    if "window_order" not in feature_values_df.columns:
        raise ValueError("feature_values_df must contain 'window_order'.")
    if "file_id" not in feature_values_df.columns:
        raise ValueError("feature_values_df must contain 'file_id'.")

    df_plot = feature_values_df.copy()

    if feature_names is None:
        reserved = set(META_COLS + ["file_id", "window_order", "split", "true_label", "prediction"])
        feature_names = [c for c in df_plot.columns if c not in reserved]

    feature_names = list(feature_names)[:2]
    if len(feature_names) == 0:
        raise ValueError("No feature columns available to plot.")

    if file_id is None:
        candidate_files = df_plot.loc[df_plot["split"] == "test", "file_id"]
        file_id = candidate_files.iloc[0] if len(candidate_files) > 0 else df_plot["file_id"].iloc[0]

    df_plot = df_plot[df_plot["file_id"] == file_id].copy().sort_values("window_order")
    if df_plot.empty:
        raise ValueError(f"No rows found for file_id={file_id!r}")

    if highlight_source not in df_plot.columns:
        raise ValueError(f"highlight_source '{highlight_source}' not found in dataframe.")

    source = df_plot[highlight_source]
    if highlight_class is not None:
        highlight_mask = source == highlight_class
    else:
        source_str = source.astype(str)
        unique_source = set(source_str.unique().tolist())
        if str(normal_label) in unique_source:
            highlight_mask = source_str != str(normal_label)
        elif 0 in set(pd.to_numeric(source, errors="coerce").dropna().unique().tolist()):
            source_num = pd.to_numeric(source, errors="coerce")
            highlight_mask = source_num.fillna(0) != 0
        else:
            highlight_mask = pd.Series(False, index=df_plot.index)

    palette = mondragon_palette()
    line_colors = palette[: max(2, len(feature_names))]

    fig, axes = plt.subplots(len(feature_names), 1, figsize=figsize, sharex=True)
    if len(feature_names) == 1:
        axes = [axes]

    for ax, feat, color in zip(axes, feature_names, line_colors):
        for x in df_plot.loc[highlight_mask, "window_order"].tolist():
            ax.axvspan(x - 0.45, x + 0.45, color="#cc2936", alpha=0.12, linewidth=0)
        ax.plot(df_plot["window_order"], df_plot[feat], color=color, linewidth=1.8, label=feat)
        if highlight_mask.any():
            ax.scatter(
                df_plot.loc[highlight_mask, "window_order"],
                df_plot.loc[highlight_mask, feat],
                color="#cc2936",
                s=30,
                label="Highlighted windows",
                zorder=3,
            )
        ax.set_ylabel(feat)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Window order within file")
    suptitle = f"{title_prefix} · file_id={file_id}" if title_prefix else f"file_id={file_id}"
    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Dataset helpers
# -----------------------------------------------------------------------------
def get_target_column(level: int) -> str:
    if level == 1:
        return "is_anomaly"
    if level == 2:
        return "fault_type"
    if level == 3:
        return "severity"
    raise ValueError("level must be 1, 2 or 3")


def filter_level_dataset(
    df: pd.DataFrame,
    level: int,
    normal_label: str = "normal",
    fault_type: Optional[str] = None,
) -> pd.DataFrame:
    if level == 1:
        level_df = df.copy()
    elif level == 2:
        level_df = df[(df["is_anomaly"] == 1) & (df["fault_type"] != normal_label)].copy()
    elif level == 3:
        level_df = df[
            (df["is_anomaly"] == 1)
            & (df["fault_type"] != normal_label)
            & (df["severity"].notna())
        ].copy()
        if fault_type is not None:
            level_df = level_df[level_df["fault_type"] == fault_type].copy()
    else:
        raise ValueError("level must be 1, 2 or 3")
    return level_df.reset_index(drop=True)


def summarize_level_3_targets(df: pd.DataFrame, normal_label: str = "normal") -> pd.DataFrame:
    level3_df = filter_level_dataset(df, level=3, normal_label=normal_label)
    if level3_df.empty:
        return pd.DataFrame(columns=["fault_type", "severity", "n_windows", "n_files"])

    return (
        level3_df.groupby(["fault_type", "severity"])
        .agg(n_windows=("file_id", "size"), n_files=("file_id", "nunique"))
        .reset_index()
        .sort_values(["fault_type", "severity"])
        .reset_index(drop=True)
    )


def get_valid_fault_types_for_level_3(
    df: pd.DataFrame,
    normal_label: str = "normal",
    min_classes: int = 2,
    min_files_per_class: int = 1,
) -> List[str]:
    level3_df = filter_level_dataset(df, level=3, normal_label=normal_label)
    if level3_df.empty:
        return []

    valid = []
    for fault_type, group in level3_df.groupby("fault_type"):
        per_class_files = group.groupby("severity")["file_id"].nunique()
        if len(per_class_files) >= min_classes and (per_class_files >= min_files_per_class).all():
            valid.append(fault_type)
    return sorted(valid)


def get_feature_columns(df: pd.DataFrame, meta_cols: Sequence[str] = META_COLS) -> List[str]:
    return [c for c in df.columns if c not in meta_cols]


def drop_columns_by_prefix(
    df: pd.DataFrame,
    prefixes: Sequence[str],
    meta_cols: Sequence[str] = META_COLS,
) -> pd.DataFrame:
    if not prefixes:
        return df.copy()

    keep_cols = []
    for col in df.columns:
        if col in meta_cols:
            keep_cols.append(col)
            continue
        if any(col.startswith(prefix) for prefix in prefixes):
            continue
        keep_cols.append(col)
    return df[keep_cols].copy()


def check_domain_alignment(*dfs: pd.DataFrame, meta_cols: Sequence[str] = META_COLS) -> bool:
    if len(dfs) < 2:
        return True

    ref = dfs[0][list(meta_cols)].reset_index(drop=True)
    for df in dfs[1:]:
        cur = df[list(meta_cols)].reset_index(drop=True)
        if len(cur) != len(ref):
            return False
        if not ref.equals(cur):
            return False
    return True


def prepare_dataset(
    df: pd.DataFrame,
    level: int,
    normal_label: str = "normal",
    fault_type: Optional[str] = None,
    selected_features: Optional[Sequence[str]] = None,
    drop_prefixes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    level_df = filter_level_dataset(df, level=level, normal_label=normal_label, fault_type=fault_type)

    if drop_prefixes:
        level_df = drop_columns_by_prefix(level_df, prefixes=drop_prefixes)

    target_col = get_target_column(level)
    feature_cols = get_feature_columns(level_df)

    if selected_features is not None:
        missing = sorted(set(selected_features) - set(feature_cols))
        if missing:
            raise ValueError("Some selected features are not present in the dataset: " + ", ".join(missing[:10]))
        feature_cols = list(selected_features)

    if not feature_cols:
        raise ValueError("No feature columns available after filtering and feature selection.")

    return {
        "filtered_df": level_df,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "X": level_df[feature_cols].copy(),
        "y": level_df[target_col].copy(),
        "groups": level_df["file_id"].copy(),
    }


def build_fusion_dataset(
    domain_dfs: Mapping[str, pd.DataFrame],
    selected_features_by_domain: Mapping[str, Sequence[str]],
    level: int,
    normal_label: str = "normal",
    fault_type: Optional[str] = None,
    meta_cols: Sequence[str] = META_COLS,
) -> pd.DataFrame:
    missing_domains = sorted(set(selected_features_by_domain) - set(domain_dfs))
    if missing_domains:
        raise ValueError(f"selected_features_by_domain contains unknown domains: {missing_domains}")

    filtered = {}
    for domain_name, domain_df in domain_dfs.items():
        filtered[domain_name] = filter_level_dataset(
            domain_df,
            level=level,
            normal_label=normal_label,
            fault_type=fault_type,
        )

    if not check_domain_alignment(*filtered.values(), meta_cols=meta_cols):
        raise ValueError(
            "The filtered domain datasets are not aligned row by row. "
            "Check the creation order of your datasets before fusion."
        )

    base_key = next(iter(filtered))
    blocks = [filtered[base_key][list(meta_cols)].reset_index(drop=True)]

    for domain_name, feat_list in selected_features_by_domain.items():
        df_domain = filtered[domain_name]
        missing_feats = sorted(set(feat_list) - set(df_domain.columns))
        if missing_feats:
            raise ValueError(f"Missing features in domain '{domain_name}': {missing_feats[:10]}")
        blocks.append(df_domain[list(feat_list)].reset_index(drop=True))

    return pd.concat(blocks, axis=1)


# -----------------------------------------------------------------------------
# Split helpers
# -----------------------------------------------------------------------------
def make_group_file_split(
    df_filtered: pd.DataFrame,
    target_col: str,
    group_col: str = "file_id",
    test_size: float = 0.2,
    random_state: int = 42,
    require_all_classes: bool = True,
    max_tries: int = 100,
) -> Tuple[set, set]:
    if df_filtered.empty:
        raise ValueError("Cannot split an empty dataframe.")

    all_labels = set(df_filtered[target_col].unique().tolist())

    for offset in range(max_tries):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state + offset,
        )
        train_idx, test_idx = next(splitter.split(df_filtered, df_filtered[target_col], df_filtered[group_col]))

        train_df = df_filtered.iloc[train_idx]
        test_df = df_filtered.iloc[test_idx]
        train_files = set(train_df[group_col].unique().tolist())
        test_files = set(test_df[group_col].unique().tolist())

        if not require_all_classes:
            return train_files, test_files

        train_labels = set(train_df[target_col].unique().tolist())
        test_labels = set(test_df[target_col].unique().tolist())
        if train_labels == all_labels and test_labels == all_labels:
            return train_files, test_files

    raise ValueError(
        "Could not find a valid group split where both train and test contain all classes. "
        "Try another test_size, inspect per-file label distribution, or disable require_all_classes."
    )


def apply_file_split(
    prepared: Mapping[str, Any],
    train_files: Iterable[Any],
    test_files: Iterable[Any],
    group_col: str = "file_id",
) -> Dict[str, Any]:
    train_files = set(train_files)
    test_files = set(test_files)
    df = prepared["filtered_df"]
    target_col = prepared["target_col"]
    feature_cols = prepared["feature_cols"]

    train_df = df[df[group_col].isin(train_files)].copy()
    test_df = df[df[group_col].isin(test_files)].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train or test dataframe is empty after applying file split.")

    return {
        "train_df": train_df,
        "test_df": test_df,
        "X_train": train_df[feature_cols].copy(),
        "X_test": test_df[feature_cols].copy(),
        "y_train": train_df[target_col].copy(),
        "y_test": test_df[target_col].copy(),
    }


# -----------------------------------------------------------------------------
# Model, metrics and importance
# -----------------------------------------------------------------------------
def build_model(
    model_type: str = "rf",
    random_state: int = 42,
    class_weight: Optional[str] = None,
    **model_params: Any,
):
    defaults = {
        "n_estimators": 200,
        "random_state": random_state,
        "n_jobs": -1,
    }
    if class_weight is not None:
        defaults["class_weight"] = class_weight
    defaults.update(model_params)

    model_type = model_type.lower()
    if model_type in {"rf", "random_forest", "randomforest"}:
        return RandomForestClassifier(**defaults)
    if model_type in {"et", "extra_trees", "extratrees"}:
        return ExtraTreesClassifier(**defaults)
    raise ValueError("model_type must be one of: 'rf', 'random_forest', 'et', 'extra_trees'")


def evaluate_predictions(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Optional[Sequence[Any]] = None,
    positive_label: Any = 1,
) -> Tuple[Dict[str, float], str, np.ndarray, List[Any]]:
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)

    if labels is None:
        labels = _safe_sorted_unique(list(y_true.unique()) + list(y_pred.unique()))
    else:
        labels = list(labels)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

    if len(labels) == 2 and positive_label in labels:
        negative_candidates = [lab for lab in labels if lab != positive_label]
        negative_label = negative_candidates[0] if negative_candidates else None
        if negative_label is not None:
            metrics["recall_negative"] = recall_score(y_true, y_pred, pos_label=negative_label, zero_division=0)
        metrics["recall_positive"] = recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
        if positive_label == 1:
            metrics["recall_normal"] = metrics.get("recall_negative", np.nan)
            metrics["recall_anomaly"] = metrics["recall_positive"]

    for col in BASE_METRIC_COLS:
        metrics.setdefault(col, np.nan)

    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return metrics, report, cm, list(labels)


def get_feature_importance(
    model,
    X_train: pd.DataFrame,
    X_eval: pd.DataFrame,
    y_eval: Sequence[Any],
    method: str = "tree",
    random_state: int = 42,
    scoring: str = "f1_macro",
    n_repeats: int = 5,
) -> pd.DataFrame:
    if method == "tree":
        if not hasattr(model, "feature_importances_"):
            raise ValueError("Model does not expose tree-based feature importances.")
        values = np.asarray(model.feature_importances_, dtype=float)
    elif method == "permutation":
        perm = permutation_importance(
            model,
            X_eval,
            y_eval,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring="f1_macro" if scoring == "f1_macro" else None,
            n_jobs=-1,
        )
        values = np.asarray(perm.importances_mean, dtype=float)
    else:
        raise ValueError("method must be 'tree' or 'permutation'.")

    return (
        pd.DataFrame({"feature": list(X_train.columns), "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def select_top_features(importance_df: pd.DataFrame, k: int = 5) -> List[str]:
    if importance_df.empty:
        return []
    return importance_df.head(k)["feature"].tolist()


def run_single_experiment(
    df: pd.DataFrame,
    level: int,
    train_files: Optional[Iterable[Any]] = None,
    test_files: Optional[Iterable[Any]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    normal_label: str = "normal",
    fault_type: Optional[str] = None,
    model_type: str = "rf",
    class_weight: Optional[str] = None,
    model_params: Optional[Mapping[str, Any]] = None,
    selected_features: Optional[Sequence[str]] = None,
    drop_prefixes: Optional[Sequence[str]] = None,
    require_all_classes: Optional[bool] = None,
    importance_method: str = "tree",
    importance_scoring: str = "f1_macro",
) -> Dict[str, Any]:
    prepared = prepare_dataset(
        df,
        level=level,
        normal_label=normal_label,
        fault_type=fault_type,
        selected_features=selected_features,
        drop_prefixes=drop_prefixes,
    )

    if require_all_classes is None:
        require_all_classes = True

    if train_files is None or test_files is None:
        train_files, test_files = make_group_file_split(
            prepared["filtered_df"],
            target_col=prepared["target_col"],
            test_size=test_size,
            random_state=random_state,
            require_all_classes=require_all_classes,
        )

    split_data = apply_file_split(prepared, train_files=train_files, test_files=test_files)

    model = build_model(
        model_type=model_type,
        random_state=random_state,
        class_weight=class_weight,
        **(dict(model_params or {})),
    )
    model.fit(split_data["X_train"], split_data["y_train"])

    y_pred = model.predict(split_data["X_test"])
    metrics, report, cm, labels = evaluate_predictions(split_data["y_test"], y_pred, labels=None, positive_label=1)
    importance_df = get_feature_importance(
        model,
        X_train=split_data["X_train"],
        X_eval=split_data["X_test"],
        y_eval=split_data["y_test"],
        method=importance_method,
        random_state=random_state,
        scoring=importance_scoring,
    )

    return {
        "filtered_df": prepared["filtered_df"],
        "target_col": prepared["target_col"],
        "feature_cols": prepared["feature_cols"],
        "train_files": sorted(train_files),
        "test_files": sorted(test_files),
        "train_df": split_data["train_df"],
        "test_df": split_data["test_df"],
        "X_train": split_data["X_train"],
        "X_test": split_data["X_test"],
        "y_train": split_data["y_train"],
        "y_test": split_data["y_test"],
        "model": model,
        "model_type": model_type,
        "model_params": dict(model_params or {}),
        "random_state": random_state,
        "class_weight": class_weight,
        "metrics": metrics,
        "report": report,
        "cm": cm,
        "y_pred": y_pred,
        "labels": labels,
        "feature_importance": importance_df,
        "level": level,
        "fault_type": fault_type,
        "normal_label": normal_label,
        "drop_prefixes": list(drop_prefixes or []),
    }


def compare_domains(
    domain_dfs: Mapping[str, pd.DataFrame],
    level: int,
    reference_domain: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    normal_label: str = "normal",
    fault_type: Optional[str] = None,
    model_type: str = "rf",
    class_weight: Optional[str] = None,
    model_params: Optional[Mapping[str, Any]] = None,
    selected_features_by_domain: Optional[Mapping[str, Sequence[str]]] = None,
    drop_prefixes_by_domain: Optional[Mapping[str, Sequence[str]]] = None,
    require_all_classes: Optional[bool] = None,
    importance_method: str = "tree",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    if not domain_dfs:
        raise ValueError("domain_dfs cannot be empty.")

    if reference_domain is None:
        reference_domain = next(iter(domain_dfs))
    if reference_domain not in domain_dfs:
        raise ValueError(f"reference_domain '{reference_domain}' not present in domain_dfs.")

    ref_prepared = prepare_dataset(
        domain_dfs[reference_domain],
        level=level,
        normal_label=normal_label,
        fault_type=fault_type,
        selected_features=None,
        drop_prefixes=None,
    )

    if require_all_classes is None:
        require_all_classes = True

    train_files, test_files = make_group_file_split(
        ref_prepared["filtered_df"],
        target_col=ref_prepared["target_col"],
        test_size=test_size,
        random_state=random_state,
        require_all_classes=require_all_classes,
    )

    results_by_domain: Dict[str, Dict[str, Any]] = {}
    rows = []

    for domain_name, df_domain in domain_dfs.items():
        selected_features = None if selected_features_by_domain is None else selected_features_by_domain.get(domain_name)
        drop_prefixes = None if drop_prefixes_by_domain is None else drop_prefixes_by_domain.get(domain_name)

        result = run_single_experiment(
            df_domain,
            level=level,
            train_files=train_files,
            test_files=test_files,
            random_state=random_state,
            normal_label=normal_label,
            fault_type=fault_type,
            model_type=model_type,
            class_weight=class_weight,
            model_params=model_params,
            selected_features=selected_features,
            drop_prefixes=drop_prefixes,
            require_all_classes=require_all_classes,
            importance_method=importance_method,
        )
        results_by_domain[domain_name] = result
        row = {"domain": domain_name}
        row.update({col: result["metrics"].get(col, np.nan) for col in expected_metric_columns(level)})
        rows.append(row)

    summary_df = ensure_metric_columns(pd.DataFrame(rows), level)
    sort_metric = select_metric_column(summary_df, level)
    summary_df = summary_df.sort_values(sort_metric, ascending=False).reset_index(drop=True)
    return summary_df, results_by_domain


def run_topk_sweep(
    df: pd.DataFrame,
    base_result: Mapping[str, Any],
    k_values: Sequence[int] = (2, 5, 10),
    level: Optional[int] = None,
    model_type: Optional[str] = None,
    class_weight: Optional[str] = None,
    model_params: Optional[Mapping[str, Any]] = None,
    importance_method: str = "tree",
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]]]:
    inferred_level = level if level is not None else int(base_result["level"])
    inferred_model_type = model_type if model_type is not None else str(base_result["model_type"])

    rows = []
    results_by_k: Dict[int, Dict[str, Any]] = {}

    for k in k_values:
        selected_feats = select_top_features(base_result["feature_importance"], k=k)
        res = run_single_experiment(
            df,
            level=inferred_level,
            train_files=base_result["train_files"],
            test_files=base_result["test_files"],
            random_state=int(base_result.get("random_state", 42)),
            normal_label=base_result.get("normal_label", "normal"),
            fault_type=base_result.get("fault_type"),
            model_type=inferred_model_type,
            class_weight=class_weight,
            model_params=model_params if model_params is not None else base_result.get("model_params", {}),
            selected_features=selected_feats,
            require_all_classes=True,
            importance_method=importance_method,
        )
        results_by_k[k] = res
        row = {"k": k}
        row.update({col: res["metrics"].get(col, np.nan) for col in expected_metric_columns(inferred_level)})
        rows.append(row)

    summary_df = ensure_metric_columns(pd.DataFrame(rows), inferred_level)
    sort_metric = select_metric_column(summary_df, inferred_level)
    return summary_df.sort_values(sort_metric, ascending=False).reset_index(drop=True), results_by_k


def run_fusion_experiment(
    domain_dfs: Mapping[str, pd.DataFrame],
    selected_features_by_domain: Mapping[str, Sequence[str]],
    level: int,
    train_files: Optional[Iterable[Any]] = None,
    test_files: Optional[Iterable[Any]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    normal_label: str = "normal",
    fault_type: Optional[str] = None,
    model_type: str = "rf",
    class_weight: Optional[str] = None,
    model_params: Optional[Mapping[str, Any]] = None,
    require_all_classes: Optional[bool] = None,
    importance_method: str = "tree",
) -> Dict[str, Any]:
    fusion_df = build_fusion_dataset(
        domain_dfs=domain_dfs,
        selected_features_by_domain=selected_features_by_domain,
        level=level,
        normal_label=normal_label,
        fault_type=fault_type,
    )

    return run_single_experiment(
        fusion_df,
        level=level,
        train_files=train_files,
        test_files=test_files,
        test_size=test_size,
        random_state=random_state,
        normal_label=normal_label,
        fault_type=fault_type,
        model_type=model_type,
        class_weight=class_weight,
        model_params=model_params,
        selected_features=None,
        drop_prefixes=None,
        require_all_classes=require_all_classes,
        importance_method=importance_method,
    )


def run_no_microphone_experiment(
    df: pd.DataFrame,
    level: int,
    train_files: Iterable[Any],
    test_files: Iterable[Any],
    random_state: int = 42,
    normal_label: str = "normal",
    fault_type: Optional[str] = None,
    model_type: str = "rf",
    class_weight: Optional[str] = None,
    model_params: Optional[Mapping[str, Any]] = None,
    require_all_classes: bool = True,
    importance_method: str = "tree",
) -> Dict[str, Any]:
    return run_single_experiment(
        df,
        level=level,
        train_files=train_files,
        test_files=test_files,
        random_state=random_state,
        normal_label=normal_label,
        fault_type=fault_type,
        model_type=model_type,
        class_weight=class_weight,
        model_params=model_params,
        selected_features=None,
        drop_prefixes=["microphone_"],
        require_all_classes=require_all_classes,
        importance_method=importance_method,
    )


def result_to_row(result: Mapping[str, Any], experiment_name: str, domain: Optional[str] = None) -> Dict[str, Any]:
    row = {
        "experiment": experiment_name,
        "domain": domain,
        "level": result.get("level"),
        "fault_type": result.get("fault_type"),
        "model_type": result.get("model_type"),
        "n_features": len(result.get("feature_cols", [])),
    }
    for col in expected_metric_columns(result.get("level")):
        row[col] = result.get("metrics", {}).get(col, np.nan)
    return row


def get_best_candidate(
    candidates_df: pd.DataFrame,
    detailed: Mapping[str, Dict[str, Any]],
    level: int,
    metric: Optional[str] = None,
):
    if candidates_df.empty:
        raise ValueError("candidates_df is empty.")

    metric = select_metric_column(candidates_df, level, requested=metric)

    tmp = candidates_df.copy()
    tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce").replace([np.inf, -np.inf], np.nan)
    tmp = tmp.dropna(subset=[metric])

    if tmp.empty:
        raise ValueError(f"All values are NaN/Inf for metric '{metric}'.")

    best_row = tmp.sort_values(metric, ascending=False).iloc[0].to_dict()
    exp_name = best_row["experiment"]

    if exp_name not in detailed:
        raise KeyError(f"Experiment '{exp_name}' not found in detailed results.")

    best_pack = detailed[exp_name]
    return best_row, best_pack["result"], best_pack["spec"], best_pack


# -----------------------------------------------------------------------------
# Saving helpers
# -----------------------------------------------------------------------------
def save_model(model, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return path


def extract_top_feature_values(result: Mapping[str, Any], top_n: int = 2) -> Tuple[pd.DataFrame, List[str]]:
    importance_df = result["feature_importance"]
    selected_features = select_top_features(importance_df, k=top_n)
    if len(selected_features) == 0:
        raise ValueError("No selected features available.")

    full_df = result["filtered_df"].copy()
    feature_cols = list(result["feature_cols"])
    target_col = str(result["target_col"])

    missing = sorted(set(selected_features) - set(full_df.columns))
    if missing:
        raise ValueError(f"Selected features not found in filtered_df: {missing}")

    full_df["window_order"] = full_df.groupby("file_id").cumcount()
    full_df["split"] = np.where(full_df["file_id"].isin(result["train_files"]), "train", "test")
    X_full = full_df[feature_cols].copy()
    full_df["prediction"] = result["model"].predict(X_full)
    full_df["true_label"] = full_df[target_col]

    keep_meta = [c for c in META_COLS if c in full_df.columns]
    out_cols = ["file_id", "window_order", "split", "true_label", "prediction"] + keep_meta
    out_cols = list(dict.fromkeys(out_cols))
    out_cols += selected_features
    return full_df[out_cols].copy(), selected_features


def save_top_feature_values(result: Mapping[str, Any], out_csv: str | Path, top_n: int = 2) -> Path:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    feature_values_df, _ = extract_top_feature_values(result, top_n=top_n)
    feature_values_df.to_csv(out_csv, index=False)
    return out_csv


def save_result_bundle(
    result: Mapping[str, Any],
    output_dir: str | Path,
    experiment_name: str,
    task: str,
    domain: Optional[str] = None,
    save_metadata: bool = False,
    save_top2_values: bool = True,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = experiment_name.replace(" ", "_").replace("/", "-")

    paths: Dict[str, Path] = {
        "model": save_model(result["model"], output_dir / f"{safe_name}.joblib"),
    }

    if save_metadata:
        paths["metadata"] = save_json(
            {
                "experiment_name": experiment_name,
                "task": task,
                "domain": domain,
                "level": result.get("level"),
                "fault_type": result.get("fault_type"),
                "model_type": result.get("model_type"),
                "model_params": result.get("model_params"),
                "random_state": result.get("random_state"),
                "class_weight": result.get("class_weight"),
                "train_files": result.get("train_files"),
                "test_files": result.get("test_files"),
                "n_features": len(result.get("feature_cols", [])),
                "top2_feature_names": select_top_features(result["feature_importance"], k=2),
                "metrics": result.get("metrics"),
            },
            output_dir / f"{safe_name}_metadata.json",
        )

    if save_top2_values:
        paths["top2_values"] = save_top_feature_values(
            result=result,
            out_csv= f"Datos/Transformados/SAD/{safe_name}_top2_feature_values.csv",
            top_n=2,
        )

    return paths
