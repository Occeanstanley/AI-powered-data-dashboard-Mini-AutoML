import os
import io
import json
import base64
import zipfile
import tempfile
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import gradio as gr

from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, KFold, cross_val_score
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay,
    r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance

# ---------- bigger plots everywhere (no gr.Plot height needed) ----------
FIGSIZE = (10, 7)

# =========================
# Utility helpers
# =========================
def infer_task_type(df: pd.DataFrame, target: str) -> str:
    y = df[target]
    if pd.api.types.is_numeric_dtype(y):
        return "classification" if y.nunique(dropna=True) <= 10 else "regression"
    return "classification"

def build_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
            ("cat", Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
        ],
        remainder="drop",
    )
    return pre, num_cols, cat_cols

def eval_classification(y_true, y_pred, proba=None):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }
    try:
        if proba is not None and len(np.unique(y_true)) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, proba[:, 1]))
    except Exception:
        pass
    return metrics

def eval_regression(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
    }

def df_info_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique()
    })

def fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def save_fig(fig, out_dir: str, name: str) -> str:
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_distributions_as_images(df: pd.DataFrame, max_cols: int = 9) -> List[Image.Image]:
    images = []
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols[:max_cols]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df[col].dropna(), bins=30)
        ax.set_title(col)
        images.append(fig_to_pil(fig))
    return images

def plot_distributions_figs(df: pd.DataFrame, max_cols: int = 9) -> List[Tuple[str, plt.Figure]]:
    out = []
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols[:max_cols]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df[col].dropna(), bins=30)
        ax.set_title(col)
        out.append((col, fig))
    return out

def plot_correlation_fig(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        return None
    corr = df[num_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(corr, aspect="auto")
    ax.set_xticks(range(len(num_cols))); ax.set_xticklabels(num_cols, rotation=90)
    ax.set_yticks(range(len(num_cols))); ax.set_yticklabels(num_cols)
    fig.colorbar(im, ax=ax)
    ax.set_title("Correlation (Pearson)")
    return fig

def read_uploaded_csv(file_obj) -> pd.DataFrame:
    if file_obj is None:
        raise ValueError("No file provided.")
    if isinstance(file_obj, str):
        return pd.read_csv(file_obj)
    try:
        return pd.read_csv(file_obj)
    except Exception:
        file_obj.seek(0)
        data = file_obj.read()
        return pd.read_csv(io.BytesIO(data), encoding_errors="ignore")

def suggest_default_target(df: pd.DataFrame) -> Optional[str]:
    for key in ["price", "target", "label", "y", "output"]:
        for c in df.columns:
            if c.lower() == key:
                return c
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        stds = df[num_cols].std(numeric_only=True)
        return stds.sort_values(ascending=False).index[0]
    return df.columns[-1] if len(df.columns) else None

# =========================
# Sample datasets
# =========================
def load_sample(sample_name: str) -> pd.DataFrame:
    if sample_name == "Iris (classification)":
        from sklearn.datasets import load_iris
        iris = load_iris(as_frame=True)
        df = iris.frame.copy()
        df.rename(columns={"target": "species"}, inplace=True)
        return df
    if sample_name == "Titanic (classification)":
        import seaborn as sns
        df = sns.load_dataset("titanic")
        df = df.dropna(subset=["survived"]).reset_index(drop=True)
        if df["survived"].dtype == bool:
            df["survived"] = df["survived"].astype(int)
        return df
    if sample_name == "California Housing (regression)":
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing(as_frame=True)
        df = data.frame.copy()
        df.rename(columns={"MedHouseVal": "price"}, inplace=True)
        return df
    raise ValueError("Unknown sample")

# =========================
# Load data callback
# =========================
def load_csv(file, sample_choice):
    if sample_choice != "Upload CSV":
        df = load_sample(sample_choice)
    else:
        if file is None:
            return "Please upload a CSV or choose a sample.", None, None, gr.update(choices=[], value=None), [], None, None, "", "", ""
        df = read_uploaded_csv(file)

    head = df.head(20)
    info_tbl = df_info_table(df)
    dist_imgs = plot_distributions_as_images(df)
    corr_fig = plot_correlation_fig(df)

    n_rows, n_cols = df.shape
    n_num = df.select_dtypes(include=[np.number]).shape[1]
    n_cat = n_cols - n_num
    miss_min = float(info_tbl["missing_pct"].min()) if len(info_tbl) else 0.0
    miss_max = float(info_tbl["missing_pct"].max()) if len(info_tbl) else 0.0
    eda_summary_md = (
        f"**Dataset summary:** {n_rows:,} rows × {n_cols} columns · "
        f"{n_num} numeric / {n_cat} categorical. "
        f"Missingness ranges from {miss_min:.1f}% to {miss_max:.1f}%."
    )
    dist_md = (
        "**Distributions:** Histograms reveal skew/outliers. "
        "Skewed features may benefit from log transforms; "
        "categoricals are one-hot encoded automatically."
    )
    corr_md = (
        "**Correlation:** Bright colors (|ρ|→1) indicate strong linear relationships. "
        "For regression, features highly correlated with the target can drive performance; "
        "for classification, correlation is less informative than mutual information."
    )

    cols = list(df.columns)
    default_target = suggest_default_target(df)
    drop_update = gr.update(choices=cols, value=default_target)

    state_json = df.to_json(orient="split")

    return (
        "File loaded!",
        head,
        info_tbl,
        drop_update,
        dist_imgs,
        corr_fig,
        state_json,
        eda_summary_md,
        dist_md,
        corr_md
    )

# =========================
# Train + explain callback
# =========================
def train_and_explain(
    state_json, target_col, auto_task, task_choice,
    test_size, random_state, dropna_target, enable_grid,
    fast_mode, row_cap, do_perm
):
    def is_identifier(series: pd.Series) -> bool:
        name_hint = str(series.name).lower()
        high_unique = series.nunique(dropna=True) / max(len(series), 1) > 0.90
        return ("id" in name_hint) or high_unique

    def safe_cv_splits_classification(y: pd.Series, requested: int = 5) -> int:
        vc = y.value_counts(dropna=False)
        min_count = int(vc.min()) if len(vc) else 0
        return max(2, min(requested, min_count))

    def safe_stratify(y: pd.Series) -> Optional[pd.Series]:
        vc = y.value_counts()
        if len(vc) < 2 or (vc.min() < 2):
            return None
        return y

    empty_return = (
        "Load data & choose a target first.",
        None, None,
        gr.update(visible=False),  # perm plot
        "", None,                  # json pretty, pipeline
        gr.update(visible=False),  # cm
        gr.update(visible=False),  # roc
        None,                      # run state
        gr.update(visible=False),  # reg scatter
        gr.update(visible=False),  # reg resid
        "", "", "", "", ""         # explanations
    )
    if state_json is None or not target_col:
        return empty_return

    try:
        random_state = int(random_state)
    except Exception:
        random_state = 42

    df = pd.read_json(state_json, orient="split")
    work_df = df.copy()
    if dropna_target:
        work_df = work_df.dropna(subset=[target_col])

    try:
        rc = int(row_cap)
    except Exception:
        rc = 0
    if rc and rc > 0 and len(work_df) > rc:
        work_df = work_df.sample(n=rc, random_state=random_state).reset_index(drop=True)

    if is_identifier(work_df[target_col]):
        return (
            f"❗ The selected target `{target_col}` looks like an identifier (contains 'id' or is >90% unique). Choose a real outcome variable.",
            None, None, gr.update(visible=False), "", None, gr.update(visible=False), gr.update(visible=False), None, gr.update(visible=False), gr.update(visible=False),
            "", "", "", "", ""
        )

    task_type = infer_task_type(work_df, target_col) if auto_task else task_choice

    if task_type == "regression" and not pd.api.types.is_numeric_dtype(work_df[target_col]):
        work_df[target_col] = pd.to_numeric(work_df[target_col], errors="coerce")
        if work_df[target_col].isna().all():
            return (
                f"❗ Could not convert `{target_col}` to numeric for regression.",
                None, None, gr.update(visible=False), "", None, gr.update(visible=False), gr.update(visible=False), None, gr.update(visible=False), gr.update(visible=False),
                "", "", "", "", ""
            )
        if dropna_target:
            work_df = work_df.dropna(subset=[target_col])

    X = work_df.drop(columns=[target_col])
    y = work_df[target_col]

    stratify = None
    if task_type == "classification":
        stratify = safe_stratify(y)
        if stratify is None:
            return (
                "❗ Classification requires at least 2 samples per class.",
                None, None, gr.update(visible=False), "", None, gr.update(visible=False), gr.update(visible=False), None, gr.update(visible=False), gr.update(visible=False),
                "", "", "", "", ""
            )

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=float(test_size), random_state=random_state, stratify=stratify
        )
    except Exception as e:
        return f"❗ train_test_split failed: {e}", None, None, gr.update(visible=False), "", None, gr.update(visible=False), gr.update(visible=False), None, gr.update(visible=False), gr.update(visible=False), "", "", "", "", ""

    pre, num_cols, cat_cols = build_preprocessor(X_train)

    # models & CV
    if task_type == "classification":
        models = {
            "Baseline (LogisticRegression)": LogisticRegression(max_iter=500),
            "RandomForestClassifier": RandomForestClassifier(random_state=random_state)
        }
        scoring_for_cv = "accuracy"
        n_requested = 3 if fast_mode else 5
        n_splits = safe_cv_splits_classification(y_train, requested=n_requested)
        cv_obj = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        rf_param_grid = {
            "model__n_estimators": [120] if fast_mode else [150, 300],
            "model__max_depth": [None, 12] if fast_mode else [None, 10, 20],
            "model__min_samples_split": [2] if fast_mode else [2, 5],
            "model__min_samples_leaf": [1] if fast_mode else [1, 2]
        }
        main_metric_key = "accuracy"
    else:
        models = {
            "Baseline (LinearRegression)": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(random_state=random_state)
        }
        scoring_for_cv = "r2"
        cv_obj = KFold(n_splits=(3 if fast_mode else 5), shuffle=True, random_state=random_state)
        rf_param_grid = {
            "model__n_estimators": [180] if fast_mode else [200, 400],
            "model__max_depth": [None, 16] if fast_mode else [None, 12, 24],
            "model__min_samples_split": [2] if fast_mode else [2, 5],
            "model__min_samples_leaf": [1] if fast_mode else [1, 2]
        }
        main_metric_key = "r2"

    leaderboard = []
    best = {"name": None, "score": -1e18, "pipe": None}

    try:
        for name, est in models.items():
            pipe = Pipeline([("prep", pre), ("model", est)])

            if enable_grid and "RandomForest" in name:
                grid = GridSearchCV(
                    estimator=pipe, param_grid=rf_param_grid,
                    scoring=scoring_for_cv, cv=cv_obj, n_jobs=-1
                )
                grid.fit(X_train, y_train)
                final_pipe = grid.best_estimator_
                cv_score = float(grid.best_score_)
                best_params = grid.best_params_
            else:
                final_pipe = pipe.fit(X_train, y_train)
                cv_score = float(cross_val_score(
                    final_pipe, X_train, y_train, cv=cv_obj,
                    scoring=scoring_for_cv, n_jobs=-1
                ).mean())
                best_params = None

            if task_type == "classification":
                y_pred = final_pipe.predict(X_test)
                proba = None
                try:
                    proba = final_pipe.predict_proba(X_test)
                except Exception:
                    pass
                holdout = eval_classification(y_test, y_pred, proba)
                primary = holdout.get(main_metric_key, None)
            else:
                y_pred = final_pipe.predict(X_test)
                holdout = eval_regression(y_test, y_pred)
                primary = holdout.get(main_metric_key, None)

            leaderboard.append({
                "model": name,
                "cv_score": cv_score,
                "holdout": holdout,
                "best_params": best_params
            })

            if primary is not None and primary > best["score"]:
                best = {"name": name, "score": float(primary), "pipe": final_pipe}
    except Exception as e:
        return f"❗ Training failed: {e}", None, None, gr.update(visible=False), "", None, gr.update(visible=False), gr.update(visible=False), None, gr.update(visible=False), gr.update(visible=False), "", "", "", "", ""

    # Leaderboard & bar
    lb_rows = [
        {"model": row["model"], "cv_score": row["cv_score"], main_metric_key: row["holdout"].get(main_metric_key, np.nan)}
        for row in leaderboard
    ]
    lb_df = pd.DataFrame(lb_rows).set_index("model").sort_values(main_metric_key, ascending=False)

    fig_bar, ax = plt.subplots(figsize=FIGSIZE)
    lb_df[main_metric_key].plot(kind="bar", ax=ax)
    ax.set_title(f"Model Comparison – {main_metric_key.upper()} (holdout)")

    bar_expl = (
        f"**Model Comparison:** Bars show the holdout **{main_metric_key.upper()}** for each model. "
        f"Higher is better. Use this to pick the champion model."
    )

    # Permutation importance (optional)
    perm_fig = None
    perm_expl = ""
    if do_perm and best["pipe"] is not None:
        try:
            result = permutation_importance(
                best["pipe"], X_test, y_test,
                n_repeats=(3 if fast_mode else 5),
                random_state=random_state, n_jobs=-1
            )
            importances = pd.Series(result.importances_mean,
                                    index=[f"f{i}" for i in range(len(result.importances_mean))])
            top = importances.sort_values(ascending=False)[:15]
            perm_fig, ax2 = plt.subplots(figsize=FIGSIZE)
            top[::-1].plot(kind="barh", ax=ax2)
            ax2.set_title(f"Permutation Importance – {best['name']}")
            perm_expl = (
                "**Permutation Importance:** Features at the top reduce performance most when shuffled → "
                "they’re most important to the model. Names are generic because of one-hot encoding."
            )
        except Exception:
            pass

    # Classification-only visuals
    cm_fig = None
    roc_fig = None
    cm_expl = ""
    roc_expl = ""
    if task_type == "classification" and best["pipe"] is not None:
        try:
            y_pred_best = best["pipe"].predict(X_test)
            cm = confusion_matrix(y_test, y_pred_best)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            cm_fig, ax_cm = plt.subplots(figsize=FIGSIZE)
            disp.plot(ax=ax_cm, colorbar=False)
            ax_cm.set_title("Confusion Matrix (best model)")
            cm_expl = (
                "**Confusion Matrix:** Rows are true classes, columns are predictions. "
                "Diagonal cells are correct predictions; off-diagonals are errors."
            )
        except Exception:
            pass
        try:
            if len(np.unique(y_test)) == 2 and hasattr(best["pipe"], "predict_proba"):
                proba_best = best["pipe"].predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, proba_best)
                roc_auc = auc(fpr, tpr)
                roc_fig, ax_roc = plt.subplots(figsize=FIGSIZE)
                ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
                ax_roc.plot([0, 1], [0, 1], linestyle="--")
                ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
                ax_roc.set_title("ROC Curve (best model)")
                ax_roc.legend(loc="lower right")
                roc_expl = (
                    "**ROC Curve:** Trade-off between true-positive and false-positive rates. "
                    "AUC closer to 1.0 is better."
                )
        except Exception:
            pass

    # Regression-only visuals
    reg_scatter_fig = None
    reg_resid_fig = None
    reg_scatter_expl = ""
    reg_resid_expl = ""
    if task_type == "regression" and best["pipe"] is not None:
        try:
            y_pred_best = best["pipe"].predict(X_test)
            reg_scatter_fig, ax_s = plt.subplots(figsize=FIGSIZE)
            ax_s.scatter(y_test, y_pred_best, s=10)
            ax_s.set_xlabel("Actual"); ax_s.set_ylabel("Predicted")
            ax_s.set_title("Predicted vs Actual (best model)")
            reg_scatter_expl = (
                "**Predicted vs Actual:** Points near the diagonal indicate accurate predictions. "
                "Systematic curves hint at non-linear patterns the model may miss."
            )

            resid = y_test - y_pred_best
            reg_resid_fig, ax_r = plt.subplots(figsize=FIGSIZE)
            ax_r.hist(resid, bins=30)
            ax_r.set_title("Residuals (best model)")
            reg_resid_expl = (
                "**Residuals:** Distribution of prediction errors (Actual − Predicted). "
                "Centered, narrow histograms indicate low, unbiased error."
            )
        except Exception:
            pass

    # Save best pipeline
    pipe_path = None
    if best["pipe"] is not None:
        tmpdir = tempfile.mkdtemp(prefix="pipeline_")
        pipe_path = os.path.join(tmpdir, f"best_pipeline_{best['name'].replace(' ', '_')}.joblib")
        joblib.dump(best["pipe"], pipe_path)

    # Visibility updates
    perm_out = gr.update(visible=False, value=None)
    if do_perm and perm_fig is not None:
        perm_out = gr.update(visible=True, value=perm_fig)

    cm_out = gr.update(visible=False, value=None)
    roc_out = gr.update(visible=False, value=None)
    if task_type == "classification":
        if cm_fig is not None:
            cm_out = gr.update(visible=True, value=cm_fig)
        if roc_fig is not None:
            roc_out = gr.update(visible=True, value=roc_fig)

    reg_scatter_out = gr.update(visible=False, value=None)
    reg_resid_out = gr.update(visible=False, value=None)
    if task_type == "regression":
        if reg_scatter_fig is not None:
            reg_scatter_out = gr.update(visible=True, value=reg_scatter_fig)
        if reg_resid_fig is not None:
            reg_resid_out = gr.update(visible=True, value=reg_resid_fig)

    run_payload = {
        "target": target_col,
        "task_type": task_type,
        "main_metric_key": main_metric_key,
        "leaderboard": leaderboard,
        "best_model_name": best["name"],
        "best_score": best["score"],
    }
    run_state_json = json.dumps(run_payload)

    pretty_lb = json.dumps(leaderboard, indent=2, default=str)
    msg = (
        f"Task: {('auto' if auto_task else task_choice)} → resolved as **{task_type}**. "
        f"Best model: **{best['name']}** with {main_metric_key}={best['score']:.4f}"
    )
    return (
        msg, lb_df, fig_bar,
        perm_out,
        pretty_lb, pipe_path,
        cm_out, roc_out,
        run_state_json,
        reg_scatter_out, reg_resid_out,
        # explanations
        bar_expl, (perm_expl if do_perm else ""),
        "", "",  # cm/roc expl will show only when visible
        reg_scatter_expl, reg_resid_expl
    )

# =========================
# Export (MD + single-file HTML + PDF + images.zip)
# =========================
def export_report(state_json: str, run_state_json: str):
    """
    Returns: status_text, md_file, html_file, pdf_file, zip_images
    - HTML is a single file with base64-embedded images.
    - Markdown references images; images.zip is provided.
    """
    if state_json is None or run_state_json is None:
        return "Load data and run training first.", None, None, None, None

    df = pd.read_json(state_json, orient="split")
    run = json.loads(run_state_json)

    mm = run["main_metric_key"]
    lb_rows = [{"model": row["model"], "cv_score": row["cv_score"], mm: row["holdout"].get(mm, np.nan)}
               for row in run["leaderboard"]]
    lb_df = pd.DataFrame(lb_rows).set_index("model").sort_values(mm, ascending=False)

    outdir = tempfile.mkdtemp(prefix="report_")
    imgs_dir = os.path.join(outdir, "images")
    os.makedirs(imgs_dir, exist_ok=True)

    # EDA figs
    dist_figs = plot_distributions_figs(df)
    dist_paths = [save_fig(fig, imgs_dir, f"dist_{col}") for col, fig in dist_figs]
    corr_fig = plot_correlation_fig(df)
    corr_path = save_fig(corr_fig, imgs_dir, "correlation") if corr_fig is not None else None

    # Leaderboard bar
    fig_bar, ax = plt.subplots(figsize=FIGSIZE)
    lb_df[mm].plot(kind="bar", ax=ax)
    ax.set_title(f"Model Comparison – {mm.upper()} (holdout)")
    bar_path = save_fig(fig_bar, imgs_dir, "model_comparison")

    # Markdown (references image files)
    n_rows, n_cols = df.shape
    md_lines = [
        f"# Mini AutoML Report",
        "",
        f"**Rows**: {n_rows:,}  |  **Columns**: {n_cols}",
        f"**Target**: `{run['target']}`  |  **Task**: **{run['task_type']}**",
        f"**Best model**: **{run['best_model_name']}**  |  **{mm}**: `{run['best_score']:.4f}`",
        "",
        "## EDA",
        "### Distributions (top numeric columns)",
    ]
    for p in dist_paths[:9]:
        md_lines.append(f"![dist]({os.path.basename(p)})")
    if corr_path:
        md_lines += ["", "### Correlation", f"![corr]({os.path.basename(corr_path)})"]
    md_lines += [
        "",
        "## Results",
        f"![comparison]({os.path.basename(bar_path)})",
        "",
        "### Leaderboard (JSON)",
        "```json",
        json.dumps(run["leaderboard"], indent=2, default=str),
        "```",
        "",
        "_Generated by AI-Powered Data Dashboard (Gradio)._",
    ]
    md_text = "\n".join(md_lines)
    md_path = os.path.join(outdir, "report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    # Self-contained HTML (inline base64)
    html_ok = True
    html_path = os.path.join(outdir, "report.html")
    try:
        def img_tag(p):
            with open(p, "rb") as fp:
                b64 = base64.b64encode(fp.read()).decode("ascii")
            return f"<img src='data:image/png;base64,{b64}' style='max-width:100%; height:auto; margin:8px 0;'/>"

        html_parts = [
            "<html><head><meta charset='utf-8'><title>Mini AutoML Report</title>",
            "<style>body{font-family:system-ui,Segoe UI,Arial,Helvetica,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;}</style>",
            "</head><body>",
            "<h1>Mini AutoML Report</h1>",
            f"<p><b>Rows:</b> {n_rows:,} &nbsp; <b>Columns:</b> {n_cols}</p>",
            f"<p><b>Target:</b> {run['target']} &nbsp; <b>Task:</b> {run['task_type']}</p>",
            f"<p><b>Best model:</b> {run['best_model_name']} &nbsp; <b>{mm}:</b> {run['best_score']:.4f}</p>",
            "<h2>EDA</h2><h3>Distributions (top numeric columns)</h3>"
        ]
        for p in dist_paths[:9]:
            html_parts.append(img_tag(p))
        if corr_path:
            html_parts.append("<h3>Correlation</h3>")
            html_parts.append(img_tag(corr_path))
        html_parts += [
            "<h2>Results</h2>",
            img_tag(bar_path),
            "<h3>Leaderboard (JSON)</h3>",
            "<pre style='white-space:pre-wrap;background:#f6f8fa;padding:12px;border-radius:8px;'>",
            json.dumps(run["leaderboard"], indent=2, default=str),
            "</pre>",
            "<p style='opacity:.7'>Generated by AI-Powered Data Dashboard (Gradio).</p>",
            "</body></html>"
        ]
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("".join(html_parts))
    except Exception:
        html_ok = False
    if not html_ok:
        html_path = None

    # PDF (helvetica font for reliability)
    pdf_ok = True
    pdf_path = os.path.join(outdir, "report.pdf")
    try:
        from fpdf import FPDF
        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.set_auto_page_break(auto=True, margin=12)
        pdf.add_page()
        pdf.set_font("helvetica", "B", 16); pdf.cell(0, 10, "Mini AutoML Report", ln=True)
        pdf.set_font("helvetica", "", 12)
        pdf.multi_cell(0, 8, f"Target: {run['target']} | Task: {run['task_type']}")
        pdf.multi_cell(0, 8, f"Best model: {run['best_model_name']} | {mm}: {run['best_score']:.4f}")
        pdf.ln(2); pdf.set_font("helvetica", "B", 14); pdf.cell(0, 8, "EDA – Distributions", ln=True)
        pdf.set_font("helvetica", "", 11)
        for i, p in enumerate(dist_paths[:6]):
            pdf.image(p, w=90)
            if i % 2 == 1:
                pdf.ln(60)
        if corr_path:
            pdf.add_page()
            pdf.set_font("helvetica", "B", 14); pdf.cell(0, 8, "Correlation", ln=True)
            pdf.image(corr_path, w=180)
        pdf.add_page(); pdf.set_font("helvetica", "B", 14); pdf.cell(0, 8, "Results", ln=True)
        pdf.image(bar_path, w=180)
        pdf.output(pdf_path)
    except Exception:
        pdf_ok = False
        pdf_path = None

    # Zip images
    zip_path = os.path.join(outdir, "images.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in [*dist_paths, *( [corr_path] if corr_path else [] ), bar_path]:
            if p:
                zf.write(p, arcname=os.path.basename(p))

    parts = [":white_check_mark: **MD**",
             (":white_check_mark: **HTML**" if html_ok else ":x: **HTML**"),
             (":white_check_mark: **PDF**" if pdf_ok else ":x: **PDF** (install `fpdf`)")]
    status = "Export status → " + " • ".join(parts)
    return status, os.path.abspath(md_path), (os.path.abspath(html_path) if html_path else None), (os.path.abspath(pdf_path) if pdf_path else None), os.path.abspath(zip_path)

# =========================
# Gradio UI
# =========================
with gr.Blocks(title="AI Data Dashboard – Mini AutoML") as demo:
    gr.Markdown("## 🤖 AI-Powered Data Dashboard (Mini AutoML)")
    gr.Markdown("Upload a CSV **or** select a sample dataset. Then pick a target and train.")

    state_json = gr.State(value=None)
    run_state = gr.State(value=None)

    with gr.Row():
        sample_choice = gr.Dropdown(
            choices=["Upload CSV", "Iris (classification)", "Titanic (classification)", "California Housing (regression)"],
            value="Upload CSV", label="Data source"
        )
        file_in = gr.File(label="Upload CSV", file_types=[".csv"], file_count="single")

    load_btn = gr.Button("Load data")
    status = gr.Markdown()

    with gr.Row():
        df_head = gr.Dataframe(label="Preview (first 20 rows)")
        info_tbl = gr.Dataframe(label="Schema / Missingness")

    eda_summary_md = gr.Markdown()
    dist_gallery = gr.Gallery(label="Numeric Distributions (up to 10)", columns=3, allow_preview=True)
    dist_md = gr.Markdown()
    corr_plot = gr.Plot(label="Correlation (Pearson)")
    corr_md = gr.Markdown()

    gr.Markdown("---")
    cols_dropdown = gr.Dropdown(choices=[], value=None, label="Target column", interactive=True)
    auto_task = gr.Checkbox(label="Auto-detect task", value=True)
    task_choice = gr.Radio(choices=["classification", "regression"], value="classification", label="If auto is off:")

    with gr.Row():
        test_size = gr.Slider(minimum=0.1, maximum=0.4, step=0.05, value=0.2, label="Test size")
        random_state = gr.Number(value=42, label="Random state", precision=0)
        dropna_target = gr.Checkbox(label="Drop rows with missing target", value=True)
        enable_grid = gr.Checkbox(label="Enable GridSearchCV (RandomForest)", value=False)

    with gr.Row():
        fast_mode = gr.Checkbox(value=True, label="Fast mode (3-fold CV + smaller grid)")
        row_cap = gr.Number(value=0, precision=0, label="Row cap (0 = no cap)")
        do_perm = gr.Checkbox(value=False, label="Compute permutation importance (slower)")

    run_btn = gr.Button("Run Training")

    msg_out = gr.Markdown()
    lb_df = gr.Dataframe(label="Results Leaderboard")
    bar_plot = gr.Plot(label="Holdout primary metric")
    bar_expl = gr.Markdown()

    perm_plot = gr.Plot(label="Permutation importance (best model)", visible=False)
    perm_expl = gr.Markdown(visible=False)

    cm_plot = gr.Plot(label="Confusion Matrix (best model)", visible=False)
    cm_expl = gr.Markdown(visible=False)
    roc_plot = gr.Plot(label="ROC Curve (binary, best model)", visible=False)
    roc_expl = gr.Markdown(visible=False)

    reg_scatter = gr.Plot(label="Predicted vs Actual (best model)", visible=False)
    reg_scatter_expl = gr.Markdown(visible=False)
    reg_resid = gr.Plot(label="Residuals (best model)", visible=False)
    reg_resid_expl = gr.Markdown(visible=False)

    with gr.Accordion("Show raw leaderboard JSON", open=False):
        lb_json = gr.Code(label="Leaderboard (JSON)", language="json")

    pipe_file = gr.File(label="Download best pipeline (.joblib)")

    gr.Markdown("---")
    export_btn = gr.Button("📤 Export Report (Markdown, HTML, PDF + images)")
    export_status = gr.Markdown()
    md_out = gr.File(label="report.md")
    html_out = gr.File(label="report.html")
    pdf_out = gr.File(label="report.pdf")
    zip_out = gr.File(label="images.zip")

    # Wiring
    load_btn.click(
        fn=load_csv,
        inputs=[file_in, sample_choice],
        outputs=[
            status, df_head, info_tbl, cols_dropdown,
            dist_gallery, corr_plot, state_json,
            eda_summary_md, dist_md, corr_md
        ],
    )

    run_btn.click(
        fn=train_and_explain,
        inputs=[state_json, cols_dropdown, auto_task, task_choice,
                test_size, random_state, dropna_target, enable_grid,
                fast_mode, row_cap, do_perm],
        outputs=[
            msg_out, lb_df, bar_plot,
            perm_plot,
            lb_json, pipe_file,
            cm_plot, roc_plot,
            run_state,
            reg_scatter, reg_resid,
            bar_expl, perm_expl, cm_expl, roc_expl, reg_scatter_expl, reg_resid_expl
        ],
    )

    export_btn.click(
        fn=export_report,
        inputs=[state_json, run_state],
        outputs=[export_status, md_out, html_out, pdf_out, zip_out],
    )

# Spaces auto-launch; local dev:
if __name__ == "__main__":
    demo.launch()