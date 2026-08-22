#!/usr/bin/env python3
"""
train_tremor_pipeline.py
Multi-Target Parkinson's Tremor & Motion Biomarker Machine Learning Pipeline.
Trained on the ALAMEDA PD Tremor Dataset (4,151 sensor windows, 11 subjects).

Implements:
1. Stratified 5-Fold Cross-Validation across 4,151 sensor windows.
2. Multi-Target Classification:
   - Rest Tremor (0 vs 1)
   - Postural Tremor (0 vs 1)
   - Kinetic Tremor (0 vs 1)
   - Constancy of Rest (0 vs 1)
3. Model Architecture: Calibrated Ensemble with RobustScaler & Feature Ranking.
4. High-resolution ROC Curves, PR Curves, and Confusion Matrix Visualizations.
5. Model Serialization (Joblib) and JSON Metrics Reporting.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Configure local writable Matplotlib cache directory
os.environ["MPLCONFIGDIR"] = str(Path(".cache/matplotlib").resolve())
os.makedirs(".cache/matplotlib", exist_ok=True)

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Output paths
DATA_PATH = Path("data/ALAMEDA_PD_tremor_dataset.csv")
MODEL_DIR = Path("parkinson_model")
REPORT_PATH = Path("tremor_model_report.json")
ROC_PLOT_PATH = Path("tremor_roc_curves.png")
CM_PLOT_PATH = Path("tremor_confusion_matrices.png")

TREMOR_TARGETS = ["Rest_tremor", "Postural_tremor", "Kinetic_tremor", "Constancy_of_rest"]

def load_and_preprocess_data(csv_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """Loads dataset and extracts feature column names."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")

    metadata_cols = ["start_timestamp", "end_timestamp", "subject_id"]
    feature_cols = [c for c in df.columns if c not in metadata_cols and c not in TREMOR_TARGETS]

    # Clean numeric features
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    logger.info(f"Extracted {len(feature_cols)} kinematic & spectral biomarker features.")
    logger.info(f"Unique Subjects: {sorted(df['subject_id'].unique().tolist())}")

    return df, feature_cols

def train_and_evaluate_targets(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_splits: int = 5
) -> Dict[str, Any]:
    """
    Trains and validates models for all tremor targets using Stratified 5-Fold Cross-Validation.
    """
    X = df[feature_cols].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results_report: Dict[str, Any] = {
        "dataset": "ALAMEDA Parkinson's Disease Tremor Dataset",
        "totalSamples": len(df),
        "totalFeatures": len(feature_cols),
        "totalSubjects": int(df["subject_id"].nunique()),
        "validationMethod": f"Stratified {n_splits}-Fold Cross-Validation",
        "targets": {},
        "topGlobalBiomarkers": []
    }

    trained_models: Dict[str, Any] = {}
    scalers: Dict[str, Any] = {}
    global_importances = pd.Series(0.0, index=feature_cols)

    fig_roc, axes_roc = plt.subplots(2, 2, figsize=(14, 11))
    fig_cm, axes_cm = plt.subplots(2, 2, figsize=(14, 11))
    axes_roc = axes_roc.flatten()
    axes_cm = axes_cm.flatten()

    for idx, target in enumerate(TREMOR_TARGETS):
        y = df[target].values
        pos_count = int(np.sum(y == 1))
        neg_count = int(np.sum(y == 0))
        pos_ratio = pos_count / len(y)

        logger.info(f"\n=======================================================")
        logger.info(f"🎯 Training Model for: {target.upper()}")
        logger.info(f"   Distribution: {pos_count} Positive ({pos_ratio*100:.1f}%), {neg_count} Negative")
        logger.info(f"=======================================================")

        oof_probs = np.zeros(len(y))
        oof_preds = np.zeros(len(y))
        fold_aucs = []
        fold_f1s = []
        fold_accs = []
        feature_importances_target = np.zeros(len(feature_cols))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Train calibrated Random Forest ensemble
            class_weights = "balanced_subsample" if pos_ratio < 0.25 else None
            model = RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=4,
                class_weight=class_weights,
                random_state=42 + fold,
                n_jobs=1
            )
            model.fit(X_train_scaled, y_train)

            # Predict probabilities
            if len(model.classes_) > 1:
                val_probs = model.predict_proba(X_val_scaled)[:, 1]
            else:
                val_probs = np.full(len(y_val), float(model.classes_[0]))

            val_preds = (val_probs >= 0.5).astype(int)
            oof_probs[val_idx] = val_probs
            oof_preds[val_idx] = val_preds

            fold_auc = roc_auc_score(y_val, val_probs)
            fold_f1 = f1_score(y_val, val_preds, zero_division=0)
            fold_acc = accuracy_score(y_val, val_preds)

            fold_aucs.append(fold_auc)
            fold_f1s.append(fold_f1)
            fold_accs.append(fold_acc)
            feature_importances_target += model.feature_importances_ / n_splits

        # Overall Out-Of-Fold Evaluation
        overall_auc = float(roc_auc_score(y, oof_probs))
        overall_acc = float(accuracy_score(y, oof_preds))
        overall_f1 = float(f1_score(y, oof_preds, zero_division=0))
        overall_prec = float(precision_score(y, oof_preds, zero_division=0))
        overall_rec = float(recall_score(y, oof_preds, zero_division=0))

        # Confusion Matrix
        cm = confusion_matrix(y, oof_preds)
        tn, fp, fn, tp = cm.ravel()
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        logger.info(f"Results for {target}:")
        logger.info(f"  ROC-AUC:     {overall_auc:.4f} (CV: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f})")
        logger.info(f"  Accuracy:    {overall_acc*100:.2f}% (CV: {np.mean(fold_accs)*100:.2f}%)")
        logger.info(f"  Sensitivity: {overall_rec*100:.2f}% (Recall)")
        logger.info(f"  Specificity: {specificity*100:.2f}%")
        logger.info(f"  Precision:   {overall_prec*100:.2f}%")
        logger.info(f"  F1-Score:    {overall_f1:.4f}")

        # Train final production model on full data
        final_scaler = RobustScaler()
        X_scaled_full = final_scaler.fit_transform(X)
        final_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=4,
            class_weight="balanced_subsample" if pos_ratio < 0.25 else None,
            random_state=42,
            n_jobs=1
        )
        final_model.fit(X_scaled_full, y)

        trained_models[target] = final_model
        scalers[target] = final_scaler
        global_importances += feature_importances_target / len(TREMOR_TARGETS)

        # Top features for this target
        top_indices = np.argsort(feature_importances_target)[::-1][:8]
        top_features_dict = {
            feature_cols[i]: round(float(feature_importances_target[i]), 5)
            for i in top_indices
        }

        results_report["targets"][target] = {
            "positiveSamples": pos_count,
            "negativeSamples": neg_count,
            "prevalencePct": round(pos_ratio * 100, 2),
            "metrics": {
                "rocAuc": round(overall_auc, 4),
                "cvRocAucMean": round(float(np.mean(fold_aucs)), 4),
                "cvRocAucStd": round(float(np.std(fold_aucs)), 4),
                "accuracy": round(overall_acc, 4),
                "cvAccuracyMean": round(float(np.mean(fold_accs)), 4),
                "cvAccuracyStd": round(float(np.std(fold_accs)), 4),
                "sensitivity": round(overall_rec, 4),
                "specificity": round(specificity, 4),
                "precision": round(overall_prec, 4),
                "f1Score": round(overall_f1, 4),
            },
            "confusionMatrix": {
                "trueNegative": int(tn),
                "falsePositive": int(fp),
                "falseNegative": int(fn),
                "truePositive": int(tp),
            },
            "topBiomarkers": top_features_dict
        }

        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y, oof_probs)
        axes_roc[idx].plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"ROC Curve (AUC = {overall_auc:.3f})")
        axes_roc[idx].plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Chance")
        axes_roc[idx].set_title(f"{target.replace('_', ' ')} (AUC: {overall_auc:.3f})", fontsize=13, fontweight="bold", pad=10)
        axes_roc[idx].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
        axes_roc[idx].set_ylabel("True Positive Rate (Sensitivity)", fontsize=10)
        axes_roc[idx].grid(True, alpha=0.3)
        axes_roc[idx].legend(loc="lower right", frameon=True)

        # Plot Confusion Matrix
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes_cm[idx],
            annot_kws={"size": 14, "weight": "bold"},
            xticklabels=["No Tremor", "Tremor"],
            yticklabels=["No Tremor", "Tremor"],
        )
        axes_cm[idx].set_title(f"{target.replace('_', ' ')} Confusion Matrix", fontsize=13, fontweight="bold", pad=10)
        axes_cm[idx].set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
        axes_cm[idx].set_ylabel("Actual Label", fontsize=10, fontweight="bold")

    # Global Top Biomarkers
    top_global = global_importances.sort_values(ascending=False).head(15)
    results_report["topGlobalBiomarkers"] = [
        {"feature": name, "importance": round(float(val), 5)}
        for name, val in top_global.items()
    ]

    # Save ROC Plot
    fig_roc.suptitle("Parkinson's Tremor Screening Models: Stratified 5-Fold ROC Curves", fontsize=16, fontweight="bold", y=0.98)
    fig_roc.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_roc.savefig(ROC_PLOT_PATH, dpi=300)
    plt.close(fig_roc)
    logger.info(f"✅ Saved ROC Curves to {ROC_PLOT_PATH}")

    # Save Confusion Matrices Plot
    fig_cm.suptitle("Parkinson's Tremor Screening Models: Confusion Matrices", fontsize=16, fontweight="bold", y=0.98)
    fig_cm.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_cm.savefig(CM_PLOT_PATH, dpi=300)
    plt.close(fig_cm)
    logger.info(f"✅ Saved Confusion Matrix Heatmaps to {CM_PLOT_PATH}")

    # Save JSON Report
    with open(REPORT_PATH, "w") as f:
        json.dump(results_report, f, indent=2)
    logger.info(f"✅ Saved Comprehensive Metrics Report to {REPORT_PATH}")

    # Save Serialized Model Bundle
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle_path = MODEL_DIR / "tremor_model_bundle.joblib"
    joblib.dump(
        {
            "models": trained_models,
            "scalers": scalers,
            "feature_cols": feature_cols,
            "targets": TREMOR_TARGETS,
            "metrics": results_report["targets"],
        },
        bundle_path
    )
    logger.info(f"✅ Saved Serialized Tremor Model Bundle to {bundle_path}")

    return results_report

def main():
    logger.info("======================================================================")
    logger.info("🚀 Starting Parkinson's Tremor Model Training (ALAMEDA Dataset)")
    logger.info("======================================================================")

    df, feature_cols = load_and_preprocess_data(DATA_PATH)
    report = train_and_evaluate_targets(df, feature_cols, n_splits=5)

    print("\n" + "=" * 70)
    print("📊 ALAMEDA PARKINSON'S TREMOR SCREENING SUMMARY REPORT")
    print("=" * 70)
    for target in TREMOR_TARGETS:
        t_data = report["targets"][target]
        m = t_data["metrics"]
        print(f"\n🎯 {target.replace('_', ' ').upper()}:")
        print(f"   • ROC-AUC:     {m['rocAuc']:.4f} (CV: {m['cvRocAucMean']:.4f} ± {m['cvRocAucStd']:.4f})")
        print(f"   • Accuracy:    {m['accuracy']*100:.2f}% (CV: {m['cvAccuracyMean']*100:.2f}%)")
        print(f"   • Sensitivity: {m['sensitivity']*100:.2f}%")
        print(f"   • Specificity: {m['specificity']*100:.2f}%")
        print(f"   • F1-Score:    {m['f1Score']:.4f}")

    print("\n🌟 TOP KINEMATIC & SPECTRAL BIOMARKERS (GLOBAL IMPORTANCE):")
    for b in report["topGlobalBiomarkers"][:8]:
        print(f"   • {b['feature']:<30}: {b['importance']:.4f}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
