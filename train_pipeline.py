#!/usr/bin/env python3
"""
train_pipeline.py
Full Fine-Tuning and Evaluation Pipeline for Parkinson's Disease Classification from Spiral & Wave Drawings.
Model: gianlab/swin-tiny-patch4-window7-224-finetuned-parkinson-classification
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    set_seed
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import StratifiedKFold

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL_ID = "gianlab/swin-tiny-patch4-window7-224-finetuned-parkinson-classification"
VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

class DrawingDataset(Dataset):
    """
    PyTorch Dataset for drawing images with optional data augmentation.
    """
    def __init__(self, image_paths: List[Path], labels: List[int], processor: Any, is_train: bool = False):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
        self.is_train = is_train

        # Light data augmentation for training
        self.train_augmentation = transforms.Compose([
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]

        # Load and convert to RGB
        image = Image.open(path).convert("RGB")

        if self.is_train:
            image = self.train_augmentation(image)

        # Process with AutoImageProcessor (handles resize 224x224 & normalization)
        processed = self.processor(images=image, return_tensors="pt")
        pixel_values = processed["pixel_values"].squeeze(0)  # Shape: (3, 224, 224)

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label, dtype=torch.long)
        }

def collect_image_paths_and_labels(data_dir: Path) -> Tuple[List[Path], List[int], Dict[str, int], Dict[int, str]]:
    """
    Scans a directory expecting subfolders for each class ('healthy', 'parkinson').
    Returns image paths, integer labels, label2id, and id2label.
    """
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    if not class_names:
        raise ValueError(f"No class directories found inside {data_dir}")

    # Standardize label mapping: healthy=0, parkinson=1
    if "healthy" in class_names and "parkinson" in class_names:
        class_names = ["healthy", "parkinson"]

    label2id = {name: idx for idx, name in enumerate(class_names)}
    id2label = {idx: name for idx, name in enumerate(class_names)}

    paths = []
    labels = []
    for cls_name in class_names:
        cls_dir = data_dir / cls_name
        for file_path in cls_dir.iterdir():
            if file_path.suffix.lower() in VALID_EXTS:
                paths.append(file_path)
                labels.append(label2id[cls_name])

    return paths, labels, label2id, id2label

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    # Also compute binary metrics assuming positive class is index 1 (parkinson)
    precision_bin, recall_bin, f1_bin, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0
    )
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "precision_parkinson": float(precision_bin),
        "recall_parkinson": float(recall_bin),
        "f1_parkinson": float(f1_bin),
    }

def plot_and_save_confusion_matrix(y_true, y_pred, class_names, output_path: Path, title: str):
    """
    Plots a high-quality confusion matrix heatmap and saves to disk.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"}
    )
    
    ax.set_title(title, fontsize=14, pad=15, weight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, labelpad=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved confusion matrix plot to {output_path}")

def run_main_training(
    modality: str,
    data_base_dir: Path,
    output_dir: Path,
    model_id: str,
    epochs: int = 8,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Runs the main fine-tuning pipeline on the train split and evaluates on the test split.
    """
    set_seed(seed)
    logger.info("=" * 70)
    logger.info(f"STARTING MAIN FINE-TUNING PIPELINE FOR MODALITY: {modality.upper()}")
    logger.info("=" * 70)

    train_dir = data_base_dir / modality / "training"
    test_dir = data_base_dir / modality / "testing"

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Missing train or test directory in {data_base_dir / modality}")

    # Load file lists
    train_paths, train_labels, label2id, id2label = collect_image_paths_and_labels(train_dir)
    test_paths, test_labels, _, _ = collect_image_paths_and_labels(test_dir)

    class_names = [id2label[i] for i in sorted(id2label.keys())]

    logger.info(f"Class mapping: {label2id}")
    logger.info(f"Train samples: {len(train_paths)} -> { {cls: train_labels.count(label2id[cls]) for cls in class_names} }")
    logger.info(f"Test samples:  {len(test_paths)} -> { {cls: test_labels.count(label2id[cls]) for cls in class_names} }")

    # Image processor
    processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)

    # Datasets
    train_dataset = DrawingDataset(train_paths, train_labels, processor=processor, is_train=True)
    test_dataset = DrawingDataset(test_paths, test_labels, processor=processor, is_train=False)

    # Model
    model = AutoModelForImageClassification.from_pretrained(
        model_id,
        num_labels=len(class_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    save_checkpoint_dir = output_dir / modality
    save_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training Arguments
    training_args_dict = {
        "output_dir": str(save_checkpoint_dir / "checkpoints"),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": 0.01,
        "logging_dir": str(save_checkpoint_dir / "logs"),
        "logging_steps": 2,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_f1",
        "greater_is_better": True,
        "save_total_limit": 1,
        "report_to": "none",
        "seed": seed,
        "remove_unused_columns": False,
    }

    try:
        training_args = TrainingArguments(eval_strategy="epoch", **training_args_dict)
    except TypeError:
        training_args = TrainingArguments(evaluation_strategy="epoch", **training_args_dict)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info(f"Training for {epochs} epochs (batch_size={batch_size}, lr={learning_rate})...")
    train_result = trainer.train()
    logger.info("Training completed. Evaluating best model on test split...")

    # Final Evaluation on test set
    eval_metrics = trainer.evaluate(eval_dataset=test_dataset)

    # Predictions for confusion matrix
    predictions_output = trainer.predict(test_dataset)
    y_preds = np.argmax(predictions_output.predictions, axis=1)
    y_true = np.array(test_labels)

    # Save best model to final destination
    best_model_path = save_checkpoint_dir / "best_model"
    trainer.save_model(str(best_model_path))
    processor.save_pretrained(str(best_model_path))
    logger.info(f"Saved best model checkpoint to {best_model_path}")

    # Generate Confusion Matrix PNG
    cm_path = Path(f"{modality}_confusion_matrix.png")
    plot_and_save_confusion_matrix(
        y_true=y_true,
        y_pred=y_preds,
        class_names=class_names,
        output_path=cm_path,
        title=f"Confusion Matrix: {modality.capitalize()} Dataset (Test Split)"
    )

    cm = confusion_matrix(y_true, y_preds)

    summary = {
        "modality": modality,
        "train_counts": {cls: train_labels.count(label2id[cls]) for cls in class_names},
        "test_counts": {cls: test_labels.count(label2id[cls]) for cls in class_names},
        "accuracy": eval_metrics.get("eval_accuracy"),
        "f1": eval_metrics.get("eval_f1"),
        "precision": eval_metrics.get("eval_precision"),
        "recall": eval_metrics.get("eval_recall"),
        "f1_parkinson": eval_metrics.get("eval_f1_parkinson"),
        "confusion_matrix": cm.tolist(),
        "cm_path": str(cm_path),
        "best_model_path": str(best_model_path)
    }

    return summary

def run_cross_validation(
    modality: str,
    data_base_dir: Path,
    output_dir: Path,
    model_id: str,
    n_splits: int = 5,
    epochs: int = 8,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Runs 5-fold Stratified Cross-Validation on the full modality dataset.
    """
    set_seed(seed)
    logger.info("=" * 70)
    logger.info(f"STARTING {n_splits}-FOLD STRATIFIED CROSS-VALIDATION FOR: {modality.upper()}")
    logger.info("=" * 70)

    # Combine all samples (train + test) for comprehensive CV
    train_dir = data_base_dir / modality / "training"
    test_dir = data_base_dir / modality / "testing"

    train_paths, train_labels, label2id, id2label = collect_image_paths_and_labels(train_dir)
    test_paths, test_labels, _, _ = collect_image_paths_and_labels(test_dir)

    all_paths = train_paths + test_paths
    all_labels = train_labels + test_labels

    class_names = [id2label[i] for i in sorted(id2label.keys())]
    logger.info(f"Total CV pool: {len(all_paths)} images across classes: { {cls: all_labels.count(label2id[cls]) for cls in class_names} }")

    processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_metrics = []

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(all_paths, all_labels)):
        logger.info(f"--- Running Fold {fold_idx + 1}/{n_splits} ---")

        fold_train_paths = [all_paths[i] for i in train_indices]
        fold_train_labels = [all_labels[i] for i in train_indices]
        fold_val_paths = [all_paths[i] for i in val_indices]
        fold_val_labels = [all_labels[i] for i in val_indices]

        train_dataset = DrawingDataset(fold_train_paths, fold_train_labels, processor=processor, is_train=True)
        val_dataset = DrawingDataset(fold_val_paths, fold_val_labels, processor=processor, is_train=False)

        # Fresh model instance for each fold
        model = AutoModelForImageClassification.from_pretrained(
            model_id,
            num_labels=len(class_names),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )

        fold_out_dir = output_dir / modality / f"cv_fold_{fold_idx + 1}"
        
        args_dict = {
            "output_dir": str(fold_out_dir),
            "num_train_epochs": epochs,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": 0.01,
            "save_strategy": "no",
            "report_to": "none",
            "seed": seed + fold_idx,
            "remove_unused_columns": False,
        }

        try:
            training_args = TrainingArguments(eval_strategy="no", **args_dict)
        except TypeError:
            training_args = TrainingArguments(evaluation_strategy="no", **args_dict)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        # Evaluate on validation fold
        val_pred_output = trainer.predict(val_dataset)
        y_fold_pred = np.argmax(val_pred_output.predictions, axis=1)
        y_fold_true = np.array(fold_val_labels)

        acc = accuracy_score(y_fold_true, y_fold_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_fold_true, y_fold_pred, average="weighted", zero_division=0)
        
        fold_metrics.append({
            "fold": fold_idx + 1,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        })

        logger.info(f"Fold {fold_idx + 1} Result -> Accuracy: {acc:.4f} ({acc*100:.1f}%) | F1: {f1:.4f}")

    accuracies = [m["accuracy"] for m in fold_metrics]
    f1s = [m["f1"] for m in fold_metrics]
    precisions = [m["precision"] for m in fold_metrics]
    recalls = [m["recall"] for m in fold_metrics]

    cv_summary = {
        "modality": modality,
        "n_splits": n_splits,
        "fold_metrics": fold_metrics,
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "precision_mean": float(np.mean(precisions)),
        "precision_std": float(np.std(precisions)),
        "recall_mean": float(np.mean(recalls)),
        "recall_std": float(np.std(recalls)),
    }

    logger.info(f"\n{n_splits}-Fold CV Results for {modality.upper()}:")
    logger.info(f"Accuracy:  {cv_summary['accuracy_mean']:.4f} ± {cv_summary['accuracy_std']:.4f}")
    logger.info(f"F1 Score:  {cv_summary['f1_mean']:.4f} ± {cv_summary['f1_std']:.4f}")

    return cv_summary

def print_final_summary(test_summary: Dict[str, Any], cv_summary: Dict[str, Any]):
    modality = test_summary["modality"]
    print("\n" + "=" * 70)
    print(f"📊 SUMMARY REPORT: {modality.upper()} DATASET")
    print("=" * 70)
    print(f"1. Dataset Distribution:")
    print(f"   - Training Set: {test_summary['train_counts']}")
    print(f"   - Testing Set:  {test_summary['test_counts']}")
    print(f"\n2. Held-Out Test Set Performance:")
    print(f"   - Accuracy:           {test_summary['accuracy']:.4f} ({test_summary['accuracy']*100:.1f}%)")
    print(f"   - F1 (Weighted):      {test_summary['f1']:.4f}")
    print(f"   - Precision (Weight): {test_summary['precision']:.4f}")
    print(f"   - Recall (Weighted):  {test_summary['recall']:.4f}")
    print(f"   - Parkinson F1:       {test_summary['f1_parkinson']:.4f}")
    print(f"\n3. Confusion Matrix (Test Split):")
    cm = np.array(test_summary['confusion_matrix'])
    print(f"   [Healthy Pred, Parkinson Pred]")
    print(f"   Healthy True:   {cm[0]}")
    print(f"   Parkinson True: {cm[1]}")
    print(f"   Saved Plot: {test_summary['cm_path']}")
    print(f"\n4. 5-Fold Cross-Validation Generalization:")
    print(f"   - CV Mean Accuracy:   {cv_summary['accuracy_mean']:.4f} ± {cv_summary['accuracy_std']:.4f}")
    print(f"   - CV Mean F1 Score:   {cv_summary['f1_mean']:.4f} ± {cv_summary['f1_std']:.4f}")
    print(f"   - Saved Checkpoint:   {test_summary['best_model_path']}")
    print("=" * 70 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Parkinson's Disease Drawing Classification Pipeline")
    parser.add_argument("--modality", choices=["spiral", "wave", "both"], default="both", help="Which modality to run")
    parser.add_argument("--data_dir", type=str, default="parkinsons_data", help="Path to extracted dataset")
    parser.add_argument("--output_dir", type=str, default="parkinsons_finetuned", help="Output checkpoint directory")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID, help="HF Pretrained Model ID")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    data_base_dir = Path(args.data_dir)
    output_base_dir = Path(args.output_dir)

    modalities = ["spiral", "wave"] if args.modality == "both" else [args.modality]

    all_results = {}

    for mod in modalities:
        test_res = run_main_training(
            modality=mod,
            data_base_dir=data_base_dir,
            output_dir=output_base_dir,
            model_id=args.model_id,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seed=args.seed
        )
        cv_res = run_cross_validation(
            modality=mod,
            data_base_dir=data_base_dir,
            output_dir=output_base_dir,
            model_id=args.model_id,
            n_splits=args.cv_folds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seed=args.seed
        )
        print_final_summary(test_res, cv_res)
        all_results[mod] = {"test": test_res, "cv": cv_res}

    # Comparative analysis if both modalities ran
    if len(modalities) > 1:
        print("\n" + "#" * 70)
        print("🏆 MODALITY COMPARISON: SPIRAL VS WAVE")
        print("#" * 70)
        sp_test = all_results["spiral"]["test"]
        sp_cv = all_results["spiral"]["cv"]
        wv_test = all_results["wave"]["test"]
        wv_cv = all_results["wave"]["cv"]

        print(f"{'Metric':<25} | {'Spiral':<20} | {'Wave':<20}")
        print("-" * 70)
        print(f"{'Test Accuracy':<25} | {sp_test['accuracy']:.4f} ({sp_test['accuracy']*100:.1f}%)" + " " * 9 + f"| {wv_test['accuracy']:.4f} ({wv_test['accuracy']*100:.1f}%)")
        print(f"{'Test F1 (Weighted)':<25} | {sp_test['f1']:.4f}" + " " * 14 + f"| {wv_test['f1']:.4f}")
        print(f"{'Test Parkinson F1':<25} | {sp_test['f1_parkinson']:.4f}" + " " * 14 + f"| {wv_test['f1_parkinson']:.4f}")
        print(f"{'5-Fold CV Accuracy':<25} | {sp_cv['accuracy_mean']:.4f} ± {sp_cv['accuracy_std']:.4f}" + " " * 4 + f"| {wv_cv['accuracy_mean']:.4f} ± {wv_cv['accuracy_std']:.4f}")
        print(f"{'5-Fold CV F1':<25} | {sp_cv['f1_mean']:.4f} ± {sp_cv['f1_std']:.4f}" + " " * 4 + f"| {wv_cv['f1_mean']:.4f} ± {wv_cv['f1_std']:.4f}")
        print("-" * 70)

        better_test = "Spiral" if sp_test["accuracy"] > wv_test["accuracy"] else ("Wave" if wv_test["accuracy"] > sp_test["accuracy"] else "Tie")
        better_cv = "Spiral" if sp_cv["accuracy_mean"] > wv_cv["accuracy_mean"] else ("Wave" if wv_cv["accuracy_mean"] > sp_cv["accuracy_mean"] else "Tie")

        print(f"\nConclusion:")
        print(f"- Higher Held-Out Test Accuracy: {better_test}")
        print(f"- Higher 5-Fold Cross-Validation Mean Accuracy: {better_cv}")
        print("#" * 70 + "\n")

if __name__ == "__main__":
    main()
