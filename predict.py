#!/usr/bin/env python3
"""
predict.py
Inference and Clinical Screening Utility for Parkinson's Disease Drawing Classification.
Supports both Spiral and Wave modalities.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Union

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

DEFAULT_SPIRAL_MODEL = Path("parkinsons_finetuned/spiral/best_model")
DEFAULT_WAVE_MODEL = Path("parkinsons_finetuned/wave/best_model")
VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

class ParkinsonPredictor:
    def __init__(self, model_path: Union[str, Path]):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_path}")

        # Load processor and model
        self.processor = AutoImageProcessor.from_pretrained(str(self.model_path), use_fast=False)
        self.model = AutoModelForImageClassification.from_pretrained(str(self.model_path))
        self.model.eval()

        self.id2label = self.model.config.id2label

    def predict_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Classifies a single drawing image.
        Returns predicted class, confidence, raw probabilities, and clinical risk interpretation.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).squeeze(0)

        pred_idx = torch.argmax(probs).item()
        pred_label = self.id2label.get(pred_idx, str(pred_idx))
        confidence = probs[pred_idx].item()

        prob_dict = {
            self.id2label.get(i, str(i)): float(probs[i].item())
            for i in range(len(probs))
        }

        # Clinical interpretation
        parkinson_prob = prob_dict.get("parkinson", 0.0)
        if parkinson_prob >= 0.70:
            risk_level = "High Probability of Parkinsonian Drawing Tremor"
        elif parkinson_prob >= 0.50:
            risk_level = "Borderline / Moderate Indicators Detected"
        else:
            risk_level = "Normal / Healthy Drawing Pattern"

        return {
            "image_path": str(image_path.resolve()),
            "predicted_label": pred_label,
            "confidence": confidence,
            "probabilities": prob_dict,
            "parkinson_probability": parkinson_prob,
            "risk_assessment": risk_level
        }

def main():
    parser = argparse.ArgumentParser(description="Predict Parkinson's from Sketch (Spiral or Wave)")
    parser.add_argument("input_path", type=str, help="Path to image file or directory of images")
    parser.add_argument("--modality", choices=["spiral", "wave"], default=None, help="Sketch modality (auto-detected if omitted)")
    args = parser.parse_args()

    input_path = Path(args.input_path)

    # Determine modality
    modality = args.modality
    if modality is None:
        if "spiral" in str(input_path).lower():
            modality = "spiral"
        elif "wave" in str(input_path).lower():
            modality = "wave"
        else:
            modality = "spiral"  # default fallback

    model_dir = DEFAULT_SPIRAL_MODEL if modality == "spiral" else DEFAULT_WAVE_MODEL
    print(f"Loading {modality.upper()} model from {model_dir}...")
    predictor = ParkinsonPredictor(model_dir)

    # Single image or batch directory
    if input_path.is_file():
        res = predictor.predict_image(input_path)
        print("\n" + "=" * 60)
        print(f"🏥 PARKINSON'S SCREENING RESULT ({modality.upper()} MODALITY)")
        print("=" * 60)
        print(f"File:           {input_path.name}")
        print(f"Prediction:     {res['predicted_label'].upper()}")
        print(f"Confidence:     {res['confidence']*100:.2f}%")
        print(f"Probabilities:  Healthy: {res['probabilities'].get('healthy', 0)*100:.2f}% | Parkinson: {res['probabilities'].get('parkinson', 0)*100:.2f}%")
        print(f"Assessment:     {res['risk_assessment']}")
        print("=" * 60 + "\n")
    elif input_path.is_dir():
        image_files = [f for f in input_path.glob("*") if f.suffix.lower() in VALID_EXTS]
        print(f"\nEvaluating {len(image_files)} images in {input_path}...\n")
        correct = 0
        total = len(image_files)

        print(f"{'Filename':<30} | {'Prediction':<12} | {'Confidence':<12} | {'Parkinson Prob':<15}")
        print("-" * 75)
        for img in sorted(image_files):
            res = predictor.predict_image(img)
            print(f"{img.name:<30} | {res['predicted_label']:<12} | {res['confidence']*100:>9.2f}% | {res['parkinson_probability']*100:>13.2f}%")

if __name__ == "__main__":
    main()
