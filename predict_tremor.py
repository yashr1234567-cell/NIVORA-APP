#!/usr/bin/env python3
"""
predict_tremor.py
Standalone & Batch Inference CLI for Parkinson's Tremor Screening.
Uses the multi-target model bundle trained on the ALAMEDA PD dataset.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Union, List

import joblib
import numpy as np
import pandas as pd

DEFAULT_BUNDLE_PATH = Path("models/parkinsons/tremor/tremor_model_bundle.joblib")
DATA_PATH = Path("data/ALAMEDA_PD_tremor_dataset.csv")

class TremorPredictor:
    def __init__(self, bundle_path: Union[str, Path] = DEFAULT_BUNDLE_PATH):
        self.bundle_path = Path(bundle_path)
        if not self.bundle_path.exists():
            raise FileNotFoundError(f"Model bundle not found at {self.bundle_path}. Run train_tremor_pipeline.py first.")

        bundle = joblib.load(self.bundle_path)
        self.models = bundle["models"]
        self.scalers = bundle["scalers"]
        self.feature_cols = bundle["feature_cols"]
        self.targets = bundle["targets"]

    def predict_features(self, feature_dict: Dict[str, float]) -> Dict[str, Any]:
        """Classifies a single sensor feature window."""
        vec = np.array([[feature_dict.get(col, 0.0) for col in self.feature_cols]], dtype=float)

        predictions: Dict[str, Any] = {}
        probabilities: Dict[str, float] = {}

        for target in self.targets:
            model = self.models[target]
            scaler = self.scalers[target]
            vec_scaled = scaler.transform(vec)

            prob = float(model.predict_proba(vec_scaled)[0, 1])
            pred = int(prob >= 0.5)

            probabilities[target] = round(prob, 4)
            predictions[target] = {
                "detected": bool(pred == 1),
                "probability": round(prob * 100, 2),
                "status": "POSITIVE" if pred == 1 else "NEGATIVE"
            }

        # Composite Screening Tremor Burden (0 - 100)
        # Weighted by clinical severity: Rest (35%), Postural (30%), Kinetic (25%), Constancy (10%)
        weighted_score = (
            probabilities.get("Rest_tremor", 0.0) * 0.35 +
            probabilities.get("Postural_tremor", 0.0) * 0.30 +
            probabilities.get("Kinetic_tremor", 0.0) * 0.25 +
            probabilities.get("Constancy_of_rest", 0.0) * 0.10
        ) * 100.0

        risk_score = int(round(weighted_score))
        if risk_score >= 65:
            tier = "High Tremor Burden / Elevated Parkinsonian Signature"
            level = "ELEVATED"
        elif risk_score >= 35:
            tier = "Moderate Tremor Burden / Borderline Motor Incoordination"
            level = "MODERATE"
        else:
            tier = "Normal / Minimal Tremor Activity"
            level = "LOW"

        return {
            "tremorScreeningIndex": risk_score,
            "riskLevel": level,
            "clinicalInterpretation": tier,
            "targetPredictions": predictions,
            "probabilities": probabilities,
        }

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch inference on a DataFrame."""
        results = []
        for _, row in df.iterrows():
            f_dict = row.to_dict()
            res = self.predict_features(f_dict)
            flat_res = {
                "Screening_Index": res["tremorScreeningIndex"],
                "Risk_Level": res["riskLevel"],
                "Rest_Tremor_Prob": res["probabilities"].get("Rest_tremor", 0.0),
                "Postural_Tremor_Prob": res["probabilities"].get("Postural_tremor", 0.0),
                "Kinetic_Tremor_Prob": res["probabilities"].get("Kinetic_tremor", 0.0),
                "Constancy_Prob": res["probabilities"].get("Constancy_of_rest", 0.0),
            }
            results.append(flat_res)
        return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Predict Parkinson's Tremor from Accelerometer/IMU features")
    parser.add_argument("--csv", type=str, default=None, help="Path to input CSV with motion features")
    parser.add_argument("--sample", action="store_true", help="Run screening on sample windows from ALAMEDA dataset")
    parser.add_argument("--subject", type=int, default=None, help="Filter samples by subject ID (e.g. 4, 15, 16)")
    args = parser.parse_args()

    predictor = TremorPredictor()

    if args.sample or args.csv is None:
        if not DATA_PATH.exists():
            print(f"Error: Dataset not found at {DATA_PATH}")
            return

        df = pd.read_csv(DATA_PATH)
        if args.subject is not None:
            df = df[df["subject_id"] == args.subject]
            print(f"Filtered {len(df)} sample windows for Subject ID {args.subject}.")

        sample_rows = df.sample(n=min(5, len(df)), random_state=42)

        print("\n" + "=" * 70)
        print("🏥 PARKINSON'S MULTI-TARGET TREMOR SCREENING RESULTS (SAMPLE WINDOWS)")
        print("=" * 70)

        for i, (idx, row) in enumerate(sample_rows.iterrows()):
            res = predictor.predict_features(row.to_dict())
            subj = row.get("subject_id", "Unknown")
            ts = f"{row.get('start_timestamp', '')} - {row.get('end_timestamp', '')}"

            print(f"\n[Window #{i+1} | Subject {subj} | Time: {ts}]")
            print(f"   ▶ Screening Risk Index: {res['tremorScreeningIndex']}/100 [{res['riskLevel']}]")
            print(f"   ▶ Interpretation:       {res['clinicalInterpretation']}")
            print(f"   ▶ Rest Tremor:          {res['targetPredictions']['Rest_tremor']['status']} ({res['targetPredictions']['Rest_tremor']['probability']}%)")
            print(f"   ▶ Postural Tremor:      {res['targetPredictions']['Postural_tremor']['status']} ({res['targetPredictions']['Postural_tremor']['probability']}%)")
            print(f"   ▶ Kinetic Tremor:       {res['targetPredictions']['Kinetic_tremor']['status']} ({res['targetPredictions']['Kinetic_tremor']['probability']}%)")
            print(f"   ▶ Constancy of Rest:    {res['targetPredictions']['Constancy_of_rest']['status']} ({res['targetPredictions']['Constancy_of_rest']['probability']}%)")

        print("\n" + "=" * 70 + "\n")
    elif args.csv:
        input_csv = Path(args.csv)
        if not input_csv.exists():
            print(f"Error: CSV file not found at {input_csv}")
            return
        df = pd.read_csv(input_csv)
        print(f"Loaded {len(df)} rows from {input_csv}. Running batch screening...")
        out_df = predictor.predict_dataframe(df)
        out_path = input_csv.parent / f"{input_csv.stem}_tremor_predictions.csv"
        out_df.to_csv(out_path, index=False)
        print(f"✅ Saved predictions to: {out_path}")

if __name__ == "__main__":
    main()
