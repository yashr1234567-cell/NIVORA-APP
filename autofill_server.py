#!/usr/bin/env python3
"""
autofill_server.py
Flask API and Premium Verification UI for ML-7 Form Autofill/Predictor.
Provides:
1. / - Classy Web UI to verify predictions (Role / Age Bracket -> Category / Tier).
2. /predict - JSON POST endpoint.
3. /health - Health check.
"""

import os
import sys
import joblib
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for all routes

# Define paths for model and encoders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")

# Load model and encoders
if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODERS_PATH):
    print("Error: Trained model.pkl or encoders.pkl not found.")
    print("Please run the training script first: python3 scripts/train_autofill.py")
    sys.exit(1)

print(f"Loading model from {MODEL_PATH}...")
model = joblib.load(MODEL_PATH)

print(f"Loading encoders from {ENCODERS_PATH}...")
encoders = joblib.load(ENCODERS_PATH)
role_encoder = encoders["role"]
age_bracket_encoder = encoders["age_bracket"]
category_encoder = encoders["category"]

# Get valid options for the UI
VALID_ROLES = list(role_encoder.classes_)
VALID_AGE_BRACKETS = list(age_bracket_encoder.classes_)

# Fixed mapping from Category to Tier
TIER_MAPPING = {
    "Basic": "Tier 1",
    "Standard": "Tier 2",
    "Premium": "Tier 3"
}

# Premium HTML Template for model verification UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Nivora AI — Predictive Profile Suite</title>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #070913;
      --card-bg: rgba(15, 18, 36, 0.45);
      --border-color: rgba(255, 255, 255, 0.08);
      --primary: #38bdf8;
      --primary-glow: rgba(56, 189, 248, 0.15);
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --accent: #a855f7;
      --success: #34d399;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', sans-serif;
      background-color: var(--bg);
      background-image: 
        radial-gradient(at 0% 0%, rgba(124, 58, 237, 0.12) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(56, 189, 248, 0.1) 0px, transparent 50%);
      color: var(--text);
      line-height: 1.6;
      padding: 40px 20px;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
      overflow-x: hidden;
    }
    .card {
      background: var(--card-bg);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border: 1px solid var(--border-color);
      border-radius: 24px;
      padding: 40px;
      width: 100%;
      max-width: 580px;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
      position: relative;
    }
    .card::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 1px;
      background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.15), transparent);
    }
    .brand-header {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 32px;
    }
    .logo-glow {
      width: 12px;
      height: 12px;
      background: var(--primary);
      border-radius: 50%;
      box-shadow: 0 0 12px var(--primary);
      animation: pulse 2.5s infinite;
    }
    .brand-title {
      font-family: 'Outfit', sans-serif;
      font-size: 14px;
      font-weight: 700;
      letter-spacing: 0.15em;
      text-transform: uppercase;
      color: var(--primary);
    }
    h1 {
      font-family: 'Outfit', sans-serif;
      font-size: 32px;
      font-weight: 800;
      letter-spacing: -0.02em;
      color: #ffffff;
      margin-bottom: 8px;
    }
    .subtitle {
      font-size: 15px;
      font-weight: 400;
      color: var(--text-muted);
      margin-bottom: 32px;
    }
    .form-group {
      margin-bottom: 24px;
    }
    label {
      display: block;
      font-size: 13px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 10px;
      color: var(--text-muted);
    }
    select {
      width: 100%;
      padding: 14px 18px;
      background: rgba(17, 24, 39, 0.6);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      color: #ffffff;
      font-family: inherit;
      font-size: 15px;
      font-weight: 500;
      outline: none;
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      cursor: pointer;
      appearance: none;
      -webkit-appearance: none;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%239ca3af'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M19 9l-7 7-7-7'%3E%3C/path%3E%3C/svg%3E");
      background-repeat: no-repeat;
      background-position: right 18px center;
      background-size: 16px;
    }
    select:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 4px var(--primary-glow);
      background-color: rgba(17, 24, 39, 0.85);
    }
    .results-card {
      margin-top: 36px;
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid rgba(255, 255, 255, 0.05);
      border-radius: 16px;
      padding: 24px;
      opacity: 0;
      transform: translateY(10px);
      transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
      pointer-events: none;
    }
    .results-card.visible {
      opacity: 1;
      transform: translateY(0);
      pointer-events: auto;
    }
    .results-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
      padding-bottom: 12px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .results-title {
      font-family: 'Outfit', sans-serif;
      font-size: 15px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: #fff;
    }
    .results-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-bottom: 20px;
    }
    .result-item {
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid rgba(255, 255, 255, 0.03);
      padding: 16px;
      border-radius: 12px;
    }
    .result-label {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 6px;
    }
    .result-val {
      font-family: 'Outfit', sans-serif;
      font-size: 18px;
      font-weight: 600;
      color: #ffffff;
    }
    .confidence-section {
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid rgba(255, 255, 255, 0.03);
      padding: 16px;
      border-radius: 12px;
    }
    .confidence-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
    }
    .confidence-bar-bg {
      height: 6px;
      background: rgba(255, 255, 255, 0.05);
      border-radius: 10px;
      overflow: hidden;
    }
    .confidence-bar-fill {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--primary), var(--accent));
      border-radius: 10px;
      transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .conf-value {
      font-family: 'Outfit', sans-serif;
      font-weight: 700;
      font-size: 15px;
      color: var(--primary);
    }
    @keyframes pulse {
      0%, 100% { opacity: 0.8; transform: scale(1); }
      50% { opacity: 1; transform: scale(1.15); box-shadow: 0 0 16px var(--primary); }
    }
  </style>
</head>
<body>
  <div class="card">
    <div class="brand-header">
      <div class="logo-glow"></div>
      <div class="brand-title">NIVORA MEDICAL SUITE</div>
    </div>
    
    <h1>Smart Autofill</h1>
    <div class="subtitle">Select a role or age bracket to compute the default category profile.</div>
    
    <div class="form-group">
      <label for="role">User Role</label>
      <select id="role">
        <option value="">Select Role</option>
        {% for r in roles %}
        <option value="{{ r }}">{{ r }}</option>
        {% endfor %}
      </select>
    </div>
    
    <div class="form-group">
      <label for="ageBracket">Age Bracket</label>
      <select id="ageBracket">
        <option value="">Select Age Bracket</option>
        {% for b in age_brackets %}
        <option value="{{ b }}">{{ b }}</option>
        {% endfor %}
      </select>
    </div>
    
    <div id="resultsCard" class="results-card">
      <div class="results-header">
        <div class="results-title">Predicted Profile</div>
      </div>
      
      <div class="results-grid">
        <div class="result-item">
          <div class="result-label">Category</div>
          <div class="result-val" id="resCategory">-</div>
        </div>
        <div class="result-item">
          <div class="result-label">Tier Level</div>
          <div class="result-val" id="resTier">-</div>
        </div>
      </div>
      
      <div class="confidence-section">
        <div class="confidence-header">
          <div class="result-label" style="margin-bottom: 0;">Predictive Confidence</div>
          <div class="conf-value" id="resConfidence">0%</div>
        </div>
        <div class="confidence-bar-bg">
          <div id="confidenceBar" class="confidence-bar-fill"></div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const roleSelect = document.getElementById('role');
    const ageBracketSelect = document.getElementById('ageBracket');
    const resultsCard = document.getElementById('resultsCard');
    
    const resCategory = document.getElementById('resCategory');
    const resTier = document.getElementById('resTier');
    const resConfidence = document.getElementById('resConfidence');
    const confidenceBar = document.getElementById('confidenceBar');
    
    async function updatePredictions() {
      const roleVal = roleSelect.value;
      const ageBracketVal = ageBracketSelect.value;
      
      if (!roleVal && !ageBracketVal) {
        resultsCard.classList.remove('visible');
        return;
      }
      
      try {
        const response = await fetch('/predict', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            role: roleVal || null,
            age_bracket: ageBracketVal || null
          })
        });
        
        const data = await response.json();
        
        if (response.ok) {
          resCategory.textContent = data.category;
          resTier.textContent = data.tier;
          
          const percent = (data.confidence * 100).toFixed(0);
          resConfidence.textContent = percent + '%';
          confidenceBar.style.width = percent + '%';
          
          resultsCard.classList.add('visible');
        } else {
          console.error(data.error);
        }
      } catch (err) {
        console.error("API communications interrupted", err);
      }
    }
    
    roleSelect.addEventListener('change', updatePredictions);
    ageBracketSelect.addEventListener('change', updatePredictions);
  </script>
</body>
</html>
"""

def predict_category(role, age_bracket):
    """
    Encodes the role and age_bracket inputs (handles missing values with -1)
    and predicts the category, derived tier, and confidence score.
    """
    if role:
        if role not in role_encoder.classes_:
            raise ValueError(f"Role '{role}' is not recognized.")
        role_encoded = role_encoder.transform([role])[0]
    else:
        role_encoded = -1
        
    if age_bracket:
        if age_bracket not in age_bracket_encoder.classes_:
            raise ValueError(f"Age bracket '{age_bracket}' is not recognized.")
        age_bracket_encoded = age_bracket_encoder.transform([age_bracket])[0]
    else:
        age_bracket_encoded = -1

    if role_encoded == -1 and age_bracket_encoded == -1:
        raise ValueError("Either 'role' or 'age_bracket' must be provided.")

    features = np.array([[role_encoded, age_bracket_encoded]])
    category_encoded = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    confidence = float(probabilities[category_encoded])
    category_decoded = category_encoder.inverse_transform([category_encoded])[0]
    derived_tier = TIER_MAPPING.get(category_decoded, "Unknown")
    
    return {
        "category": category_decoded,
        "tier": derived_tier,
        "confidence": round(confidence, 4)
    }

@app.route("/", methods=["GET"])
def index():
    return render_template_string(
        HTML_TEMPLATE, 
        roles=VALID_ROLES, 
        age_brackets=VALID_AGE_BRACKETS
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        role = data.get("role")
        age_bracket = data.get("age_bracket")
        
        if not role and not age_bracket:
            return jsonify({"error": "Either 'role' or 'age_bracket' must be provided"}), 400
            
        prediction_result = predict_category(role, age_bracket)
        return jsonify(prediction_result)
        
    except ValueError as val_err:
        return jsonify({"error": str(val_err)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

if __name__ == "__main__":
    print("Starting Form Autofill Flask API Server...")
    app.run(host="127.0.0.1", port=5000, debug=True)
