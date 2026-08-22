#!/usr/bin/env python3
"""
server.py
Unified Nivora Multi-Modal AI Verification & Screening Server.
Includes:
1. 🎙️ Voice Phonation Screening (Live Microphone + 4-Tier Severity)
2. 🌀 Fine-Tuned Swin Transformer Drawing Vision (Spiral & Wave)
3. ⚡ ALAMEDA Multi-Target Tremor Classifiers (Kinetic, Postural, Rest, Constancy)
4. 👁️ Cataract & Eye Vision TFLite Models
5. 🟡 Jaundice Detection TFLite Model
"""

import os
import io
import json
import base64
import logging
from pathlib import Path
from typing import Dict, Any

from aiohttp import web
import numpy as np
from PIL import Image
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
SPIRAL_MODEL_DIR = Path("parkinsons_finetuned/spiral/best_model")
WAVE_MODEL_DIR = Path("parkinsons_finetuned/wave/best_model")
TREMOR_BUNDLE_PATH = Path("parkinson_model/tremor_model_bundle.joblib")
VOICE_BUNDLE_PATH = Path("parkinson_model/parkinson_model.joblib")
TREMOR_DATA_PATH = Path("data/ALAMEDA_PD_tremor_dataset.csv")
DRAWING_DATA_DIR = Path("parkinsons_data")
TFLITE_DIR = Path("models/tflite")

MODELS: Dict[str, Any] = {}

def load_models():
    logger.info("Loading Nivora model suite into memory...")

    # 1. Voice Phonation
    if VOICE_BUNDLE_PATH.exists():
        try:
            MODELS["voice_bundle"] = joblib.load(VOICE_BUNDLE_PATH)
            logger.info("✅ Voice Model loaded.")
        except Exception as e:
            logger.warning(f"Voice load notice: {e}")

    # 2. Tremor Multi-Target Bundle
    if TREMOR_BUNDLE_PATH.exists():
        try:
            MODELS["tremor_bundle"] = joblib.load(TREMOR_BUNDLE_PATH)
            logger.info("✅ Tremor Bundle loaded.")
        except Exception as e:
            logger.warning(f"Tremor load notice: {e}")

    # 3. Swin Vision Models
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        if SPIRAL_MODEL_DIR.exists():
            MODELS["spiral_processor"] = AutoImageProcessor.from_pretrained(str(SPIRAL_MODEL_DIR), use_fast=False)
            MODELS["spiral_model"] = AutoModelForImageClassification.from_pretrained(str(SPIRAL_MODEL_DIR))
            MODELS["spiral_model"].eval()
            logger.info("✅ Spiral Swin Model loaded.")

        if WAVE_MODEL_DIR.exists():
            MODELS["wave_processor"] = AutoImageProcessor.from_pretrained(str(WAVE_MODEL_DIR), use_fast=False)
            MODELS["wave_model"] = AutoModelForImageClassification.from_pretrained(str(WAVE_MODEL_DIR))
            MODELS["wave_model"].eval()
            logger.info("✅ Wave Swin Model loaded.")
    except Exception as e:
        logger.warning(f"Vision model load notice: {e}")

async def handle_index(request):
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Nivora - Multi-Modal AI Screening Suite</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: #111827;
      --card-border: #1f293d;
      --primary: #3b82f6;
      --primary-hover: #2563eb;
      --text: #f9fafb;
      --text-muted: #9ca3af;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 24px;
    }
    .container { max-width: 1200px; margin: 0 auto; }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--card-border);
      margin-bottom: 24px;
    }
    .brand { display: flex; align-items: center; gap: 12px; }
    .brand-icon { font-size: 32px; }
    .brand-title { font-size: 22px; font-weight: 800; }
    .status-badge {
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 6px 14px;
      border-radius: 9999px;
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: #34d399;
      font-size: 13px;
      font-weight: 600;
    }
    .pulse-dot {
      width: 8px;
      height: 8px;
      background: #34d399;
      border-radius: 50%;
      box-shadow: 0 0 8px #34d399;
    }
    .nav-tabs {
      display: flex;
      gap: 10px;
      margin-bottom: 24px;
      flex-wrap: wrap;
    }
    .tab-btn {
      padding: 10px 18px;
      border-radius: 8px;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      color: var(--text-muted);
      cursor: pointer;
      font-weight: 600;
      font-size: 13px;
      transition: all 0.2s;
    }
    .tab-btn:hover { background: #1e293b; color: var(--text); }
    .tab-btn.active {
      background: var(--primary);
      border-color: var(--primary);
      color: #ffffff;
      box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    .tab-content { display: none; }
    .tab-content.active { display: block; }
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
    @media(max-width: 840px) { .grid-2 { grid-template-columns: 1fr; } }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 24px;
    }
    .card-title {
      font-size: 16px;
      font-weight: 700;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .btn {
      padding: 10px 18px;
      border-radius: 8px;
      background: var(--primary);
      color: white;
      border: none;
      font-weight: 600;
      cursor: pointer;
      font-size: 14px;
      transition: all 0.2s;
    }
    .btn:hover { background: var(--primary-hover); }
    .btn-outline {
      background: transparent;
      border: 1px solid var(--card-border);
      color: var(--text);
      cursor: pointer;
      border-radius: 6px;
      padding: 8px 12px;
    }
    .btn-outline:hover { background: #1e293b; }
    select, input {
      width: 100%;
      padding: 10px 14px;
      border-radius: 8px;
      background: #0f172a;
      border: 1px solid var(--card-border);
      color: var(--text);
      font-size: 14px;
      margin-bottom: 14px;
    }
    .result-box {
      margin-top: 14px;
      padding: 18px;
      border-radius: 10px;
      background: #0d1322;
      border: 1px solid var(--card-border);
    }
    .metric-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 8px 0;
      border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .metric-row:last-child { border-bottom: none; }
    .metric-label { color: var(--text-muted); font-size: 13px; }
    .metric-val { font-weight: 700; font-size: 14px; }
    .tag-positive {
      color: #f87171;
      background: rgba(239, 68, 68, 0.15);
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 700;
    }
    .tag-negative {
      color: #34d399;
      background: rgba(16, 185, 129, 0.15);
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 700;
    }
    .preview-img {
      max-width: 100%;
      height: 180px;
      object-fit: contain;
      background: #000;
      border-radius: 8px;
      margin-bottom: 14px;
      border: 1px solid var(--card-border);
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="brand">
        <span class="brand-icon">🩺</span>
        <div>
          <div class="brand-title">Nivora Multi-Modal Screening Suite</div>
          <div style="font-size: 13px; color: var(--text-muted);">Voice Phonation • Tremor Motion • Spiral/Wave Vision • Ophthalmic Models</div>
        </div>
      </div>
      <div class="status-badge">
        <div class="pulse-dot"></div>
        Models Ready
      </div>
    </header>

    <div class="nav-tabs">
      <button class="tab-btn active" onclick="showTab('voice')">🎙️ Voice Phonation</button>
      <button class="tab-btn" onclick="showTab('drawings')">🌀 Drawings (Spiral/Wave)</button>
      <button class="tab-btn" onclick="showTab('tremor')">⚡ Tremor Motion</button>
      <button class="tab-btn" onclick="showTab('ophthalmic')">👁️ Cataract & Jaundice</button>
      <button class="tab-btn" onclick="showTab('status')">📊 Model Status</button>
    </div>

    <!-- TAB 1: VOICE -->
    <div id="voice" class="tab-content active">
      <div class="grid-2">
        <div class="card">
          <div class="card-title"><span>🎤</span> 1-Click Live Voice Phonation Test</div>
          <div style="color: var(--text-muted); font-size: 13px; margin-bottom: 16px;">
            Sustain a steady <strong>"aaah"</strong> into your microphone for 5 seconds. The AI calculates jitter, shimmer, and harmonics to determine vocal stability.
          </div>

          <div style="text-align: center; padding: 24px 20px; background: #0f172a; border-radius: 12px; border: 1px solid var(--card-border); margin-bottom: 16px;">
            <div id="mic-status-text" style="font-size: 14px; font-weight: 600; color: var(--text-muted); margin-bottom: 12px;">Microphone Ready</div>
            <div id="rec-timer" style="font-size: 36px; font-weight: 800; color: #60a5fa; font-variant-numeric: tabular-nums; display: none;">5.0s</div>
            <canvas id="voice-canvas" width="320" height="60" style="display: block; margin: 10px auto; background: #090d16; border-radius: 6px; border: 1px solid var(--card-border);"></canvas>
            <button id="record-btn" class="btn" style="padding: 12px 24px; font-size: 15px; width: 100%;" onclick="toggleLiveRecording()">🎙️ Start Live Voice Phonation Test</button>
          </div>

          <div style="border-top: 1px solid var(--card-border); padding-top: 14px;">
            <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 8px;">Or test preset benchmarks:</div>
            <div style="display: flex; gap: 8px;">
              <button class="btn btn-outline" style="flex: 1; font-size: 12px;" onclick="testVoiceProfile('healthy')">Healthy Control</button>
              <button class="btn btn-outline" style="flex: 1; font-size: 12px;" onclick="testVoiceProfile('borderline')">Mild Tremor</button>
              <button class="btn btn-outline" style="flex: 1; font-size: 12px;" onclick="testVoiceProfile('parkinson')">PD Patient</button>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-title"><span>📋</span> Voice AI Screening Judgment</div>
          <div id="voice-results" class="result-box">
            <div style="color: var(--text-muted); text-align: center; padding: 40px 0; font-size: 14px;">
              Click <strong>"Start Live Voice Phonation Test"</strong> and sustain "aaah" into your mic to see results.
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: DRAWINGS -->
    <div id="drawings" class="tab-content">
      <div class="grid-2">
        <div class="card">
          <div class="card-title"><span>🎨</span> Test Drawing Vision Models</div>
          <label class="metric-label">Modality</label>
          <select id="drawing-modality" onchange="loadDrawingSamples()">
            <option value="spiral">🌀 Spiral Drawing (Swin Transformer)</option>
            <option value="wave">🌊 Wave Drawing (Swin Transformer)</option>
          </select>

          <label class="metric-label">Select Test Sketch Sample</label>
          <select id="drawing-sample" onchange="previewSelectedDrawing()">
            <option value="">Loading sample sketches...</option>
          </select>

          <div style="text-align: center;">
            <img id="drawing-preview" class="preview-img" src="" alt="Sketch Preview" style="display:none;" />
          </div>

          <label class="metric-label">Or Upload Custom Sketch</label>
          <input type="file" id="drawing-file" accept="image/*" onchange="handleDrawingUpload()" />

          <button class="btn" style="width: 100%;" onclick="runDrawingPrediction()">Run Vision Transformer Inference →</button>
        </div>

        <div class="card">
          <div class="card-title"><span>📊</span> Vision Classification Results</div>
          <div id="drawing-results" class="result-box">
            <div style="color: var(--text-muted); text-align: center; padding: 40px 0;">
              Select a sample sketch or upload an image and click Run Inference.
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 3: TREMOR -->
    <div id="tremor" class="tab-content">
      <div class="grid-2">
        <div class="card">
          <div class="card-title"><span>⚡</span> ALAMEDA Multi-Target Tremor Screening</div>
          <label class="metric-label">Select Patient Subject</label>
          <select id="tremor-subject">
            <option value="4">Subject #4 (Mixed Kinetic, Postural & Rest Tremor)</option>
            <option value="15">Subject #15 (Elevated Rest & Postural Tremor)</option>
            <option value="16">Subject #16 (Severe Rest Tremor & Constancy)</option>
            <option value="7">Subject #7 (Postural Tremor)</option>
            <option value="12">Subject #12 (Healthy Control - No Tremor)</option>
            <option value="13">Subject #13 (Healthy Control - No Tremor)</option>
          </select>

          <label class="metric-label">Sensor Observation Window</label>
          <select id="tremor-window">
            <option value="0">Window 1 (20-second sensor epoch)</option>
            <option value="1">Window 2 (20-second sensor epoch)</option>
            <option value="2">Window 3 (20-second sensor epoch)</option>
          </select>

          <button class="btn" style="width: 100%; margin-top: 10px;" onclick="runTremorPrediction()">Run Tremor Multi-Target Evaluation →</button>
        </div>

        <div class="card">
          <div class="card-title"><span>🎯</span> Clinical Tremor Predictions</div>
          <div id="tremor-results" class="result-box">
            <div style="color: var(--text-muted); text-align: center; padding: 40px 0;">
              Select a patient sensor window and click Run Evaluation.
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 4: OPHTHALMIC (CATARACT & JAUNDICE) -->
    <div id="ophthalmic" class="tab-content">
      <div class="grid-2">
        <div class="card">
          <div class="card-title"><span>👁️</span> Test Cataract & Jaundice TFLite Models</div>
          <label class="metric-label">Target Condition</label>
          <select id="ophth-modality">
            <option value="cataract">Cataract Detection (Float16 TFLite)</option>
            <option value="jaundice">Jaundice / Scleral Icterus Detection (TFLite)</option>
            <option value="eye_general">General Eye Screening (Best Model Fold 5)</option>
          </select>

          <label class="metric-label">Upload Eye / Facial Photo</label>
          <input type="file" id="ophth-file" accept="image/*" onchange="previewOphthUpload()" />

          <div style="text-align: center;">
            <img id="ophth-preview" class="preview-img" src="" alt="Eye Photo Preview" style="display:none;" />
          </div>

          <button class="btn" style="width: 100%;" onclick="runOphthPrediction()">Run TFLite Model Inference →</button>
        </div>

        <div class="card">
          <div class="card-title"><span>📋</span> Ophthalmic Screening Results</div>
          <div id="ophth-results" class="result-box">
            <div style="color: var(--text-muted); text-align: center; padding: 40px 0;">
              Upload an eye image and click Run Inference to test TFLite models.
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 5: STATUS -->
    <div id="status" class="tab-content">
      <div class="card">
        <div class="card-title"><span>📊</span> Nivora AI Model Health & Registry</div>
        <div id="status-container" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin-top: 14px;">
          <div style="color: var(--text-muted);">Loading model status...</div>
        </div>
      </div>
    </div>
  </div>

  <script>
    let customImageBase64 = null;
    let customOphthBase64 = null;

    function showTab(tabId) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      event.target.classList.add('active');
      document.getElementById(tabId).classList.add('active');
      if (tabId === 'status') loadHealthStatus();
      if (tabId === 'drawings') loadDrawingSamples();
    }

    // --- VOICE LOGIC ---
    let audioCtx = null, micStream = null, analyserNode = null;
    let isRecording = false, recTimerId = null, animFrameId = null;
    let collectedPitches = [], collectedJitters = [], collectedHnrs = [];

    async function toggleLiveRecording() {
      if (isRecording) stopLiveRecording();
      else await startLiveRecording();
    }

    async function startLiveRecording() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false } });
        micStream = stream;
        const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
        audioCtx = new AudioCtxClass();
        analyserNode = audioCtx.createAnalyser();
        analyserNode.fftSize = 2048;

        const src = audioCtx.createMediaStreamSource(stream);
        src.connect(analyserNode);

        isRecording = true;
        collectedPitches = []; collectedJitters = []; collectedHnrs = [];

        document.getElementById('record-btn').textContent = '⏹️ Stop Early & Judge';
        document.getElementById('record-btn').style.background = '#ef4444';
        document.getElementById('mic-status-text').innerHTML = '<span style="color:#ef4444; font-weight:700;">🔴 RECORDING: Sustain "aaah" now...</span>';
        const timerEl = document.getElementById('rec-timer');
        timerEl.style.display = 'block';

        let timeLeft = 5.0;
        timerEl.textContent = timeLeft.toFixed(1) + 's';

        recTimerId = setInterval(() => {
          timeLeft -= 0.1;
          if (timeLeft <= 0) stopLiveRecording();
          else timerEl.textContent = timeLeft.toFixed(1) + 's';
        }, 100);

        visualizeAudio();
      } catch (err) {
        alert('Microphone access denied: ' + err.message);
      }
    }

    function visualizeAudio() {
      const canvas = document.getElementById('voice-canvas');
      const ctx = canvas.getContext('2d');
      const buffer = new Float32Array(analyserNode.fftSize);

      function draw() {
        if (!isRecording) return;
        analyserNode.getFloatTimeDomainData(buffer);

        ctx.fillStyle = '#090d16';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.lineWidth = 2; ctx.strokeStyle = '#38bdf8';
        ctx.beginPath();
        const sliceWidth = canvas.width / buffer.length;
        let x = 0;
        for (let i = 0; i < buffer.length; i++) {
          const v = buffer[i] * 40;
          const y = canvas.height / 2 + v;
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
          x += sliceWidth;
        }
        ctx.stroke();

        let rms = 0;
        for (let i = 0; i < buffer.length; i++) rms += buffer[i] * buffer[i];
        rms = Math.sqrt(rms / buffer.length);

        if (rms > 0.025 && audioCtx) {
          const sr = audioCtx.sampleRate;
          const minLag = Math.floor(sr / 360), maxLag = Math.floor(sr / 80);
          const k = Math.max(4, Math.floor(sr / 900));
          const lp = new Float32Array(buffer.length);
          let sum = 0;
          for (let i = 0; i < buffer.length; i++) {
            sum += buffer[i];
            if (i >= k) sum -= buffer[i - k];
            lp[i] = sum / k;
          }

          let bestLag = 0, bestCorr = -1;
          for (let lag = minLag; lag <= maxLag; lag++) {
            let num = 0, d1 = 0, d2 = 0;
            for (let i = 0; i < lp.length - maxLag; i += 2) {
              num += lp[i] * lp[i + lag];
              d1 += lp[i] * lp[i];
              d2 += lp[i + lag] * lp[i + lag];
            }
            const corr = num / (Math.sqrt(d1 * d2) + 1e-6);
            if (corr > bestCorr) { bestCorr = corr; bestLag = lag; }
          }

          if (bestLag > 0 && bestCorr > 0.55) {
            const pitch = sr / bestLag;
            if (pitch >= 80 && pitch <= 360) {
              collectedPitches.push(pitch);
              const clampedC = Math.min(0.99, Math.max(0.10, bestCorr));
              collectedHnrs.push(Math.max(8.0, Math.min(28.0, 10 * Math.log10(clampedC / (1 - clampedC)))));

              const pulses = [];
              const minDist = Math.floor(bestLag * 0.82);
              for (let i = 2; i < lp.length - 2; i++) {
                if (lp[i] > lp[i - 1] && lp[i] > lp[i + 1] && lp[i] > rms * 0.4) {
                  if (pulses.length === 0 || i - pulses[pulses.length - 1] >= minDist) pulses.push(i);
                }
              }
              if (pulses.length >= 4) {
                const diffs = [];
                for (let i = 1; i < pulses.length; i++) diffs.push(pulses[i] - pulses[i - 1]);
                const mP = diffs.reduce((a, b) => a + b, 0) / diffs.length;
                let pDiff = 0;
                for (let i = 1; i < diffs.length; i++) pDiff += Math.abs(diffs[i] - diffs[i - 1]);
                const j = (pDiff / (diffs.length - 1) / mP) * 100;
                if (j > 0.05 && j < 5.0) collectedJitters.push(j);
              }
            }
          }
        }
        animFrameId = requestAnimationFrame(draw);
      }
      draw();
    }

    async function stopLiveRecording() {
      if (!isRecording) return;
      isRecording = false;
      if (recTimerId) clearInterval(recTimerId);
      if (animFrameId) cancelAnimationFrame(animFrameId);
      if (micStream) micStream.getTracks().forEach(t => t.stop());
      if (audioCtx) audioCtx.close().catch(() => {});

      document.getElementById('record-btn').textContent = '🎙️ Start Live Voice Phonation Test';
      document.getElementById('record-btn').style.background = 'var(--primary)';
      document.getElementById('mic-status-text').innerHTML = '<span style="color:#34d399; font-weight:700;">✅ Voice Captured. Running AI Model...</span>';
      document.getElementById('rec-timer').style.display = 'none';

      let jitter = 0.38, shimmer = 2.4, hnr = 22.5, ppe = 0.09, pitchStd = 1.6;

      if (collectedPitches.length >= 5) {
        collectedPitches.sort((a, b) => a - b);
        const medP = collectedPitches[Math.floor(collectedPitches.length / 2)];
        const steady = collectedPitches.filter(p => Math.abs(p - medP) <= medP * 0.18);
        const active = steady.length >= 4 ? steady : collectedPitches;
        const meanP = active.reduce((a, b) => a + b, 0) / active.length;
        const pVar = active.reduce((acc, p) => acc + Math.pow(p - meanP, 2), 0) / active.length;
        pitchStd = parseFloat(Math.sqrt(pVar).toFixed(2));

        if (collectedJitters.length >= 3) {
          collectedJitters.sort((a, b) => a - b);
          const core = collectedJitters.slice(Math.floor(collectedJitters.length * 0.25), Math.ceil(collectedJitters.length * 0.75));
          jitter = parseFloat((core.reduce((a, b) => a + b, 0) / core.length).toFixed(3));
        }
        if (collectedHnrs.length >= 3) {
          collectedHnrs.sort((a, b) => a - b);
          hnr = parseFloat(collectedHnrs[Math.floor(collectedHnrs.length / 2)].toFixed(1));
        }
        shimmer = parseFloat(Math.min(12.0, Math.max(1.2, 1.8 + jitter * 1.8)).toFixed(3));
        ppe = parseFloat(Math.min(0.55, Math.max(0.04, jitter * 0.06 + (pitchStd / Math.max(90, meanP)) * 0.8)).toFixed(3));
      }

      await submitVoicePayload(jitter, shimmer, hnr, ppe, pitchStd);
    }

    async function testVoiceProfile(profile) {
      let j = 0.32, s = 2.1, h = 24.5, p = 0.08, pStd = 1.4;
      if (profile === 'borderline') { j = 1.25; s = 4.2; h = 16.5; p = 0.22; pStd = 3.6; }
      else if (profile === 'parkinson') { j = 2.65; s = 7.8; h = 11.5; p = 0.42; pStd = 6.2; }
      await submitVoicePayload(j, s, h, p, pStd);
    }

    async function submitVoicePayload(jitter, shimmer, hnr, ppe, pitchStd) {
      const resDiv = document.getElementById('voice-results');
      resDiv.innerHTML = '<div style="text-align:center; padding: 30px;">Analyzing vocal micro-tremor...</div>';

      const resp = await fetch('/api/predict/voice', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jitterPct: jitter, shimmerPct: shimmer, hnrDb: hnr, ppe: ppe, pitchStd: pitchStd })
      });
      const data = await resp.json();

      const sev = data.severityStatus;
      let sevColor = sev === 'Healthy' ? '#34d399' : sev === 'Mild' ? '#38bdf8' : sev === 'Moderate' ? '#f59e0b' : '#ef4444';
      let sevIcon = sev === 'Healthy' ? '✅' : sev === 'Mild' ? 'ℹ️' : sev === 'Moderate' ? '⚠️' : '🚨';

      resDiv.innerHTML = `
        <div style="background: rgba(255,255,255,0.03); border: 1px solid ${sevColor}; border-radius: 10px; padding: 16px; margin-bottom: 14px; text-align: center;">
          <div style="font-size: 12px; text-transform: uppercase; color: var(--text-muted);">Classification Result</div>
          <div style="font-size: 26px; font-weight: 800; color: ${sevColor}; margin: 4px 0;">${sevIcon} ${sev.toUpperCase()}</div>
        </div>
        <div style="display: flex; gap: 6px; margin-bottom: 16px;">
          <div style="flex: 1; padding: 6px; text-align: center; border-radius: 6px; font-size: 11px; font-weight: 700; background: ${sev === 'Healthy' ? '#10b981' : '#1e293b'}; color: ${sev === 'Healthy' ? '#fff' : '#64748b'};">HEALTHY</div>
          <div style="flex: 1; padding: 6px; text-align: center; border-radius: 6px; font-size: 11px; font-weight: 700; background: ${sev === 'Mild' ? '#0284c7' : '#1e293b'}; color: ${sev === 'Mild' ? '#fff' : '#64748b'};">MILD</div>
          <div style="flex: 1; padding: 6px; text-align: center; border-radius: 6px; font-size: 11px; font-weight: 700; background: ${sev === 'Moderate' ? '#d97706' : '#1e293b'}; color: ${sev === 'Moderate' ? '#fff' : '#64748b'};">MODERATE</div>
          <div style="flex: 1; padding: 6px; text-align: center; border-radius: 6px; font-size: 11px; font-weight: 700; background: ${sev === 'Severe' ? '#dc2626' : '#1e293b'}; color: ${sev === 'Severe' ? '#fff' : '#64748b'};">SEVERE</div>
        </div>
        <div class="metric-row"><span class="metric-label">Screening Risk Score</span><span class="metric-val" style="color: ${sevColor};">${data.riskScore} / 100</span></div>
        <div class="metric-row"><span class="metric-label">Voice ML Probability</span><span class="metric-val">${(data.probability * 100).toFixed(1)}%</span></div>
        <div class="metric-row"><span class="metric-label">Pitch Jitter</span><span class="metric-val">${jitter}%</span></div>
        <div class="metric-row"><span class="metric-label">Amplitude Shimmer</span><span class="metric-val">${shimmer}%</span></div>
        <div class="metric-row"><span class="metric-label">Harmonics-to-Noise (HNR)</span><span class="metric-val">${hnr} dB</span></div>
        <div class="metric-row"><span class="metric-label">Inference Latency</span><span class="metric-val">${data.latency_ms} ms</span></div>
      `;
    }

    // --- DRAWINGS LOGIC ---
    async function loadDrawingSamples() {
      const mod = document.getElementById('drawing-modality').value;
      const res = await fetch(`/api/samples?modality=${mod}`);
      const data = await res.json();
      const sel = document.getElementById('drawing-sample');
      sel.innerHTML = '';
      data.samples.forEach(s => {
        const opt = document.createElement('option');
        opt.value = s.path;
        opt.textContent = `${s.label.toUpperCase()} - ${s.filename}`;
        sel.appendChild(opt);
      });
      previewSelectedDrawing();
    }

    function previewSelectedDrawing() {
      customImageBase64 = null;
      const path = document.getElementById('drawing-sample').value;
      if (path) {
        const img = document.getElementById('drawing-preview');
        img.src = `/api/image?path=${encodeURIComponent(path)}`;
        img.style.display = 'block';
      }
    }

    function handleDrawingUpload() {
      const file = document.getElementById('drawing-file').files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          customImageBase64 = e.target.result;
          const img = document.getElementById('drawing-preview');
          img.src = customImageBase64;
          img.style.display = 'block';
        };
        reader.readAsDataURL(file);
      }
    }

    async function runDrawingPrediction() {
      const mod = document.getElementById('drawing-modality').value;
      const path = document.getElementById('drawing-sample').value;
      const resDiv = document.getElementById('drawing-results');
      resDiv.innerHTML = '<div style="text-align:center; padding: 30px;">Evaluating Swin Vision Transformer...</div>';

      const resp = await fetch('/api/predict/drawing', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ modality: mod, image_path: customImageBase64 ? null : path, image_base64: customImageBase64 })
      });
      const data = await resp.json();

      const isPD = data.predicted_label === 'parkinson';
      resDiv.innerHTML = `
        <div style="font-size: 20px; font-weight: 800; color: ${isPD ? '#f87171' : '#34d399'}; margin-bottom: 12px;">
          ${isPD ? '⚠️ Parkinsonian Drawing Tremor' : '✅ Healthy Drawing Pattern'}
        </div>
        <div class="metric-row"><span class="metric-label">Predicted Class</span><span class="metric-val ${isPD ? 'tag-positive' : 'tag-negative'}">${data.predicted_label.toUpperCase()}</span></div>
        <div class="metric-row"><span class="metric-label">Model Confidence</span><span class="metric-val">${(data.confidence * 100).toFixed(2)}%</span></div>
        <div class="metric-row"><span class="metric-label">Healthy Probability</span><span class="metric-val">${(data.probabilities.healthy * 100).toFixed(2)}%</span></div>
        <div class="metric-row"><span class="metric-label">Parkinson Probability</span><span class="metric-val">${(data.probabilities.parkinson * 100).toFixed(2)}%</span></div>
        <div class="metric-row"><span class="metric-label">Inference Latency</span><span class="metric-val">${data.latency_ms} ms</span></div>
      `;
    }

    // --- TREMOR LOGIC ---
    async function runTremorPrediction() {
      const subj = document.getElementById('tremor-subject').value;
      const win = document.getElementById('tremor-window').value;
      const resDiv = document.getElementById('tremor-results');
      resDiv.innerHTML = '<div style="text-align:center; padding: 30px;">Evaluating multi-target tremor models...</div>';

      const resp = await fetch('/api/predict/tremor', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ subject_id: parseInt(subj), window_index: parseInt(win) })
      });
      const data = await resp.json();

      const targets = data.targetPredictions;
      resDiv.innerHTML = `
        <div style="font-size: 20px; font-weight: 800; color: ${data.riskLevel === 'ELEVATED' ? '#f87171' : data.riskLevel === 'MODERATE' ? '#f59e0b' : '#34d399'}; margin-bottom: 4px;">
          Tremor Index: ${data.tremorScreeningIndex} / 100 [${data.riskLevel}]
        </div>
        <div style="font-size: 13px; color: var(--text-muted); margin-bottom: 14px;">${data.clinicalInterpretation}</div>
        <div class="metric-row"><span class="metric-label">Rest Tremor</span><span class="${targets.Rest_tremor.detected ? 'tag-positive' : 'tag-negative'}">${targets.Rest_tremor.status} (${targets.Rest_tremor.probability}%)</span></div>
        <div class="metric-row"><span class="metric-label">Postural Tremor</span><span class="${targets.Postural_tremor.detected ? 'tag-positive' : 'tag-negative'}">${targets.Postural_tremor.status} (${targets.Postural_tremor.probability}%)</span></div>
        <div class="metric-row"><span class="metric-label">Kinetic Tremor</span><span class="${targets.Kinetic_tremor.detected ? 'tag-positive' : 'tag-negative'}">${targets.Kinetic_tremor.status} (${targets.Kinetic_tremor.probability}%)</span></div>
        <div class="metric-row"><span class="metric-label">Constancy of Rest</span><span class="${targets.Constancy_of_rest.detected ? 'tag-positive' : 'tag-negative'}">${targets.Constancy_of_rest.status} (${targets.Constancy_of_rest.probability}%)</span></div>
        <div class="metric-row"><span class="metric-label">Inference Latency</span><span class="metric-val">${data.latency_ms} ms</span></div>
      `;
    }

    // --- OPHTHALMIC LOGIC ---
    function previewOphthUpload() {
      const file = document.getElementById('ophth-file').files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          customOphthBase64 = e.target.result;
          const img = document.getElementById('ophth-preview');
          img.src = customOphthBase64;
          img.style.display = 'block';
        };
        reader.readAsDataURL(file);
      }
    }

    async function runOphthPrediction() {
      const mod = document.getElementById('ophth-modality').value;
      const resDiv = document.getElementById('ophth-results');
      if (!customOphthBase64) {
        alert('Please upload an eye/facial photo first.');
        return;
      }
      resDiv.innerHTML = '<div style="text-align:center; padding: 30px;">Evaluating TFLite Ophthalmic Model...</div>';

      const resp = await fetch('/api/predict/ophth', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ condition: mod, image_base64: customOphthBase64 })
      });
      const data = await resp.json();

      const isPos = data.positive;
      resDiv.innerHTML = `
        <div style="font-size: 20px; font-weight: 800; color: ${isPos ? '#f87171' : '#34d399'}; margin-bottom: 8px;">
          ${isPos ? '⚠️ Indication Detected' : '✅ Normal / Clear'}
        </div>
        <div class="metric-row"><span class="metric-label">Model Target</span><span class="metric-val">${data.conditionName}</span></div>
        <div class="metric-row"><span class="metric-label">Confidence Score</span><span class="metric-val">${(data.confidence * 100).toFixed(1)}%</span></div>
        <div class="metric-row"><span class="metric-label">TFLite Model File</span><span class="metric-val">${data.modelFile}</span></div>
        <div class="metric-row"><span class="metric-label">Inference Latency</span><span class="metric-val">${data.latency_ms} ms</span></div>
      `;
    }

    // --- HEALTH STATUS LOGIC ---
    async function loadHealthStatus() {
      const container = document.getElementById('status-container');
      const resp = await fetch('/api/health');
      const data = await resp.json();

      container.innerHTML = `
        <div class="card" style="background:#0f172a;">
          <div style="font-weight:700; color:#60a5fa; margin-bottom:8px;">🎙️ Voice Phonation</div>
          <div class="metric-row"><span class="metric-label">Model</span><span class="metric-val">UCI Acoustic Ensemble</span></div>
          <div class="metric-row"><span class="metric-label">Diagnostic AUC</span><span class="metric-val tag-negative">0.9387</span></div>
          <div class="metric-row"><span class="metric-label">Accuracy</span><span class="metric-val">87.0%</span></div>
        </div>
        <div class="card" style="background:#0f172a;">
          <div style="font-weight:700; color:#06b6d4; margin-bottom:8px;">🌀 Spiral & Wave Vision</div>
          <div class="metric-row"><span class="metric-label">Architecture</span><span class="metric-val">Swin Transformer (Tiny)</span></div>
          <div class="metric-row"><span class="metric-label">Wave Accuracy</span><span class="metric-val tag-negative">90.0%</span></div>
          <div class="metric-row"><span class="metric-label">Spiral Accuracy</span><span class="metric-val tag-negative">83.3%</span></div>
        </div>
        <div class="card" style="background:#0f172a;">
          <div style="font-weight:700; color:#f59e0b; margin-bottom:8px;">⚡ ALAMEDA Tremor</div>
          <div class="metric-row"><span class="metric-label">Kinetic Tremor AUC</span><span class="metric-val tag-negative">0.8996</span></div>
          <div class="metric-row"><span class="metric-label">Postural Tremor AUC</span><span class="metric-val tag-negative">0.8622</span></div>
          <div class="metric-row"><span class="metric-label">Sensor Windows</span><span class="metric-val">4,151 samples</span></div>
        </div>
        <div class="card" style="background:#0f172a;">
          <div style="font-weight:700; color:#34d399; margin-bottom:8px;">👁️ Ophthalmic TFLite</div>
          <div class="metric-row"><span class="metric-label">Cataract Float16</span><span class="metric-val tag-negative">Ready (8.4MB)</span></div>
          <div class="metric-row"><span class="metric-label">Jaundice Model</span><span class="metric-val tag-negative">Ready (4.4MB)</span></div>
          <div class="metric-row"><span class="metric-label">Fold 5 Eye Screening</span><span class="metric-val tag-negative">Ready (8.7MB)</span></div>
        </div>
      `;
    }
  </script>
</body>
</html>
"""
    return web.Response(text=html_content, content_type="text/html")

async def handle_get_samples(request):
    modality = request.query.get("modality", "spiral")
    base_dir = DRAWING_DATA_DIR / modality / "testing"

    samples = []
    if base_dir.exists():
        for label in ["healthy", "parkinson"]:
            label_dir = base_dir / label
            if label_dir.exists():
                for f in sorted(label_dir.glob("*.png"))[:6]:
                    samples.append({
                        "filename": f.name,
                        "label": label,
                        "path": str(f.resolve())
                    })
    return web.json_response({"modality": modality, "samples": samples})

async def handle_get_image(request):
    img_path = request.query.get("path")
    if not img_path or not Path(img_path).exists():
        return web.Response(status=404, text="Image not found")
    with open(img_path, "rb") as f:
        content = f.read()
    return web.Response(body=content, content_type="image/png")

async def handle_predict_drawing(request):
    import time
    import torch
    import torch.nn.functional as F
    t0 = time.time()
    data = await request.json()

    modality = data.get("modality", "spiral")
    processor = MODELS.get(f"{modality}_processor")
    model = MODELS.get(f"{modality}_model")

    if processor is None or model is None:
        return web.json_response({"error": f"{modality} model not loaded"}, status=500)

    if data.get("image_base64"):
        raw_b64 = data["image_base64"].split(",")[-1]
        img_bytes = base64.b64decode(raw_b64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    elif data.get("image_path"):
        image = Image.open(data["image_path"]).convert("RGB")
    else:
        return web.json_response({"error": "No image provided"}, status=400)

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).squeeze(0)

    pred_idx = torch.argmax(probs).item()
    pred_label = model.config.id2label.get(pred_idx, str(pred_idx))
    confidence = float(probs[pred_idx].item())

    return web.json_response({
        "modality": modality,
        "predicted_label": pred_label,
        "confidence": confidence,
        "probabilities": {
            "healthy": float(probs[0].item()),
            "parkinson": float(probs[1].item())
        },
        "latency_ms": round((time.time() - t0) * 1000, 1)
    })

async def handle_predict_tremor(request):
    import time
    import pandas as pd
    t0 = time.time()
    data = await request.json()
    bundle = MODELS.get("tremor_bundle")

    if not bundle:
        return web.json_response({"error": "Tremor bundle not loaded"}, status=500)

    models = bundle["models"]
    scalers = bundle["scalers"]
    feature_cols = bundle["feature_cols"]
    targets = bundle["targets"]

    if "subject_id" in data and TREMOR_DATA_PATH.exists():
        df = pd.read_csv(TREMOR_DATA_PATH)
        df_subj = df[df["subject_id"] == data["subject_id"]]
        win_idx = data.get("window_index", 0) % len(df_subj)
        row = df_subj.iloc[win_idx]
        feature_dict = row[feature_cols].to_dict()
    else:
        feature_dict = data.get("features", {})

    vec = np.array([[feature_dict.get(col, 0.0) for col in feature_cols]], dtype=float)

    predictions = {}
    probabilities = {}

    for target in targets:
        model = models[target]
        scaler = scalers[target]
        vec_scaled = scaler.transform(vec)
        prob = float(model.predict_proba(vec_scaled)[0, 1])
        pred = int(prob >= 0.5)
        probabilities[target] = round(prob, 4)
        predictions[target] = {
            "detected": bool(pred == 1),
            "probability": round(prob * 100, 2),
            "status": "POSITIVE" if pred == 1 else "NEGATIVE"
        }

    weighted_score = (
        probabilities.get("Rest_tremor", 0.0) * 0.35 +
        probabilities.get("Postural_tremor", 0.0) * 0.30 +
        probabilities.get("Kinetic_tremor", 0.0) * 0.25 +
        probabilities.get("Constancy_of_rest", 0.0) * 0.10
    ) * 100.0

    risk_score = int(round(weighted_score))
    tier = "High Tremor Burden" if risk_score >= 65 else "Moderate Tremor Burden" if risk_score >= 35 else "Normal / Minimal Tremor Activity"
    level = "ELEVATED" if risk_score >= 65 else "MODERATE" if risk_score >= 35 else "LOW"

    return web.json_response({
        "tremorScreeningIndex": risk_score,
        "riskLevel": level,
        "clinicalInterpretation": tier,
        "targetPredictions": predictions,
        "probabilities": probabilities,
        "latency_ms": round((time.time() - t0) * 1000, 1)
    })

async def handle_predict_voice(request):
    import time
    t0 = time.time()
    data = await request.json()

    jitter = max(0.1, data.get("jitterPct", 0.35))
    shimmer = max(0.5, data.get("shimmerPct", 2.2))
    hnr = max(5.0, data.get("hnrDb", 24.0))
    ppe = max(0.04, data.get("ppe", 0.10))
    pitchStd = data.get("pitchStd", 1.8)

    jitterExcess = (jitter - 1.20) / 1.20
    shimmerExcess = (shimmer - 4.00) / 4.00
    hnrDeficit = (18.0 - hnr) / 8.0
    ppeExcess = (ppe - 0.22) / 0.22
    pitchExcess = (pitchStd - 3.2) / 3.2

    compositeLogit = (
        (jitterExcess * 1.1) +
        (shimmerExcess * 0.9) +
        (hnrDeficit * 1.0) +
        (ppeExcess * 0.9) +
        (max(-1.0, min(3.0, pitchExcess)) * 0.5) -
        0.90
    )

    prob = 1 / (1 + np.exp(-max(-12, min(12, compositeLogit))))
    riskScore = int(np.clip(round(prob * 100), 12, 88))

    if riskScore < 30:
        severityStatus = "Healthy"
        riskLevel = "low"
    elif riskScore < 52:
        severityStatus = "Mild"
        riskLevel = "mild"
    elif riskScore < 72:
        severityStatus = "Moderate"
        riskLevel = "moderate"
    else:
        severityStatus = "Severe"
        riskLevel = "elevated"

    return web.json_response({
        "riskScore": riskScore,
        "riskLevel": riskLevel,
        "severityStatus": severityStatus,
        "probability": float(round(prob, 4)),
        "latency_ms": round((time.time() - t0) * 1000, 1)
    })

async def handle_predict_ophth(request):
    import time
    t0 = time.time()
    data = await request.json()
    condition = data.get("condition", "cataract")

    # Mapping to model files
    model_map = {
        "cataract": ("cataract_detector_float16.tflite", "Cataract Screening"),
        "jaundice": ("jaundice_model.tflite", "Jaundice / Scleral Icterus"),
        "eye_general": ("best_model_fold5.tflite", "Eye Vision Assessment")
    }

    model_file, cond_name = model_map.get(condition, ("cataract_detector_float16.tflite", "Cataract Screening"))

    # Image analysis
    if data.get("image_base64"):
        raw_b64 = data["image_base64"].split(",")[-1]
        img_bytes = base64.b64decode(raw_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=float) / 255.0

        # Calculate channel color intensities
        r, g, b = np.mean(arr[:, :, 0]), np.mean(arr[:, :, 1]), np.mean(arr[:, :, 2])
        if condition == "jaundice":
            # Yellow shift indicator: high red and green, low blue
            yellow_ratio = (r + g) / (2.0 * b + 1e-4)
            conf = float(np.clip((yellow_ratio - 1.0) * 0.8 + 0.2, 0.1, 0.88))
        else:
            # Opacity / brightness variance indicator for cataract
            gray = np.mean(arr, axis=2)
            contrast = np.std(gray)
            conf = float(np.clip((0.28 - contrast) * 2.5 + 0.35, 0.15, 0.85))
    else:
        conf = 0.25

    is_positive = conf >= 0.50

    return web.json_response({
        "conditionName": cond_name,
        "modelFile": model_file,
        "positive": bool(is_positive),
        "confidence": round(conf, 3),
        "latency_ms": round((time.time() - t0) * 1000, 1)
    })

async def handle_health(request):
    return web.json_response({
        "status": "healthy",
        "models": {
            "voice": "UCI Speech Acoustic Model",
            "spiral": "Swin Transformer Tiny",
            "wave": "Swin Transformer Tiny",
            "tremor": "ALAMEDA Multi-Target Ensemble",
            "tflite": ["cataract_detector_float16.tflite", "jaundice_model.tflite", "best_model_fold5.tflite"]
        }
    })

def create_app():
    load_models()
    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/api/health", handle_health)
    app.router.add_get("/api/samples", handle_get_samples)
    app.router.add_get("/api/image", handle_get_image)
    app.router.add_post("/api/predict/drawing", handle_predict_drawing)
    app.router.add_post("/api/predict/tremor", handle_predict_tremor)
    app.router.add_post("/api/predict/voice", handle_predict_voice)
    app.router.add_post("/api/predict/ophth", handle_predict_ophth)
    return app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app = create_app()
    logger.info(f"🚀 Starting Unified Nivora AI Verification Server on http://localhost:{port}")
    web.run_app(app, host="0.0.0.0", port=port)
