import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir())
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

def generate_graphs():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(output_dir, 'pd_speech_features.csv')
    
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path, header=1)
    
    print(f"Dataset Shape: {df.shape}")
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(18, 11), dpi=300)
    
    # -------------------------------------------------------------
    # 1. Target Class Distribution
    # -------------------------------------------------------------
    ax1 = plt.subplot(2, 3, 1)
    class_counts = df['class'].value_counts().sort_index()
    colors = ['#10B981', '#EF4444']
    labels = ['Healthy (0)', "Parkinson's (1)"]
    bars = ax1.bar(labels, class_counts.values, color=colors, width=0.45, edgecolor='#334155', linewidth=1.2)
    ax1.set_title("Target Class Balance", fontsize=13, fontweight='bold', pad=10)
    ax1.set_ylabel("Number of Patients", fontsize=11)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    for bar in bars:
        height = bar.get_height()
        pct = height / len(df) * 100
        ax1.text(bar.get_x() + bar.get_width()/2., height + 8,
                 f"{int(height)} ({pct:.1f}%)", ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # -------------------------------------------------------------
    # 2. Jitter (%) Distribution: Healthy vs Parkinson's
    # -------------------------------------------------------------
    ax2 = plt.subplot(2, 3, 2)
    jitter_hc = df[df['class'] == 0]['locPctJitter'] * 100
    jitter_pd = df[df['class'] == 1]['locPctJitter'] * 100
    bp1 = ax2.boxplot([jitter_hc, jitter_pd], patch_artist=True, tick_labels=['Healthy', "Parkinson's"], showfliers=False)
    bp1['boxes'][0].set_facecolor('#A7F3D0')
    bp1['boxes'][1].set_facecolor('#FECACA')
    ax2.axhline(1.04, color='#D97706', linestyle='--', linewidth=1.5, label='Clinical Norm (<1.04%)')
    ax2.set_title("Local Jitter (Pitch Perturbation %)", fontsize=13, fontweight='bold', pad=10)
    ax2.set_ylabel("Jitter (%)", fontsize=11)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # -------------------------------------------------------------
    # 3. Harmonics-to-Noise Ratio (HNR in dB)
    # -------------------------------------------------------------
    ax3 = plt.subplot(2, 3, 3)
    hnr_hc = df[df['class'] == 0]['meanHarmToNoiseHarmonicity']
    hnr_pd = df[df['class'] == 1]['meanHarmToNoiseHarmonicity']
    bp2 = ax3.boxplot([hnr_hc, hnr_pd], patch_artist=True, tick_labels=['Healthy', "Parkinson's"], showfliers=False)
    bp2['boxes'][0].set_facecolor('#A7F3D0')
    bp2['boxes'][1].set_facecolor('#FECACA')
    ax3.axhline(20.0, color='#2563EB', linestyle='--', linewidth=1.5, label='Purity Baseline (>20 dB)')
    ax3.set_title("Harmonics-to-Noise Ratio (HNR)", fontsize=13, fontweight='bold', pad=10)
    ax3.set_ylabel("HNR (dB)", fontsize=11)
    ax3.legend(loc='lower left', fontsize=9)
    ax3.grid(True, linestyle='--', alpha=0.6)

    # -------------------------------------------------------------
    # 4. Scatter: Jitter vs Shimmer vs Disease Status
    # -------------------------------------------------------------
    ax4 = plt.subplot(2, 3, 4)
    sc0 = ax4.scatter(df[df['class'] == 0]['locPctJitter']*100, df[df['class'] == 0]['locShimmer']*100, 
                      c='#10B981', alpha=0.6, edgecolors='none', label='Healthy Control', s=35)
    sc1 = ax4.scatter(df[df['class'] == 1]['locPctJitter']*100, df[df['class'] == 1]['locShimmer']*100, 
                      c='#EF4444', alpha=0.5, edgecolors='none', label="Parkinson's Patient", s=35)
    ax4.set_xlim(0, 3.0)
    ax4.set_ylim(0, 15)
    ax4.set_title("Acoustic Dysphonia: Jitter vs Shimmer", fontsize=13, fontweight='bold', pad=10)
    ax4.set_xlabel("Local Jitter (%)", fontsize=11)
    ax4.set_ylabel("Local Shimmer (%)", fontsize=11)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, linestyle='--', alpha=0.6)

    # -------------------------------------------------------------
    # 5. Machine Learning Feature Importance (Random Forest)
    # -------------------------------------------------------------
    ax5 = plt.subplot(2, 3, 5)
    
    # Feature columns (exclude id and class)
    X = df.drop(columns=['id', 'class'], errors='ignore')
    y = df['class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(8)
    
    y_pos = np.arange(len(top_features))
    ax5.barh(y_pos, top_features.values, color='#0284C7', edgecolor='#0369A1', height=0.6)
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(top_features.index, fontsize=9)
    ax5.invert_yaxis()
    ax5.set_title("Top Predictive Vocal Biomarkers", fontsize=13, fontweight='bold', pad=10)
    ax5.set_xlabel("Random Forest Feature Importance", fontsize=11)
    ax5.grid(axis='x', linestyle='--', alpha=0.6)

    # -------------------------------------------------------------
    # 6. ROC Curve & AUC Score
    # -------------------------------------------------------------
    ax6 = plt.subplot(2, 3, 6)
    y_scores = rf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    
    ax6.plot(fpr, tpr, color='#0284C7', lw=2.5, label=f'Random Forest (AUC = {roc_auc:.3f})')
    ax6.plot([0, 1], [0, 1], color='#94A3B8', lw=1.5, linestyle='--', label='Baseline (AUC = 0.50)')
    ax6.fill_between(fpr, tpr, alpha=0.15, color='#0284C7')
    ax6.set_xlim([0.0, 1.0])
    ax6.set_ylim([0.0, 1.05])
    ax6.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    ax6.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    ax6.set_title('Diagnostic Screening ROC Curve', fontsize=13, fontweight='bold', pad=10)
    ax6.legend(loc="lower right", fontsize=9)
    ax6.grid(True, linestyle='--', alpha=0.6)

    # Global Title & Layout
    fig.suptitle("UCI Parkinson's Disease Speech Features & Acoustic Biomarkers Report", 
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    plot_file = os.path.join(output_dir, 'parkinson_acoustic_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Master analysis graph saved successfully to: {plot_file}")
    print(f"✅ Model Diagnostic AUC: {roc_auc:.4f}")

if __name__ == '__main__':
    generate_graphs()
