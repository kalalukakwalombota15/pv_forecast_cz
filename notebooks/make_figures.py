import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import json

RES_DIR = Path("outputs")
FIG_DIR = Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 130,
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

print("Loading predictions...")
pred = pd.read_parquet(RES_DIR / "predictions_v2.parquet")
pred.index = pd.to_datetime(pred.index, utc=True)

# ── Figure 1: Actual vs Predicted — a sample week ─────────
print("Figure 1: actual vs predicted (sample week)...")
# pick a sunny week in the test set
sample = pred.loc['2025-10-06':'2025-10-12']
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(sample.index, sample['actual'], label='Actual', color='#1A1A1A', linewidth=1.8)
ax.plot(sample.index, sample['p50'], label='Forecast (p50)', color='#1D9E75', linewidth=1.8, linestyle='--')
ax.set_ylabel('Solar generation (MW)')
ax.set_title('Day-Ahead Forecast vs Actual — Sample Week (Oct 2025)')
ax.legend(loc='upper right')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(FIG_DIR / "fig1_actual_vs_predicted.png", bbox_inches='tight')
plt.close(fig)

# ── Figure 2: Forecast with uncertainty band ──────────────
print("Figure 2: forecast with uncertainty band...")
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.fill_between(sample.index, sample['p10'], sample['p90'],
                color='#1D9E75', alpha=0.25, label='p10-p90 band')
ax.plot(sample.index, sample['p50'], label='Forecast (p50)', color='#1D9E75', linewidth=1.8)
ax.plot(sample.index, sample['actual'], label='Actual', color='#1A1A1A', linewidth=1.4)
ax.set_ylabel('Solar generation (MW)')
ax.set_title('Day-Ahead Forecast with p10-p90 Uncertainty Band — Sample Week')
ax.legend(loc='upper right')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_uncertainty_band.png", bbox_inches='tight')
plt.close(fig)

# ── Figure 3: Feature importance ──────────────────────────
print("Figure 3: feature importance...")
imp = pd.read_csv(RES_DIR / "feature_importance_v2.csv").head(15)
imp = imp.iloc[::-1]  # ascending for horizontal bar
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(imp['feature'], imp['importance'], color='#0C447C')
ax.set_xlabel('Importance (split count)')
ax.set_title('Top 15 Feature Importances — Day-Ahead LightGBM')
ax.grid(axis='y', alpha=0)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_feature_importance.png", bbox_inches='tight')
plt.close(fig)

# ── Figure 4: Model comparison ────────────────────────────
print("Figure 4: model comparison...")
with open(RES_DIR / "metrics_v2.json") as f:
    metrics = json.load(f)

models = ['Physical_baseline', 'Persistence_24h', 'LightGBM_dayahead']
labels = ['Physical\nbaseline', 'Persistence', 'LightGBM\nday-ahead']
mae_vals  = [metrics[m]['MAE'] for m in models]
rmse_vals = [metrics[m]['RMSE'] for m in models]

x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.bar(x - width/2, mae_vals, width, label='MAE', color='#1D9E75')
b2 = ax.bar(x + width/2, rmse_vals, width, label='RMSE', color='#0C447C')
ax.set_ylabel('Error (MW)')
ax.set_title('Model Comparison on Test Set (Oct-Dec 2025)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
for b in list(b1) + list(b2):
    ax.annotate(f'{b.get_height():.0f}',
                xy=(b.get_x() + b.get_width()/2, b.get_height()),
                xytext=(0, 3), textcoords='offset points',
                ha='center', fontsize=9)
ax.grid(axis='x', alpha=0)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_model_comparison.png", bbox_inches='tight')
plt.close(fig)

print(f"\nAll 4 figures saved to {FIG_DIR}")
