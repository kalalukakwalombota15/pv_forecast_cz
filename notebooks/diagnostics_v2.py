#!/usr/bin/env python3
"""
diagnostics_v2.py — Day-ahead PV forecast diagnostics (Tier 1 + PIT calibration).
Reads predictions_v2.parquet (actual/p10/p50/p90) and joins clearness_index
from master_features_v2.parquet. Outputs CSVs to outputs/ and PNGs to outputs/figures/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ============================ CONFIG ============================
PRED_PATH     = "outputs/predictions_v2.parquet"     # has: actual, p10, p50, p90
MASTER_PATH   = "data/master_features_v2.parquet"    # has: clearness_index (+ all features)

COL_ACTUAL    = "actual"
COL_P50       = "p50"
COL_P10       = "p10"
COL_P90       = "p90"
COL_CLEARNESS = "clearness_index"

DAYLIGHT_MIN_MW = 1.0
MAG_EDGES   = [0, 200, 500, 800, np.inf]
MAG_LABELS  = ["0-200", "200-500", "500-800", "800+"]

OUT_DIR     = "outputs"
FIG_DIR     = "outputs/figures"
# ================================================================

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def load_data():
    test = pd.read_parquet(PRED_PATH)
    test.index = pd.to_datetime(test.index, utc=True)

    # join clearness_index from master features
    master = pd.read_parquet(MASTER_PATH)
    master.index = pd.to_datetime(master.index, utc=True)
    if COL_CLEARNESS in master.columns:
        test = test.join(master[[COL_CLEARNESS]], how="left")

    required = [COL_ACTUAL, COL_P50, COL_P10, COL_P90]
    missing = [c for c in required if c not in test.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}. Available: {list(test.columns)}")

    test = test.dropna(subset=required)
    test["hour"]     = test.index.hour
    test["residual"] = test[COL_ACTUAL] - test[COL_P50]   # +ve = under-forecast
    print(f"[load] test rows: {len(test)}  range: {test.index.min()} -> {test.index.max()}")
    return test


def daylight(test):
    return test[test[COL_ACTUAL] >= DAYLIGHT_MIN_MW].copy()


def error_by_hour(test):
    d = daylight(test)
    g = d.groupby("hour")
    out = pd.DataFrame({
        "ME":  g["residual"].mean(),
        "MAE": g["residual"].apply(lambda x: x.abs().mean()),
        "n":   g.size(),
    }).reset_index()
    out.to_csv(f"{OUT_DIR}/diag_error_by_hour.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    colors = ["#c0392b" if v < 0 else "#27ae60" for v in out["ME"]]
    ax1.bar(out["hour"], out["ME"], color=colors)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_ylabel("Mean error (MW)\n[+ = under-forecast]")
    ax1.set_title("Signed error by hour of day (daylight, full test set)")
    ax2.bar(out["hour"], out["MAE"], color="#2c3e50")
    ax2.set_ylabel("MAE (MW)")
    ax2.set_xlabel("Hour of day")
    ax2.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/diag_error_by_hour.png", dpi=150)
    plt.close(fig)
    print("[1.1] error_by_hour done")


def error_by_magnitude(test):
    d = daylight(test)
    d["mag_bin"] = pd.cut(d[COL_ACTUAL], bins=MAG_EDGES, labels=MAG_LABELS,
                          include_lowest=True, right=False)
    g = d.groupby("mag_bin", observed=True)
    out = pd.DataFrame({
        "MAE": g["residual"].apply(lambda x: x.abs().mean()),
        "ME":  g["residual"].mean(),
        "n":   g.size(),
    }).reset_index()
    out.to_csv(f"{OUT_DIR}/diag_error_by_magnitude.csv", index=False)

    x = np.arange(len(out)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, out["MAE"], w, label="MAE", color="#2c3e50")
    ax.bar(x + w/2, out["ME"],  w, label="ME (signed)", color="#e67e22")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(out["mag_bin"])
    ax.set_xlabel("Actual generation bin (MW)")
    ax.set_ylabel("MW")
    ax.set_title("Error by output magnitude (daylight, full test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/diag_error_by_magnitude.png", dpi=150)
    plt.close(fig)
    print("[1.2] error_by_magnitude done")


def error_by_clearness(test):
    if COL_CLEARNESS not in test.columns:
        print(f"[1.3] SKIPPED - '{COL_CLEARNESS}' not in columns")
        return
    d = daylight(test).dropna(subset=[COL_CLEARNESS])
    if len(d) == 0:
        print("[1.3] SKIPPED - no clearness data after daylight filter")
        return
    q1, q2 = d[COL_CLEARNESS].quantile([1/3, 2/3])
    def regime(v):
        if v <= q1: return "overcast"
        if v <= q2: return "partly"
        return "clear"
    d["regime"] = d[COL_CLEARNESS].apply(regime)
    order = ["overcast", "partly", "clear"]
    g = d.groupby("regime", observed=True)
    out = pd.DataFrame({
        "MAE": g["residual"].apply(lambda x: x.abs().mean()),
        "ME":  g["residual"].mean(),
        "n":   g.size(),
    }).reindex(order).reset_index()
    out.to_csv(f"{OUT_DIR}/diag_error_by_clearness.csv", index=False)

    x = np.arange(len(out)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, out["MAE"], w, label="MAE", color="#2c3e50")
    ax.bar(x + w/2, out["ME"],  w, label="ME (signed)", color="#e67e22")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(out["regime"])
    ax.set_xlabel(f"Clearness regime (tertile cuts: {q1:.3f}, {q2:.3f})")
    ax.set_ylabel("MW")
    ax.set_title("Error by clearness regime (daylight, full test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/diag_error_by_clearness.png", dpi=150)
    plt.close(fig)
    print(f"[1.3] error_by_clearness done - cuts {q1:.3f}, {q2:.3f}")


def residual_distribution(test):
    d = daylight(test)
    r = d["residual"].values
    summary = pd.DataFrame({
        "metric": ["mean", "std", "skew", "kurtosis", "q05", "q50", "q95", "n"],
        "value":  [np.mean(r), np.std(r, ddof=1), stats.skew(r), stats.kurtosis(r),
                   np.quantile(r, 0.05), np.quantile(r, 0.50),
                   np.quantile(r, 0.95), len(r)],
    })
    summary.to_csv(f"{OUT_DIR}/diag_residual_stats.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(r, bins=60, density=True, color="#95a5a6", alpha=0.7)
    xs = np.linspace(r.min(), r.max(), 400)
    kde = stats.gaussian_kde(r)
    ax.plot(xs, kde(xs), color="#2c3e50", lw=2)
    ax.axvline(0, color="black", lw=1)
    ax.axvline(np.mean(r), color="#c0392b", lw=1.5, ls="--",
               label=f"mean = {np.mean(r):.1f} MW")
    ax.set_xlabel("Residual: actual - forecast (MW)  [+ = under-forecast]")
    ax.set_ylabel("Density")
    ax.set_title(f"Residual distribution (daylight)  skew = {stats.skew(r):.2f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/diag_residual_dist.png", dpi=150)
    plt.close(fig)
    print(f"[1.4] residual_distribution done - mean {np.mean(r):.1f}, skew {stats.skew(r):.2f}")


def pit_histogram(test):
    d = daylight(test)
    a   = d[COL_ACTUAL].values
    p10 = d[COL_P10].values
    p50 = d[COL_P50].values
    p90 = d[COL_P90].values

    pit = np.empty_like(a, dtype=float)
    for i in range(len(a)):
        y = a[i]
        if y <= p10[i]:
            pit[i] = max(0.10 * (y - (p10[i] - (p50[i] - p10[i]))) / max(p50[i] - p10[i], 1e-6), 0.0)
        elif y <= p50[i]:
            pit[i] = 0.10 + 0.40 * (y - p10[i]) / max(p50[i] - p10[i], 1e-6)
        elif y <= p90[i]:
            pit[i] = 0.50 + 0.40 * (y - p50[i]) / max(p90[i] - p50[i], 1e-6)
        else:
            pit[i] = min(0.90 + 0.10 * (y - p90[i]) / max(p90[i] - p50[i], 1e-6), 1.0)
    pit = np.clip(pit, 0, 1)

    picp_80 = np.mean((a >= p10) & (a <= p90))
    stats_df = pd.DataFrame({
        "metric": ["PICP_p10_p90 (nominal 0.80)", "PIT_mean (ideal 0.50)",
                   "PIT_std (ideal ~0.289)", "n"],
        "value":  [picp_80, pit.mean(), pit.std(), len(pit)],
    })
    stats_df.to_csv(f"{OUT_DIR}/diag_pit_stats.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pit, bins=10, range=(0, 1), density=True,
            color="#2980b9", alpha=0.8, edgecolor="white")
    ax.axhline(1.0, color="#c0392b", lw=1.5, ls="--", label="ideal uniform")
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density")
    ax.set_title(f"PIT histogram (daylight)  PICP[p10,p90] = {picp_80:.3f} vs 0.80")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/diag_pit_hist.png", dpi=150)
    plt.close(fig)
    print(f"[PIT] done - PICP(80%)={picp_80:.3f}, PIT mean={pit.mean():.3f}")


def main():
    test = load_data()
    error_by_hour(test)
    error_by_magnitude(test)
    error_by_clearness(test)
    residual_distribution(test)
    pit_histogram(test)
    print("\nAll diagnostics complete. CSVs in outputs/, figures in outputs/figures/")


if __name__ == "__main__":
    main()
