#!/usr/bin/env python3
# 5-fold CV metrics for your report (text baseline).
# Outputs SMAPE (official), MAE, RMSE, R2, MedianAE, MAPE, Spearman ρ, δ-accuracy @5/10/20%.

import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error
from scipy.stats import spearmanr

try:
    from xgboost import XGBRegressor
    def make_model():
        return XGBRegressor(
            n_estimators=600, max_depth=8, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.8,
            objective="reg:absoluteerror", tree_method="hist",
            random_state=1337, n_jobs=os.cpu_count()
        )
except Exception:
    from sklearn.ensemble import HistGradientBoostingRegressor
    def make_model():
        return HistGradientBoostingRegressor(max_depth=8, learning_rate=0.07,
                                             max_iter=600, random_state=1337)

def smape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(np.clip(y_pred, 1e-6, None), float)
    return float(np.mean(np.abs(y_true - y_pred) / np.clip((np.abs(y_true)+np.abs(y_pred))/2.0, 1e-6, None))) * 100.0

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(np.clip(y_pred, 1e-6, None), float)
    mask = np.abs(y_true) > 1e-6
    if mask.sum() == 0: return np.nan
    return float(np.mean(np.abs((y_true[mask]-y_pred[mask]) / y_true[mask]))) * 100.0

def delta_accuracy(y_true, y_pred, pct):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    rel = np.abs(y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None)
    return float(np.mean(rel <= pct)) * 100.0

def main():
    train_csv = os.path.join("dataset","train.csv")
    if not os.path.exists(train_csv):
        raise FileNotFoundError("dataset/train.csv not found")
    df = pd.read_csv(train_csv)
    assert {"sample_id","catalog_content","image_link","price"} <= set(df.columns)

    y = df["price"].values.astype(float)
    tfidf_w = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.95, max_features=120_000, strip_accents="unicode")
    tfidf_c = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=3, max_df=0.98, max_features=100_000)

    Xw = tfidf_w.fit_transform(df["catalog_content"].astype(str))
    Xc = tfidf_c.fit_transform(df["catalog_content"].astype(str))
    X  = Xw + Xc

    kf = KFold(n_splits=5, shuffle=True, random_state=1337)
    oof = np.zeros(len(df), float)
    for i, (tr, va) in enumerate(kf.split(X), start=1):
        m = make_model(); m.fit(X[tr], y[tr])
        oof[va] = np.clip(m.predict(X[va]), 1e-6, None)
        print(f"[Fold {i}] MAE={mean_absolute_error(y[va], oof[va]):.3f} | SMAPE={smape(y[va], oof[va]):.2f}")

    metrics = {
        "SMAPE(%)": round(smape(y, oof), 4),
        "MAE": round(mean_absolute_error(y, oof), 4),
        "RMSE": round(float(np.sqrt(np.mean((y - oof)**2))), 4),
        "R2": round(r2_score(y, oof), 6),
        "MedianAE": round(median_absolute_error(y, oof), 4),
        "MAPE(%)": round(mape(y, oof), 4),
        "Spearman_rho": round(float(spearmanr(y, oof).correlation), 6),
        "DeltaAcc@5%(%)": round(delta_accuracy(y, oof, 0.05), 2),
        "DeltaAcc@10%(%)": round(delta_accuracy(y, oof, 0.10), 2),
        "DeltaAcc@20%(%)": round(delta_accuracy(y, oof, 0.20), 2),
    }
    print("\n=== Cross-Validation Metrics (5-fold, text-only baseline) ===")
    for k,v in metrics.items(): print(f"{k:>18}: {v}")

    os.makedirs("outputs_metrics", exist_ok=True)
    with open(os.path.join("outputs_metrics","cv_metrics_report.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved → outputs_metrics/cv_metrics_report.json")

if __name__ == "__main__":
    main()
