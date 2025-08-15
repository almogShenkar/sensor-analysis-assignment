# Task Summary — Answers (Nexar Sensor Analysis)

This document answers the required items from the README **clearly and directly**, based on your latest runs.

---

## 1) Evaluate the model on `inference.csv` using `inference_labels.csv`

**Status:** Previously (before the mapping fix) your inference evaluation printed:
```
accuracy=0.355, precision=0.112, recall=0.042, f1=0.062,
roc_auc=0.239, pr_auc=0.378
```
After the **mapping fix** (ensure you use `predict_proba` column aligned to `'collision'` and map labels as `{normal:0, collision:1}`), re-run the inference cell to obtain **updated metrics**.  
If the updated inference metrics are still substantially lower than test (see §2), treat this as **true distribution shift** rather than a polarity bug.

> How to compute (already in notebook): merge `inference.csv` with `inference_labels.csv` on an available key, map labels to {0,1}, take **P(collision)** from `predict_proba`, and compute thresholded + threshold-free metrics.

---

## 2) Compare results to performance on `test.csv`

**Your TEST metrics (after mapping fix, threshold=0.5):**
- Accuracy **0.945**
- Precision **1.00**
- Recall **0.89**
- F1 **0.942**
- ROC-AUC **0.928**
- PR-AUC **0.956**

**Comparison (using your earlier INFERENCE numbers pre-fix):**
- Inference is **much worse** than test across all metrics (e.g., PR-AUC ~0.38 vs 0.96).  
- This gap is **too large** to be explained by threshold choice alone → expect **domain shift** between test and inference distributions.

> Action: Re-run inference after the mapping fix. If the gap persists (likely smaller but still material), proceed with the EDA & mitigations below.

---

## 3) EDA — Understand dataset differences (what to check)

Run the advanced notebook’s drift section (or quick checks below) to pinpoint root causes:

- **Feature drift (distribution):** Compute JS divergence / KS p-values per feature between **test** and **inference**. Flag the top drifting features.
- **Representation drift:** PCA (or UMAP) on standardized features: overlay test vs inference; look for shifted clusters.
- **Phone/mount orientation:** If features include axis-specific stats, check whether axis magnitudes/energies rotate or skew across sets.
- **Sampling/windowing:** Confirm identical sampling rate and window length; spectral features can drift with small timing changes.
- **Context mix:** Compare positive rate and key aggregates (speed proxies, vibration energy) across sets.
- **Label alignment noise:** Ensure window anchors for inference labels match the feature windows (off-by-one windowing will crush recall).

**Quick code sketch (already present in the advanced notebook):**
- `jensenshannon` per feature (JS²), `ks_2sample` p-values.
- PCA scatter for test vs inference on common numeric columns.

---

## 4) Immediate workaround (POC-friendly)

- **Cost-aware threshold:** Choose threshold *t* that minimizes `expected_cost = c_FP*FP + c_FN*FN` on validation/test, reflecting the business penalty for missed collisions (usually `c_FN >> c_FP`). Expect recall ↑ with small precision trade-off.
- **Recalibration:** If reliability is off, fit **isotonic** on validation and re-evaluate on inference (Brier ↓, recall more stable).
- **Simple feature hygiene:** Prefer orientation-invariant magnitudes (e.g., `||acc||`) and per-window standardization; remove obviously spurious features.
- **Filter bad segments:** If you detect specific devices/firmware producing outliers, temporarily downweight or filter while you collect fixes.

---

## 5) Long-term fix

- **Data-centric loop:** Tighten label policy (window anchoring), curate a **gold set**, and re-label ambiguous clips.
- **Distribution coverage:** Augment training with mount/orientation variants, sampling jitter, and context regimes that mirror inference.
- **Model/feature robustness:** Use orientation-invariant and spectral features; optionally explore a small sequence model if windows encode temporal patterns.
- **Retraining & monitoring:** Establish retrain cadence and deploy drift monitors (positive rate, mean `y_prob`, Brier, top JS/PSI). Shadow-test new thresholds/models on `X_inf` before switching.
- **Documentation:** Record chosen threshold and business costs; keep calibration plots and drift tables alongside metrics in your monitoring JSON.

---

## 6) Questions to Reflect On — Answers

**Q1. Why might performance drop on the inference set vs test?**  
**A.** Real-world **domain shift**: phone/mount orientation changes, sampling/windowing differences, different driving contexts, and possible label/window misalignment. Confirm with JS/KS drift tables and PCA overlays.

**Q2. What immediate mitigation would you deploy?**  
**A.** **Cost-aware threshold** + **isotonic calibration** on validation; apply simple orientation-invariant feature tweaks. This yields recall gains quickly without retraining.

**Q3. What is your long-term fix?**  
**A.** Clarify label timing policy; expand training to cover the inference distribution; prefer robust features; set retrain triggers on drift; periodically refresh calibration.

**Q4. What would you monitor in prod?**  
**A.** Positive rate, mean `y_prob`, **Brier score**, Recall@Precision=X, top drifting features’ JS/PSI/KS, and calibration curves. Emit snapshots to JSON for dashboards/alerts.
