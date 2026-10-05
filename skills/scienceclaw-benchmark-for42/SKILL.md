---
name: scienceclaw-benchmark-for42
description: "Use when a task gives hourly ICU records (vitals, labs, demographics, one stay per patient) with sepsis labels and asks for causal hour-by-hour sepsis alarms on new stays, scored by normalized clinical utility (PhysioNet/CinC 2019 early sepsis prediction; corresponds to FoR42 of the companion ScienceClaw-Eval benchmark)."
metadata: { "openclaw": { "emoji": "📊" } }
---

# Early sepsis alarms from hourly ICU data

**Task.** Input: a long table, one row per patient-hour (`patient_id`, `hour`, the 40 challenge variables: vitals, labs, Age, Gender, Unit1/Unit2, HospAdmTime, ICULOS); the labelled table also has `SepsisLabel`. Deliverable: one 0/1 integer array per stay to score, as long as the stay, in order of first appearance.

**Quality.** Normalized clinical utility, higher is better: `(U_obs - U_inaction) / (U_best - U_inaction)` summed over stays; 1 is optimal, 0 equals never alarming, negative is worse than silence. Per hour: a true alarm earns up to +1 from 12 h before to 3 h after t_sepsis (first positive label + 6 h), a miss costs up to -2, a false alarm -0.05.

**Tools (`from scilib import sepsis`, no pretrained weights).**
- `fit_predict(train, [table_a, table_b])` returns `(preds, info)` with the chosen threshold and out-of-fold utility (`oof_utility`); operator `sepsis_early_warning_alarms`.
- `SepsisModel(targets, smooth, hold, prevalence, linear_weight)`: shallow LightGBM plus logistic regression on the causal features of `build_features`; tune on grouped out-of-fold utility.
- `normalized_utility(labels, preds)` (operator `sepsis_normalized_utility`), `stay_folds`, `causal_smooth`, `hold_alarms`.

**Routes.** Baseline: class-balanced logistic regression on last-observation-carried-forward vitals plus static variables, threshold chosen for utility. Stronger: the boosted ensemble, compared by patient-grouped out-of-fold utility, not row-level AUC.

**Rules.**
- Strict causality: the alarm at hour t uses rows 0..t of that stay only. Never use stay length, hours to the end of the record or whole-stay statistics: septic records end soon after onset, so "alarm near the end" scores well offline and is useless in use. Check by truncating stays and re-running; earlier hours must not change.
- Split by patient, never by row. Septic stays are rare (about 7 % in the public data), so utility is noisy; report its spread.
- The best threshold depends on the septic share (false alarms cost per hour); re-tune for a different hospital. Variables can be wholly missing at one site (EtCO2 in hospital A).
- Keep the declared output: finite 0/1 integers, per-stay lengths.
