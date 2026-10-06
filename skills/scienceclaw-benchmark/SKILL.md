---
name: scienceclaw-benchmark
description: "Index of the 23 discipline skills (FoR30-FoR52) that mirror the disciplines of the companion ScienceClaw-Eval benchmark: find which discipline skill fits a scientific task (plant panoptic segmentation, protein fitness prediction, hippocampus segmentation, building load forecasting, molecule property prediction, music source separation, sepsis early warning, French dependency parsing, contract evidence retrieval, ...). Use to pick the right discipline skill and the scilib tools it names."
metadata: { "openclaw": { "emoji": "🧪" } }
---

# ScienceClaw discipline skills

Each `scienceclaw-benchmark-forNN` skill describes one scientific task family: its inputs, the deliverable, how
quality is judged, and the scilib tools and pretrained weights that fit it. The 23 skills correspond to the 23
disciplines of the companion ScienceClaw-Eval benchmark; its evaluation data is hosted on Hugging Face
(<https://huggingface.co/datasets/beita6969/scienceclaw-eval>).

## How to use

1. Match the user's task to one row below and load that discipline skill.
2. Confirm the tools it names with `scienceclaw_tools` (`search`, `show`, `status`, `weights`) and do not plan around a
   tool or weight that is reported as unavailable on this host.
3. Build the workflow on the canvas (see `scienceclaw-canvas`): read the inputs through the task's `load_<name>` tools,
   keep the deliverable's type, shape and unit, and replay before finishing.

## Disciplines

| Code | Discipline (ANZSRC division) | Task | Quality metric | Skill |
| --- | --- | --- | --- | --- |
| FoR30 | Agricultural, veterinary and food sciences | PhenoBench hierarchical panoptic segmentation | PQ+ (higher is better) | `scienceclaw-benchmark-for30` |
| FoR31 | Biological sciences | ProteinGym substitutions | mean Spearman (higher is better) | `scienceclaw-benchmark-for31` |
| FoR32 | Biomedical and clinical sciences | MSD Task04 Hippocampus | DSC (higher is better) | `scienceclaw-benchmark-for32` |
| FoR33 | Built environment and design | BuildingsBench | balanced NRMSE (%) (lower is better) | `scienceclaw-benchmark-for33` |
| FoR34 | Chemical sciences | OGB ogbg-molhiv | ROC-AUC (higher is better) | `scienceclaw-benchmark-for34` |
| FoR35 | Commerce, management, tourism and services | Monash Tourism Monthly | mean MASE (lower is better) | `scienceclaw-benchmark-for35` |
| FoR36 | Creative arts and writing | MUSDB18 four-stem separation | mean target-median SDR (dB) (higher is better) | `scienceclaw-benchmark-for36` |
| FoR37 | Earth sciences | WeatherBench2 | 2m temperature RMSE (K) (lower is better) | `scienceclaw-benchmark-for37` |
| FoR38 | Economics | World Bank macro forecasting | mean sMAPE (%) (lower is better) | `scienceclaw-benchmark-for38` |
| FoR39 | Education | Eedi NeurIPS 2020 Task 4 | organizer 10-mask accuracy (higher is better) | `scienceclaw-benchmark-for39` |
| FoR40 | Engineering | DCASE 2024 Task 2 | official DCASE score (higher is better) | `scienceclaw-benchmark-for40` |
| FoR41 | Environmental sciences | NEON aquatics forecasting | mean CRPS (oxygen + temperature) (lower is better) | `scienceclaw-benchmark-for41` |
| FoR42 | Health sciences | PhysioNet/CinC 2019 sepsis | normalized clinical utility (higher is better) | `scienceclaw-benchmark-for42` |
| FoR43 | History, heritage and archaeology | HIPE-OCRepair 2026 | weighted cMER-micro (lower is better) | `scienceclaw-benchmark-for43` |
| FoR44 | Human society | ACIC 2016 | response-SD normalized RMSE (lower is better) | `scienceclaw-benchmark-for44` |
| FoR45 | Indigenous studies | AmericasNLP 2026 | mean sentence chrF++ (higher is better) | `scienceclaw-benchmark-for45` |
| FoR46 | Information and computing sciences | HumanEval → MBPP | execution pass@1 (higher is better) | `scienceclaw-benchmark-for46` |
| FoR47 | Language, communication and culture | CoNLL-2018 UD | LAS (higher is better) | `scienceclaw-benchmark-for47` |
| FoR48 | Law and legal studies | ContractNLI | mAP (higher is better) | `scienceclaw-benchmark-for48` |
| FoR49 | Mathematical sciences | SMT-COMP 2025 QF_NIA | oracle-agreement accuracy (higher is better) | `scienceclaw-benchmark-for49` |
| FoR50 | Philosophy and religious studies | SemEval-2023 Task 4 ValueEval | official F1 (higher is better) | `scienceclaw-benchmark-for50` |
| FoR51 | Physical sciences | Matbench phonons | MAE (lower is better) | `scienceclaw-benchmark-for51` |
| FoR52 | Psychology | Psych-201 discrete | micro accuracy (higher is better) | `scienceclaw-benchmark-for52` |

## Rules that apply to every discipline

- Use only the inputs the task declares; never use target information, and respect the temporal order of forecasting
  and clinical data.
- Prefer a frozen checkpoint whose input/output contract is documented over a model trained inside the workflow, and
  report which checkpoint (and version) produced a result. Pretrained weights may have seen similar data: say so.
- Keep a simple baseline next to a pretrained route so that the gain, if any, is visible.
