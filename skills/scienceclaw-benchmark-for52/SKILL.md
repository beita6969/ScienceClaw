---
name: scienceclaw-benchmark-for52
description: "Use when a task gives text transcripts of human behavioural experiments in which every participant response is written as <<KEY>> (Psych-101 / Psych-201 style sessions with a study id and legal response keys) and asks to predict the participant's next choice for each item as a list of keys, judged by micro accuracy. Covers per-participant cognitive models (choice kernels, Q-learning, win-stay lose-shift), population gradient boosting and language-model prompting. Corresponds to FoR52 of the companion ScienceClaw-Eval benchmark."
metadata: { "openclaw": { "emoji": "📊" } }
---

# Next-choice prediction in psychology experiments

## Task
- Inputs: items `{study, history, options[, options_text]}`, where `history` is the session text up to the response to predict (instructions, earlier trials, the participant's own earlier `<<KEY>>` responses and outcomes); optionally training sessions of other participants `{study, text, responses: [{pos, response, options}]}`.
- Deliverable: a list of strings in item order, `y[i]` one of `options[i]`.

## Quality
Micro accuracy, higher is better; a key outside the legal options counts as wrong. Baseline: participant mode, the most frequent earlier response among the legal keys (ties to the most recent, none to the first key), `scilib.psych.mode_prediction`. Accuracy on a few dozen items is noisy; prefer participant-grouped cross-validation.

## Tools
- `scilib.psych.fit_predict(items, train_sessions=None, method="auto", extra_items=None, seed=0, return_proba=False, population_prior=True, llm_answers=None, llm_weight=0.35)`; methods `mode`, `last`, `kernel`, `qlearn` (Q-learning with softmax and perseveration), `wsls`, `personal` (BIC-weighted mix of the three history models), `gbdt` (LightGBM on history features, fitted on training sessions and item histories), `auto` (half personal, half gbdt), `all` (dict of every method's keys). Operator `choice_next_response_predict`.
- `scilib.psych.cross_validate(sessions, method, n_splits=4, per_session=4, seed=0, extra_methods=())`: out-of-fold accuracy against the mode baseline, per study (operator `choice_model_cross_validate`); helpers `make_pseudo_items`, `parse_trials`, `trial_features`.
- Language-model route: `build_prompts(items, style="transcript" | "long" | "model")`, an `llm` node on the canvas (template `{item}`, `parse: text`, small `max_tokens`; code nodes cannot call the model), `parse_answers(replies, items)`, then `fit_predict(..., llm_answers=[answers])`, which mixes the answer shares into the model probabilities. Operators `choice_llm_prompts`, `choice_llm_answers_parse`. The language model reads the instruction text, which the history models do not.

## Routes
Use per-participant `qlearn` (or the best of `qlearn`, `personal`, `wsls` by `cross_validate`) for studies that appear in the training sessions, and the pooled `gbdt` for studies that do not: call `fit_predict(method="all")` and choose per item by whether its `study` id occurs in the training sessions. Switch only on visible study ids, never on target values.

## Rules
- Use only text before the response marker; no later trials, no recorded target.
- Folds must be grouped by participant; never fit on the items being predicted beyond their own earlier history.
- Output legal keys only, in the required order and length.
