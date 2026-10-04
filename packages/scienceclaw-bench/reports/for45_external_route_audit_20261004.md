# FoR45 external route audit (2026-10-04)

This is a supply-chain and contract audit only. It does not change the FoR45 scorer, reference, acceptance margin, split assignment, or any formal episode. No hidden captions were read by a tool and no formal item was rerun.

## Decision

No new eligible frozen route was added. The current CLIP image-kNN route remains diagnostic-only. The strongest public shared-task systems either use the held-out AmericasNLP development captions directly or require an external service/base checkpoint, so installing them would not satisfy the no-label-leakage and self-contained-tool requirements.

## Routes checked

| route | public evidence | contract / provenance | decision |
|---|---|---|---|
| UF Gators | [system paper](https://arxiv.org/abs/2605.20626), [code](https://github.com/dhawan98/AmericasNLP2026-Gators-Submission) | Qwen2.5-VL produces Spanish image caption; Gemini 2.5 Flash performs retrieval-augmented target-language generation. It also uses external parallel/retrieval corpora and a remote Gemini API. No approved local Qwen2.5-VL checkpoint or Gemini credentials are present. | **exclude**: not a self-contained frozen tool |
| Mila Aya Vision adapter | [adapter card](https://huggingface.co/ludolara/aya-vision-32b-americas-captioning) | LoRA adapter requires gated `CohereLabs/aya-vision-32b`; card explicitly says SFT on 50 AmericasNLP 2026 development captions per language. The local FoR45 ID/OOD pools are rebuilt from that same 50-row-per-language public development partition, so the adapter has direct target-caption overlap. License is CC-BY-NC-4.0. | **exclude**: label leakage and missing base |
| Aya GRPO adapter | [collection entry](https://huggingface.co/ludolara/aya-vision-32b-americas-grpo-captioning) | Same gated Aya Vision base and AmericasNLP-specific training family; no evidence of an independent caption universe. | **exclude**: same overlap/base/license boundary |
| Florence-2 | [official model card](https://huggingface.co/microsoft/Florence-2-large) | General image-to-text model (MIT code/weights), but it does not provide a target-language Indigenous captioning head or the required Bribri/Guarani/Nahuatl/Wixárika translation data. | **not contract-complete** |
| SigLIP / SigLIP2 | [Transformers documentation](https://huggingface.co/docs/transformers/model_doc/siglip) | Image/text encoders for similarity and retrieval; they do not generate captions. A retrieval adapter would still need a target-language generator and would not repair the missing low-resource translation route. | **not contract-complete** |
| USP Qwen3-VL + NLLB cascade | [public code and system README](https://github.com/rmaacario/americasnlp2026-usp) | The repository publishes notebooks and the recipe, but the fine-tuned NLLB checkpoints are saved to temporary Kaggle/Colab paths and are not released as frozen weights. It also requires a Qwen3-VL-8B visual stage that is absent from the local tool pool. The README reports base NLLB Guaraní dev chrF++ 19.49 versus 17.57 after its fine-tune, and notes that Bribri/Maya target tokens are missing from base NLLB. Reproducing the route here would require training new weights, which is outside the current no-training boundary. | **exclude**: no deployable frozen checkpoint |
| OPD-V-Qwen3.5-4B already on Leonardo | [model card metadata](https://huggingface.co/aniri15/OPD-V-Qwen3.5-4B) | Local checkpoint: `$L/models/OPD-V-Qwen3.5-4B/model.safetensors`, 9.7G directory; model SHA-256 `8386abfab21eff23e78e572b3f13f5f249ae9f75ff54b804772b03f4254ccfdd`. It is a general visual-detail self-distillation checkpoint, not trained for AmericasNLP or Indigenous-language captioning. A CPU generation smoke was killed by the login-node memory limit; no score or label access resulted. | **diagnostic candidate only**, no integration |

The local server does contain the approved OpenAI CLIP ViT-B/32 checkpoint used by the existing diagnostic route (`$L/models/clip/open_clip_vit_b32/ViT-B-32.pt`, SHA recorded in the handoff). Its full-pool image-kNN score is below the visible language-medoid reference on every pool, so it is not an improvement candidate.

## Why the public Aya result cannot be used as a score

The adapter card reports caption-stage SFT on 50 development examples per language and reports test chrF++ results. FoR45 reconstructs its labelled population from the same AmericasNLP 2026 development partition, then draws the local ID and proxy-OOD pools from those rows. Loading that adapter would therefore import target captions from the evaluated universe. The base model is also gated and absent from the Leonardo model root. This is a provenance exclusion, not a model-quality claim.

## Reproducibility boundary

Only public pages and remote file metadata were inspected. The remote probe did not read evaluation captions, did not invoke `score_dev`, did not start a broker route, and did not consume an ID/OOD item. Existing CLIP code and prior engineering values are unchanged. A future route is eligible only if it has (1) an independently sourced target-language generator, (2) a complete local checkpoint and pinned dependencies, and (3) provenance proving no overlap with the 50-row-per-language AmericasNLP development universe.

## Official metric and wiring cross-check

The pinned upstream `baseline/eval.py` constructs `sacrebleu.metrics.CHRF(word_order=2)`, strips each generated line, scores one caption against its target, and averages the per-row scores. The local `chrf_pp` helper uses the same scorer and the evaluator strips predictions before scoring; the focused FoR45 tests cover this equivalence. The CLIP route also remains fail-closed: its ISO-keyed visible-caption fallback, fixed top-three medoid alias, explicit checkpoint digest, and offline loading guard are all covered by focused tests. No scorer, split, or tool-to-`submit.y` wiring defect was found in this pass.
