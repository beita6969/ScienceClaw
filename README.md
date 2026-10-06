<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/banner-dark.svg">
  <img src="assets/banner.svg" alt="ScienceClaw" width="640">
</picture>

<p><b>Skills and Operators that grow from replay-verified executions.</b></p>

<p>
  <a href="https://scienceclaw.science"><img src="https://img.shields.io/badge/website-scienceclaw.science-0891b2?style=flat-square&logo=googlechrome&logoColor=white" alt="Website"></a>
  <a href="https://huggingface.co/datasets/beita6969/scienceclaw-eval"><img src="https://img.shields.io/badge/benchmark%20data-Hugging%20Face-f59e0b?style=flat-square&logo=huggingface&logoColor=white" alt="Benchmark data on Hugging Face"></a>
  <a href="https://github.com/beita6969/ScienceClaw/stargazers"><img src="https://img.shields.io/github/stars/beita6969/ScienceClaw?style=flat-square&logo=github&color=4f46e5" alt="GitHub stars"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/beita6969/ScienceClaw?style=flat-square&color=64748b" alt="License"></a>
</p>

<p>
  <a href="#quick-start"><b>Quick start</b></a> &nbsp;·&nbsp;
  <a href="#how-it-works">How it works</a> &nbsp;·&nbsp;
  <a href="#results">Results</a> &nbsp;·&nbsp;
  <a href="#use-it-from-the-gateway">Gateway</a> &nbsp;·&nbsp;
  <a href="#scienceclaw-eval">Benchmark</a> &nbsp;·&nbsp;
  <a href="packages/scienceclaw/docs/DESIGN.md">Design</a>
</p>

<img src="assets/paper/overview.png" alt="Overview of ScienceClaw" width="100%">
<p><sub><i>ScienceClaw-Eval spans 23 disciplines across the natural and social sciences, while ScienceClaw turns verified execution evidence into persistent Skill–Operator program updates.</i></sub></p>

<table>
  <tr>
    <td align="center" width="25%"><h3>23</h3><sub>disciplines across the<br>natural and social sciences</sub></td>
    <td align="center" width="25%"><h3>+16.45 %</h3><sub>mean out-of-distribution gain<br>over the frozen agent</sub></td>
    <td align="center" width="25%"><h3>98.23 %</h3><sub>of instances pass every<br>scientific hard constraint</sub></td>
    <td align="center" width="25%"><h3>91.30</h3><sub>OOD macro success rate<br>after seven rounds (from 77.72)</sub></td>
  </tr>
</table>

</div>

ScienceClaw is an agent system for scientific work that **gets better the more it is used**. It solves each task as a typed, executable workflow. When a repair is reproduced under a clean replay, it becomes a linked **Skill** (strategy) and **Operator** (a typed, executable capability), and it is kept only if it still solves its source task and improves independent validation tasks.

> [!NOTE]
> This repository is the **agent system**, built on the [OpenClaw](https://github.com/openclaw/openclaw) gateway. Its companion benchmark, **ScienceClaw-Eval**, keeps its evaluation data on [Hugging Face](https://huggingface.co/datasets/beita6969/scienceclaw-eval).

## ✨ Highlights

| Idea | In practice |
| --- | --- |
| **An evolving program** | What evolves is a versioned program of **Skills** (decomposition, workflow construction, recovery) and typed **Operators** with explicit input, output and domain contracts. |
| **Typed, executable workflows** | Ports carry a schema of type, shape, unit and provenance. Nodes are fingerprinted, so an edit reruns only what it affects. |
| **Evidence you can replay** | Every result is regenerated from a reset environment and checked against the task's hard scientific constraints. |
| **Linked Skill–Operator updates** | A reproduced repair becomes a strategy patch and a typed capability, committed together. No LLM judge is needed to form, rank or select candidates. |
| **A gate, not a guess** | An update persists only if source replay reproduces it and uses it, and independent validation tasks strictly improve within budget. |
| **You stay in control** | Updates are versioned snapshots that you promote, and that you can roll back. |

<a id="quick-start"></a>

## 🚀 Quick start

```bash
git clone https://github.com/beita6969/ScienceClaw.git
cd ScienceClaw

# One-click setup: Node, Python, the engine, its tools and weights, MCP servers, skills
chmod +x setup.sh && ./setup.sh
```

<details>
<summary>Manual installation</summary>

```bash
pnpm install && npx openclaw onboard
pip install -e packages/scienceclaw          # Python >= 3.11; extras: [all], vision, audio, nlp, forecast, materials, retrieval
python -m scienceclaw.cli setup              # install and verify every tool and pretrained weight (--profile light skips assets > 1.5 GB)
python -m scienceclaw.cli doctor             # which tool modules can run on this machine, and why not
```

The engine refuses tasks until `setup` has completed (`SCIENCECLAW_SKIP_SETUP_CHECK=1` is for development only), and the full profile downloads about 19 GB of weights. `setup.sh` runs it for you; `SCIENCECLAW_SKIP_TOOLS=1` defers it to first use.

</details>

> [!TIP]
> **Bring your own model.** The engine never assumes a model family and this repository names none. Point it at any OpenAI-compatible endpoint, a local program, or your own backend:
>
> ```bash
> export SCIENCECLAW_API_BASE_URL=<your OpenAI-compatible endpoint>
> export SCIENCECLAW_API_KEY=<your key>
> export SCIENCECLAW_MODEL=<your model name>      # or SCIENCECLAW_<POLICY|EXECUTOR|PATCH>_MODEL per role
> ```
>
> `llm.backend` can also be `command` (a local program reads the prompt and writes the completion) or `package.module:factory`. See [`configs/default.yaml`](packages/scienceclaw/configs/default.yaml).

<a id="how-it-works"></a>

## 🧠 How it works

### The problem

LLM agents increasingly solve scientific tasks by connecting reasoning to data, domain tools and executable code. But a repair that works once rarely survives: it lives in a transient context, or in a single tool or prompt, so verified executions seldom become persistent improvements. Existing work evolves individual tools or Skills, and existing benchmarks treat tasks as independent episodes, so it has been hard to tell whether verified scientific executions turn into *persistent, transferable* improvements.

> **How can verified scientific executions drive persistent and transferable program-level self-evolution without updating foundation-model parameters?**

<p align="center"><img src="assets/paper/task_formulation.png" alt="Task formulation" width="85%"></p>
<p align="center"><sub><i>Task formulation of verifiable program-level self-evolution for AI-for-Science agents.</i></sub></p>

### The answer: replay-gated, linked Skill–Operator evolution

**1. A task for program-level self-evolution.** The model Θ₀ is fixed. What evolves is an editable program `A_r = (Skills, Operators)`. A task `D_t = (D_T, D_V, D_E)` specifies its objective and inputs, its evaluation protocol and hard constraints (units, feasibility, convergence, reproducibility), and its data, tools and environment.

**2. Typed, execution-guided workflows.** A solution is a directed graph whose ports carry a schema `(type, shape, unit, provenance)`; an edge is valid only if types and shapes match and any unit conversion is explicit and recorded. The agent edits the graph one atomic action at a time, and the executor checkpoints every node by fingerprint and reruns only the affected descendants, so a long scientific workflow is repaired rather than regenerated.

**3. Evidence you can replay.** Every candidate result is regenerated from a *reset* environment and checked against the task's evaluator and hard constraints. A replay that fails and a later replay that passes bracket the shortest reproduced repair, `e⁻ → e⁺`. Replay alone never authorizes persistence.

**4. Linked Skill–Operator candidates, with no LLM judge.** The reproduced repair is split by edit type. Control edits (topology, routing, configuration) become a **Skill patch**. Generated or repaired executable nodes are grouped into convex components and abstracted into typed **Operators**, each checked by *boundary replay* in isolation. Both are committed as one atomic bundle, so a strategy never arrives without the capability behind it, nor a capability without a strategy that selects it.

**5. A replay-and-validation gate.** A candidate persists only if source replay reproduces the repair and actually *uses* every new component (`R_src = Pass ∧ Use`), every hard integrity and scientific check holds on independent validation tasks, its cost stays within budget, and it **strictly improves** the validation score over the incumbent. A noise guard requires at least two improved and at most one regressed validation episode. Otherwise the incumbent program is kept.

<p align="center"><img src="assets/paper/method_overview.png" alt="Definition and overview of the ScienceClaw task"></p>
<p align="center"><sub><i>Given task specification D_t and agent program A_r, ScienceClaw produces scientific solution Z_t and retains a candidate update only after source-task replay and independent program validation.</i></sub></p>

<a id="results"></a>

## 📊 Results

Seven evolution rounds over a common stream of 23 disciplines, with 64 IID and 64 OOD instances per discipline, one fixed foundation model, and the same tools, source stream, validation data and update budget throughout. OOD means an independently sourced dataset of the same discipline; OOD results never generate or select updates.

- **Consistent gains.** ScienceClaw improves on the frozen agent in every discipline: **+16.45 %** on average out-of-distribution (+11.57 % to +23.34 %) and +12.25 % in-distribution. Its OOD score falls 11.73 % below its IID score on average, against 16.06 % for the frozen agent.
- **It keeps improving.** The OOD macro success rate rises from **77.72 to 91.30** over seven rounds (+13.59 pp). 70 of 161 submitted candidates are promoted (43.48 %), so the gate is selective without stalling.
- **It transfers and retains.** 18 of 20 cross-family pairs transfer positively (mean +1.58 pp, against +14.69 pp within a family). Average forgetting is 0.13 pp (maximum 0.51 pp), with 11.89 % negative transfer.
- **It is reliable.** Hard constraints pass on **98.23 %** of instances and only 1.23 % of promotions are erroneous.
- **The linkage and the gate are what matter.** Committing Skills and Operators separately keeps only 60 % of the gain; removing Operator evolution keeps 59 %. Without independent IID selection only 15 % remains; without scientific constraints, source replay or the independent validator, 46 %, 55 % and 59 %.
- **Execution structure is the base.** The full system runs 4.84 planner rounds and 4.24 distinct Operators per task, repairs 71 % of failures from feedback, recovers 89 % after interruption and replays 96 % cleanly (a single-turn agent: 0 %, 31 %, 78 %).

<p align="center"><img src="assets/paper/ablation_heatmap.png" alt="Ablation heatmap across 23 disciplines"></p>
<p align="center"><sub><i>Mean of IID and OOD task-native scores of the ablation variants across 23 disciplines. Colours are normalised within each discipline (darker is better); the two groups follow higher-is-better and lower-is-better metrics.</i></sub></p>

<p align="center"><img src="assets/paper/mechanisms.png" alt="Linked mechanism, workflow mechanics, reliability and cost, and promoted candidates" width="100%"></p>
<p align="center"><sub><i>(f) Gain over the frozen agent (%) of the linked-mechanism variants on Commerce (MASE) and Law (mAP). (g) Workflow mechanics: planner rounds, distinct Operators, Operator diversity, feedback repair, checkpoint recovery and clean replay. (h) Hard-constraint pass rate against cost per OOD gain (ScienceClaw = 1), coloured by planner wall-time share. (i) Promoted candidates and promotion rate.</i></sub></p>

<details>
<summary>Per-discipline scores: ScienceClaw against the frozen agent</summary>

Task-native scores, compared only within a discipline (metrics differ in units and direction); arrows give the direction of "better". Gains are the relative improvement in the better direction, computed from the rounded scores shown.

| FoR | Discipline (metric) | Frozen IID | ScienceClaw IID | Gain | Frozen OOD | ScienceClaw OOD | Gain |
|---|---|---:|---:|---:|---:|---:|---:|
| 30 | Agricultural sci. (PQ+) ↑ | 69.4761 | **76.9612** | +10.77 % | 64.6829 | **75.7629** | +17.13 % |
| 31 | Biological sci. (Spearman) ↑ | 0.5057 | **0.5816** | +15.01 % | 0.3745 | **0.4234** | +13.06 % |
| 32 | Biomedical sci. (DSC) ↑ | 0.7816 | **0.8556** | +9.47 % | 0.7088 | **0.8226** | +16.06 % |
| 33 | Built env. (NRMSE %) ↓ | 51.0997 | **45.0065** | +11.92 % | 53.5009 | **45.0970** | +15.71 % |
| 34 | Chemical sci. (ROC-AUC) ↑ | 0.6518 | **0.7545** | +15.76 % | 0.6205 | **0.7478** | +20.52 % |
| 35 | Commerce (MASE) ↓ | 1.6009 | **1.4653** | +8.47 % | 1.8740 | **1.5986** | +14.70 % |
| 36 | Creative arts (SDR (dB)) ↑ | 8.9208 | **10.1867** | +14.19 % | 8.0993 | **9.5619** | +18.06 % |
| 37 | Earth sci. (RMSE (K)) ↓ | 1.0562 | **0.9721** | +7.96 % | 1.1364 | **0.9580** | +15.70 % |
| 38 | Economics (sMAPE (%)) ↓ | 12.7394 | **11.3320** | +11.05 % | 18.1343 | **15.5025** | +14.51 % |
| 39 | Education (10-mask acc.) ↑ | 0.5965 | **0.6878** | +15.31 % | 0.6202 | **0.7087** | +14.27 % |
| 40 | Engineering (DCASE score) ↑ | 0.5709 | **0.6535** | +14.47 % | 0.4663 | **0.5269** | +13.00 % |
| 41 | Environmental sci. (CRPS) ↓ | 0.8104 | **0.7164** | +11.60 % | 0.6504 | **0.5751** | +11.58 % |
| 42 | Health sci. (Clin. utility) ↑ | 0.4971 | **0.5514** | +10.92 % | 0.6834 | **0.7796** | +14.08 % |
| 43 | History (cMER-micro) ↓ | 0.0185 | **0.0166** | +10.27 % | 0.0296 | **0.0251** | +15.20 % |
| 44 | Human society (nRMSE) ↓ | 0.0113 | **0.0100** | +11.50 % | 0.0213 | **0.0181** | +15.02 % |
| 45 | Indigenous (chrF++) ↑ | 15.4451 | **17.4351** | +12.88 % | 13.7205 | **15.9778** | +16.45 % |
| 46 | Computing sci. (pass@1) ↑ | 0.7656 | **0.8750** | +14.29 % | 0.3750 | **0.4531** | +20.83 % |
| 47 | Language & culture (LAS) ↑ | 0.7288 | **0.8138** | +11.66 % | 0.5599 | **0.6497** | +16.04 % |
| 48 | Law (mAP) ↑ | 0.7513 | **0.8554** | +13.86 % | 0.6918 | **0.8289** | +19.82 % |
| 49 | Mathematics (Oracle acc.) ↑ | 0.8438 | **0.9531** | +12.95 % | 0.7344 | **0.8750** | +19.14 % |
| 50 | Philosophy (F1) ↑ | 0.4178 | **0.4866** | +16.47 % | 0.4166 | **0.5138** | +23.33 % |
| 51 | Physical sci. (MAE) ↓ | 36.6271 | **32.1581** | +12.20 % | 44.6757 | **37.1306** | +16.89 % |
| 52 | Psychology (Micro acc.) ↑ | 0.6088 | **0.6618** | +8.71 % | 0.5615 | **0.6586** | +17.29 % |
</details>

<details>
<summary>Continual evolution: OOD macro success rate by round</summary>

OOD macro success rate (%) of each snapshot after round *r*, with the gain over the initial program and the candidates promoted and rejected out of 161. RuleEvo is a ScienceClaw variant whose evolution uses deterministic trace projection alone, with no generative evolution roles.

| Method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | Gain (pp) | Promoted | Rejected | Rate (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen | 77.72 | 77.72 | 77.72 | 77.72 | 77.72 | 77.72 | 77.72 | 77.72 | 0.00 | 0 | 0 | – |
| RuleEvo | 77.72 | 78.40 | 79.21 | 78.87 | 79.62 | 80.37 | 80.03 | 81.32 | 3.60 | 37 | 124 | 22.98 |
| **ScienceClaw** | **77.72** | **80.43** | **82.68** | **82.47** | **86.14** | **87.57** | **89.81** | **91.30** | **13.59** | **70** | **91** | **43.48** |
</details>

Reported trajectories are final snapshots, not uncertainty estimates over source orders or model configurations, and cost comparisons are relative to the protocol, not absolute. The paper has the full tables, ablations and the limitations discussion.

<a id="use-it-from-the-gateway"></a>

## 🔌 Use it from the gateway

Enable the plugin in `~/.openclaw/openclaw.json` (see [`extensions/scienceclaw/README.md`](extensions/scienceclaw/README.md)). It registers four agent tools:

| Tool | Operations |
| --- | --- |
| `scienceclaw_canvas` | `open`, `act`, `render`, `replay`, `finish`, `status`, `list` |
| `scienceclaw_tools` | `search`, `show`, `status`, `weights`, `setup` |
| `scienceclaw_program` | `summary`, `skills`, `operators`, `show`, `history`, `rollback` |
| `scienceclaw_evolve` | `val_add`, `val_list`, `val_remove`, `propose`, `gate`, `run`, `status`, `candidates`, `show` |

**A session, end to end**

1. **Declare the task.** `scienceclaw_canvas(operation=open, task={objective, inputs, required_output, constraints})`. Each input becomes a read-only `load_<name>` tool restricted to the configured input roots; constraints (`finite`, `shape`, `type`, `range`, `nonempty`, `len_eq_input`, or an evaluator-only `metric` bar) are the acceptance test.
2. **Build the workflow.** The agent finds tools with `scienceclaw_tools`, then adds, modifies or removes one node or edge per `act`, reading the typed feedback after each edit.
3. **Verify.** `replay` re-executes the whole graph from a reset state; `finish` returns the verified deliverable.
4. **Evolve (optional).** Register independent validation tasks with `val_add` (the default noise guard needs at least two), then `propose` candidates from a finished, verified session and `gate` them. A session that contained a repair yields a linked Skill and Operator; one without a failed replay yields an Operator only. A candidate that passes becomes `ready`.
5. **You decide.** Review and promote with the CLI; every promotion is a new program version that can be rolled back.

```bash
python -m scienceclaw.cli live candidates
python -m scienceclaw.cli live show <candidate>
python -m scienceclaw.cli live promote <candidate>
python -m scienceclaw.cli live rollback <version>
```

The same engine speaks line-delimited JSON on `python -m scienceclaw.rpc` if you want to drive it from your own host.

## 🧰 Skills and tools

| Component | What it provides |
| --- | --- |
| **300+ skills** | [`skills/`](skills) holds the skill library. The 36 `scienceclaw-*` skills form the seed program: 12 general ones (canvas orchestration, evolution, and task patterns such as retrieval, prediction and verification) and the discipline skills below. Evolved Skills are written back as versioned records of the same program. |
| **Discipline skills** | `scienceclaw-benchmark-for30` … `for52`, indexed by `scienceclaw-benchmark`, describe the task family of each benchmark discipline: inputs, deliverable, how quality is judged, and the tools and weights that fit. They are retrieved like any other skill when a task matches. |
| **`scilib`** | Classical toolkits and frozen pretrained models for science: forecasting, segmentation, source separation, protein and molecular models, causal inference, parsing, formal solvers and more. Heavy models run on a GPU-tool broker when they cannot run locally. `scienceclaw_tools` and `python -m scienceclaw.cli tools` search, describe and probe them. |
| **Research protocol** | [`SCIENCE.md`](SCIENCE.md) governs literature work in the gateway: every citation must come from a tool result in the current conversation, searches cross several sources, and results are written to a file before an answer is final. |

## 🛡️ Safety and governance

> [!WARNING]
> Persistent executable updates can reuse errors and widen the attack surface. Code nodes run in a sandbox (separate process, scrubbed environment, static scan, runtime audit guard, and user/network/pid namespaces where the host supports them), but run untrusted workloads in a container as well.

- **Nothing persists without evidence.** Source replay, validation, budget and strict improvement all have to hold, and the gate fails closed when the model is unavailable.
- **Updates are your decision, and reversible.** Candidates wait as `ready` until promoted; every version is a snapshot with a receipt and can be rolled back.
- **Provenance everywhere.** Ports record units and upstream transformations; operators record their source episode, steps and parent version.

ScienceClaw should support, not replace, experts. High-stakes use needs provenance, licensing and privacy safeguards, and independent review.

<a id="scienceclaw-eval"></a>

## 🧪 ScienceClaw-Eval

ScienceClaw-Eval benchmarks continual self-evolution rather than single-shot ability. Systems share the foundation model, the initial program, the source order, the tools and the budget, and are compared on:

- a **source stream** that supplies the only evolution evidence,
- an independent **validation set** used for candidate selection,
- held-out **ID** and same-discipline cross-dataset **OOD** sets,
- a **replay** set of earlier source tasks that measures retention.

An instance counts as solved only if execution completes within budget, the task-native metric meets its acceptance rule, and every scientific hard constraint holds; success is macro-averaged over disciplines. Each task has an isolated environment and a task-specific evaluator, reviewed by domain experts and verified by reset replay.

<p align="center"><img src="assets/paper/benchmark_construction.png" alt="Construction of ScienceClaw-Eval"></p>
<p align="center"><sub><i>Construction of ScienceClaw-Eval: scientific-task collection, executable instantiation, validation and reproduction, and lineage-aware evaluation splits.</i></sub></p>

The evaluation data (64 IID and 64 OOD records for each discipline) is on Hugging Face: **[`beita6969/scienceclaw-eval`](https://huggingface.co/datasets/beita6969/scienceclaw-eval)**.

<details open>
<summary>The 23 disciplines, their tasks and metrics</summary>

| Code | Discipline (ANZSRC division) | Task | Metric | IID source | OOD source |
| --- | --- | --- | --- | --- | --- |
| FoR30 | Agricultural, veterinary and food sciences | plant and leaf panoptic segmentation | PQ+ ↑ | PhenoBench | CropAndWeedAndLeaf |
| FoR31 | Biological sciences | protein variant fitness ranking | Spearman ↑ | ProteinGym | TAPE fluorescence |
| FoR32 | Biomedical and clinical sciences | hippocampus segmentation in MRI | DSC ↑ | MSD Task04 | UCL/Dryad hippocampus |
| FoR33 | Built environment and design | building load forecasting | NRMSE (%) ↓ | BuildingsBench | EULP |
| FoR34 | Chemical sciences | molecular activity classification | ROC-AUC ↑ | OGB ogbg-molhiv | BACE |
| FoR35 | Commerce, management, tourism and services | tourism series forecasting | MASE ↓ | Monash Tourism Monthly | Monash Tourism Quarterly |
| FoR36 | Creative arts and writing | music source separation | SDR (dB) ↑ | MUSDB18 | MoisesDB |
| FoR37 | Earth sciences | 2 m temperature forecasting | RMSE (K) ↓ | WeatherBench 2 (ERA5, 2019) | WeatherBench 2 (ERA5, 2020) |
| FoR38 | Economics | macroeconomic forecasting | sMAPE (%) ↓ | World Bank WDI | World Bank WDI |
| FoR39 | Education | adaptive educational testing | 10-mask accuracy ↑ | Eedi Task 4 | EdNet-KT1 |
| FoR40 | Engineering | anomalous sound detection | DCASE score ↑ | DCASE 2024 Task 2 | DCASE 2023 Task 2 ToyNscale |
| FoR41 | Environmental sciences | probabilistic aquatic forecasting | CRPS ↓ | NEON aquatics | USGS river metabolism |
| FoR42 | Health sciences | sepsis early warning | clinical utility ↑ | PhysioNet/CinC 2019 | SepsisExp |
| FoR43 | History, heritage and archaeology | OCR post-correction | cMER-micro ↓ | HIPE-OCRepair-2026 | ICDAR 2019 POCR |
| FoR44 | Human society | causal treatment-effect estimation | nRMSE ↓ | ACIC 2016 | IHDP |
| FoR45 | Indigenous studies | Indigenous-language captioning | chrF++ ↑ | AmericasNLP 2026 | Bloom Captioning (Mam) |
| FoR46 | Information and computing sciences | code generation | pass@1 ↑ | HumanEval | SWE-bench Verified |
| FoR47 | Language, communication and culture | dependency parsing | LAS ↑ | UD: Marathi, Italian, Arabic, Croatian | UD: Indonesian, Swedish Sign Language, French, Hindi |
| FoR48 | Law and legal studies | contract evidence retrieval | mAP ↑ | ContractNLI | ACORD |
| FoR49 | Mathematical sciences | SMT satisfiability prediction | oracle-agreement accuracy ↑ | SMT-LIB 2025 | SMT-LIB 2024 |
| FoR50 | Philosophy and religious studies | human-value detection | F1 ↑ | Touché23-ValueEval | ETHICS |
| FoR51 | Physical sciences | phonon property prediction | MAE ↓ | Matbench phonons | Kyoto PhononDB |
| FoR52 | Psychology | human choice prediction | micro accuracy ↑ | Psych-201 | Psych-101 |

Every row of the dataset records its own source, license and URL; the dataset card on Hugging Face has the details.
</details>

## 🗂️ Repository

<details>
<summary>Module map: every paper object and where it lives in the code</summary>

| Paper | Code (`packages/scienceclaw/`) |
| --- | --- |
| Task `D_t = (D_T, D_V, D_E)` | `scienceclaw/task.py` (`Episode`); a live task declaration becomes one in `canvas/live.py` |
| Program `A_r = (Skills, Operators)` | `core/program.py`, versioned by `program/store.py`; seed Skills from `skills/scienceclaw-*` (`program/seed.py`); 112 typed library Operators in `program/specs/` |
| Typed workflow graph, `Compat` | `core/schema.py`, `core/graph.py`, `core/actions.py` |
| Execution-guided orchestration | `agent/` (policy, prompts, solver) and `canvas/session.py`, where the gateway agent is the policy |
| Checkpointed execution, reset replay | `runtime/executor.py`, `runtime/replay.py` |
| Sandbox and integrity | `runtime/sandbox.py`, `runtime/integrity.py`, `runtime/node_worker.py` |
| Retrieval of Skills and Operators | `core/retrieval.py` (BM25 plus metadata match) |
| Repair attribution, edit split | `evolution/attribution.py`, `evolution/split.py` |
| Skill patch, Operator abstraction, boundary replay | `evolution/skill_patch.py`, `evolution/operator_abstraction.py` |
| Linked bundle, source-replay check, gate | `evolution/bundle.py`, `evolution/validation.py` |
| Strict-improvement update | `evolution/evolver.py` (batch), `evolution/live.py` (gateway, user-promoted) |
| Scientific tools | `scilib/` (42 modules), a catalog of 292 tool functions, 28 pretrained-weight assets (`scienceclaw/tools/weights.json`) |
| Gateway integration | `extensions/scienceclaw/` plugin and `python -m scienceclaw.rpc` |
</details>

```
ScienceClaw/
├── packages/scienceclaw/    # the engine: typed workflows, runtime, Skill/Operator program, evolution, tool library
│   ├── scienceclaw/         #   core, runtime, agent, canvas, program, evolution, llm, tools, rpc, cli
│   ├── scilib/              #   42 scientific tool modules
│   ├── configs/             #   default run configuration (no model, no paths)
│   ├── scripts/             #   LLM serving, GPU-tool broker and environment scripts
│   └── docs/                #   DESIGN.md (the contract) and INTEGRATION.md
├── extensions/scienceclaw/  # gateway plugin: canvas, tools, program, evolve
├── skills/                  # 300+ skills, including the seed program and the 23 discipline skills
├── mcp-servers/             # arXiv-LaTeX and ChEMBL MCP servers
├── assets/                  # banner and the paper's figures
├── SCIENCE.md               # research protocol for the gateway agent
├── setup.sh                 # one-click setup
├── src/, ui/, apps/, ...    # the OpenClaw gateway, web UI and apps
└── docs/                    # gateway documentation
```

**Documentation:** [`DESIGN.md`](packages/scienceclaw/docs/DESIGN.md) (the full contract, including the decisions the paper leaves open) · [`INTEGRATION.md`](packages/scienceclaw/docs/INTEGRATION.md) (gateway, plugin, skill and evolution map) · [plugin guide](extensions/scienceclaw/README.md) · [`SCIENCE.md`](SCIENCE.md)

## 📬 Contact and license

mingdazhang@ieee.org · MIT, see [LICENSE](LICENSE).
