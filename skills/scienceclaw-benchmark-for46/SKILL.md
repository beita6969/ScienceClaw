---
name: scienceclaw-benchmark-for46
description: "Use when a task gives Python programming problems (a function stub with docstring or a task description, an entry-point name, a few public assert tests) and asks for one complete program per problem that must pass hidden unit tests, scored by execution pass@1 (HumanEval / MBPP style code generation; corresponds to FoR46 of the companion ScienceClaw-Eval benchmark)."
metadata: { "openclaw": { "emoji": "📊" } }
---

# Python code generation with hidden tests

**Task.** Input: problem dicts with `prompt` (HumanEval-style stub with signature and docstring, or an MBPP-style task text), `entry_point`, `visible_tests` (newline-separated `assert` lines; may be few or none) and `kind` (`python_function_stub` or other). Deliverable: a list of strings, one Python program per problem, in order. For stub problems the executed program is `prompt` followed by your string (a full function definition overrides the stub, a bare body continues it), otherwise your string alone; hidden tests are appended and run with a per-statement time limit (6 s default in `build_prompts`).

**Quality.** Execution pass@1, higher is better: the share of problems whose single program passes all hidden tests; the program must define `entry_point` with a real body.

**Tools (`from scilib import codegen`, no pretrained weights).**
- `build_prompts(problems, style="direct"|"plan"|"examples")` (operator `code_generation_prompts`).
- `sanitize(texts, problems, index=None, base=None)`: replies (fenced code, bodies, truncated fences, prose) -> programs; keeps the entry-point function and what it reaches, drops demo code, uses a failing placeholder for unusable replies (`code_reply_sanitize`).
- `repair_prompts(problems, codes, results)` -> `{"prompts", "index"}` (`code_repair_prompts`); `select_best(candidate_sets, results)` (`code_select_best_candidate`); `assembled_program`, `is_implemented`.
- `results` = per-problem dicts `passed`, `n_tests`, `n_passed`, `error`, `failing_tests`. Use the task's declared test tool if any, else a code node that runs each program plus its public asserts in a separate subprocess with a timeout.

**Routes.** Code node (`build_prompts`) -> `llm` node (template `{item}`, items = prompts, `config.parse = "text"`; `outputs` has one reply or None per item) -> code node (`sanitize`) -> public tests -> `repair_prompts` -> second `llm` node -> `sanitize(..., index=, base=)`. Code nodes cannot call the model. Identical prompts give identical replies (temperature 0, cached), so get candidate diversity from the styles or `config.seed` / `config.temperature`, then `select_best`.

**Rules.**
- Public tests are a subset of the hidden ones: never special-case visible examples; solve the stated behaviour.
- Keep the entry-point name and signature; no `input()`, prints, network or file writes; stay well inside the time limit.
- Execute generated code only in sandboxed code nodes or the task's test tool.
- Exactly one string per problem.
