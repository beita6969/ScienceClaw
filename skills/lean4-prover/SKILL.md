---
name: lean4-prover
description: "Lean 4 theorem prover for formal verification. Use when: formal theorem proving, mathematical verification, proof search, type theory exploration. NOT for: numerical computation (use scipy), symbolic math (use sympy), statistical analysis (use statsmodels)."
metadata: { "openclaw": { "emoji": "📐", "requires": { "bins": ["lean"] } } }
---

# Lean 4 Theorem Prover

Formal theorem proving, mathematical verification, proof search, and type theory exploration.

## When to Use / When NOT to Use

**Use when:** formal theorem proving, mathematical verification, proof search, type-theoretic reasoning, formalized mathematics with Mathlib, program verification.

**NOT for:** numerical computation (use scipy/numpy), symbolic algebra or calculus (use sympy), statistical analysis (use statsmodels), quick calculations.

## Installation

If `lean` is not available, install via elan (the Lean version manager):

```bash
curl https://elan.lean-lang.org/install.sh -sSf | sh
# or on macOS:
brew install elan-init
elan default leanprover/lean4:stable
```

## Basic Theorem Structure

```lean
-- Simple proposition proof
theorem my_first_theorem : 1 + 1 = 2 := by
  rfl

-- Implication
theorem modus_ponens (P Q : Prop) (hp : P) (hpq : P → Q) : Q := by
  apply hpq
  exact hp

-- Universal quantifier
theorem add_comm_example : ∀ (a b : Nat), a + b = b + a := by
  intro a b
  omega
```

## Core Tactics

```lean
-- intro: introduce hypotheses / universally quantified variables
-- apply: apply a function or lemma to the goal
-- exact: provide the exact proof term
-- rfl: reflexivity (proves a = a or definitional equalities)
-- simp: simplification using simp lemmas
-- ring: prove equalities in commutative rings
-- omega: decide linear arithmetic over Nat and Int
-- linarith: linear arithmetic reasoning with hypotheses
-- cases / rcases: case split on inductive types
-- induction: structural induction
-- constructor: prove a conjunction or existential
-- contradiction: close goal from contradictory hypotheses

-- Example combining tactics
theorem example_proof (n : Nat) (h : n > 0) : n + n > n := by
  linarith

theorem ring_example (a b : Int) : (a + b) ^ 2 = a ^ 2 + 2 * a * b + b ^ 2 := by
  ring
```

## Project Setup with Lakefile

```bash
# Create a new Lean 4 project
lake new my_project
cd my_project

# Project structure:
# my_project/
#   lakefile.lean      -- build config
#   lean-toolchain     -- Lean version
#   MyProject/
#     Basic.lean       -- source files
```

```lean
-- lakefile.lean (with Mathlib dependency)
import Lake
open Lake DSL

package my_project where
  leanOptions := #[⟨`autoImplicit, false⟩]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "master"

@[default_target]
lean_lib MyProject where
  srcDir := "MyProject"
```

```bash
lake update           # fetch dependencies
lake build            # build the project
```

## Using Mathlib

```lean
import Mathlib.Tactic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.List.Basic

-- Mathlib provides thousands of lemmas and tactics
theorem list_length_append (l1 l2 : List α) :
    (l1 ++ l2).length = l1.length + l2.length := by
  simp [List.length_append]

-- norm_num: evaluate numerical expressions
example : (7 : ℤ) ∣ 42 := by norm_num

-- positivity: prove positivity/nonnegativity
example (x : ℝ) : 0 ≤ x ^ 2 + 1 := by positivity

-- polyrith: polynomial arithmetic (requires Mathlib)
-- gcongr: generalized congruence
-- field_simp: clear denominators in field expressions
```

## REPL Interaction

```bash
# Run a single Lean file
lean MyFile.lean

# Check a file and print messages
lean --run MyFile.lean

# Interactive: use VS Code with lean4 extension for goal state
# Or use lean4 language server directly
```

```lean
-- Use #check, #eval, #print for exploration
#check Nat.add_comm         -- view type signature
#eval 2 ^ 10               -- evaluate expressions
#print Nat.rec              -- print definition
```

## Best Practices

1. Set `autoImplicit` to `false` in lakefile to catch typos in variable names.
2. Use `sorry` as a placeholder while developing proofs, then eliminate all before finalizing.
3. Start proofs with `by` block and use `?` suffix tactics (e.g., `simp?`, `exact?`) to discover lemmas.
4. Use `#check` liberally to inspect types and available lemmas.
5. Keep proofs modular: extract helper lemmas rather than writing monolithic proofs.
6. For Mathlib projects, run `lake exe cache get` to download prebuilt oleans and speed up builds.
7. Use `omega` for natural/integer arithmetic goals; `ring` for polynomial identities; `linarith` when hypotheses are needed.
