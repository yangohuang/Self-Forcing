# RTX 4090 Generated-History Training Feasibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether a single RTX 4090 can backpropagate through a 1.3B causal video generator after a detached autoregressive history block, then decide whether a controlled training comparison is justified.

**Architecture:** Use the official Self-Forcing causal Wan 1.3B implementation and a local Wan 1.3B weight file. A standalone probe runs one detached block, re-noises its output, fills the KV history, and backpropagates through a second block with low-rank trainable adapters. This is a memory and numerical feasibility test, not DMD or a full SRF reproduction. Record machine, source revision, settings, peak VRAM, elapsed time, loss and gradient values. If this gate passes, add an equivalent clean-history control and compare generated-context vs clean-context on held-out sequences; otherwise document the limiting resource and stop that route.

**Tech Stack:** PyTorch CUDA/bfloat16, official Self-Forcing Wan causal model, safetensors, pytest, RTX 4090.

---

### Task 1: Establish the reproducible baseline

**Files:**
- Create: `docs/benchmarks/2026-09-25-4090-rollout-feasibility.md`
- Create: `scripts/rollout_probe_4090.py`

- [ ] Check the local Wan checkpoint's tensor names, dtype and architectural config against `CausalWanModel`, and record the official source commit and environment versions.
- [ ] Import the model in a suitable existing environment. Install only missing packages in a dedicated environment if required, without changing the SoulX LiveAct environment.
- [ ] Add a `--dry-run` mode to the probe that validates checkpoint paths and dimensions without loading the full model; verify its output.
- [ ] Commit the baseline report and probe separately from upstream's `main` branch.

### Task 2: One generated-history backward step

**Files:**
- Modify: `scripts/rollout_probe_4090.py`
- Test: `tests/test_rollout_probe_4090.py`

- [ ] Write and run small tests for two-block scheduling, detached history, re-noise formula, loss and gradient reporting.
- [ ] Implement loading the causal Wan 1.3B generator in bfloat16 with a small LoRA adapter on selected attention projections; freeze base weights.
- [ ] Generate block 1 without gradients, re-noise its output for history, and run the cached second block with gradients. Use low latent resolution first, then increase toward the published setting only if memory allows.
- [ ] Run one optimizer step on the physical 4090. Save structured timings, peak allocated/reserved VRAM, finite-loss and nonzero-gradient checks. If OOM, document the exact configuration and one or two bounded mitigation attempts.
- [ ] Verify the probe on a second seed and commit the implementation and benchmark evidence.

### Task 3: Controlled comparison if Task 2 passes

**Files:**
- Modify: `scripts/rollout_probe_4090.py`
- Create: `docs/benchmarks/2026-09-25-generated-history-ab.md`

- [ ] Add clean-history and generated-history modes with identical model, noise, target, seed, training budget and evaluation paths.
- [ ] Train each mode on the same small, declared data split and evaluate on held-out longer clips. Report boundary discontinuity, drift and a visual artifact; make no quality claim without the held-out test.
- [ ] Repeat at least three seeds for any claimed directional improvement. Record failure cases and uncertainty.

### Task 4: SoulX and interview handoff

**Files:**
- Create: `docs/benchmarks/2026-09-25-soulx-vivix-decision.md`

- [ ] Relate the feasibility result to SoulX's observed block-12 jump and explain the gap between this Wan feasibility probe and a LiveAct forcing contribution.
- [ ] Identify a concrete next SoulX PR candidate only if it improves held-out video; keep it on the personal fork and do not open an upstream PR.
- [ ] Summarize the experiment as accurate interview evidence, separating prior merged work, new measured results and proposed research.

**Decision gates:** A successful Task 2 permits Task 3. Missing weights, incompatible code, OOM after bounded reductions, or impractical wall time trigger an explicit no-go report and Task 4; do not invent a positive A/B result.
