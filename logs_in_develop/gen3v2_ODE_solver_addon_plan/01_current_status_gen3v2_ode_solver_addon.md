# 01 Current Status: Gen3v2 ODE Solver Plan (Ranked Reuse Policy)

Date: 2026-04-13
Status: Active Start (Policy Locked)
Scope: FM-v3 evaluation ODE integration path only

---

## 1) Objective

Define a strict decision policy for ODE solver adoption in FM-v3 evaluation with this rank:

1. Use package/open-source solver first.
2. If insufficient, evaluate paid solver options (including educational discount/free license paths).
3. Build our own solver only as last fallback.

This rank is now mandatory for gen3v2.

---

## 2) Verified Current Baseline

FM-v3 currently does fixed-step Euler in sampler loop:

- [flow_matcher_v3/models/diffusion.py](../../flow_matcher_v3/models/diffusion.py#L133)
- [flow_matcher_v3/models/diffusion.py](../../flow_matcher_v3/models/diffusion.py#L135)
- [flow_matcher_v3/models/diffusion.py](../../flow_matcher_v3/models/diffusion.py#L173)

Eval path calls this sampler directly:

- [FM_v3_test/eval_FM_v3.py](../../FM_v3_test/eval_FM_v3.py#L180)
- [flow_matcher_v3/sampling/policies.py](../../flow_matcher_v3/sampling/policies.py#L52)

---

## 3) Solver Scope Clarification

Two solver classes remain strictly separated:

1. ODE integrator for flow rollout.
2. Optimization solver for projection/safety constraints.

Optimization examples (not ODE integrator replacements):

- [flow_matcher_v3/sampling/projection.py](../../flow_matcher_v3/sampling/projection.py#L138)
- [../../../SafeFlowMPC/safe_flow_mpc/SafetyFilter/SafetyFilterAcados.py](../../../SafeFlowMPC/safe_flow_mpc/SafetyFilter/SafetyFilterAcados.py#L157)

---

## 4) Reuse Audit Result (What We Can Reuse)

### 4.1 DPCC direct reuse for FM ODE integrator

No direct plug-and-play FM ODE integrator module found.

Evidence:

1. DPCC diffusion model is diffusion denoising path:
	- [../../../dpcc/diffuser/models/diffusion.py](../../../dpcc/diffuser/models/diffusion.py)
2. DPCC projection module is optimization-focused:
	- [../../../dpcc/diffuser/sampling/projection.py](../../../dpcc/diffuser/sampling/projection.py#L145)

### 4.2 Open-source package reuse candidate already in workspace

`torchdiffeq` is already used in vendored d3il code:

1. Imports:
	- [../../d3il/agents/models/beso/agents/diffusion_agents/k_diffusion/gc_sampling.py](../../d3il/agents/models/beso/agents/diffusion_agents/k_diffusion/gc_sampling.py#L8)
2. ODE call example:
	- [../../d3il/agents/models/beso/agents/diffusion_agents/k_diffusion/gc_sampling.py](../../d3il/agents/models/beso/agents/diffusion_agents/k_diffusion/gc_sampling.py#L492)
3. Dependency installation trace:
	- [../../d3il/install.sh](../../d3il/install.sh#L53)

---

## 5) Locked Rank Policy for Gen3v2

### Rank-1: Open-source package first

Primary candidate: `torchdiffeq` backend integration for FM-v3 ODE step path.

### Rank-2: Paid solver options only if rank-1 fails

Evaluate paid options only if measurable target is not met with rank-1.

### Rank-3: Build own solver only as last fallback

Allowed only after rank-1 and rank-2 are evaluated and rejected with evidence.

---

## 6) Compatibility Rule

1. Existing FM-v3 default behavior stays unchanged unless explicitly opting into new backend.
2. New solver policy is additive and gated by config keys.

---

## 7) Next Step

Use 02 to define exact execution with:

1. package-first implementation sequence,
2. paid-option trigger criteria,
3. strict fallback gate before any custom solver code.
