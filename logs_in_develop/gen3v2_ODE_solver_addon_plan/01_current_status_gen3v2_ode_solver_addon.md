# 01 Current Status: Gen3v2 ODE Solver Addon

Date: 2026-04-14
Status: Locked Baseline and Naming
Scope: FM-v3 rollout ODE method selection path only

---

## 1) Objective

Define the locked gen3v2 direction for ODE solver adoption in FM-v3 evaluation:

1. Package/open-source first.
2. Paid route second (only if package route fails).
3. Custom solver last fallback only.

This rank is mandatory for gen3v2.

---

## 2) Verified Baseline

Current FM-v3 rollout uses explicit Euler in the sampler loop:

- [flow_matcher_v3/models/diffusion.py](../../flow_matcher_v3/models/diffusion.py#L133)
- [flow_matcher_v3/models/diffusion.py](../../flow_matcher_v3/models/diffusion.py#L135)
- [flow_matcher_v3/models/diffusion.py](../../flow_matcher_v3/models/diffusion.py#L173)

Evaluation path calls this sampler behavior:

- [FM_v3_test/eval_FM_v3.py](../../FM_v3_test/eval_FM_v3.py#L180)
- [flow_matcher_v3/sampling/policies.py](../../flow_matcher_v3/sampling/policies.py#L52)

---

## 3) Locked Development Rule

Implementation must follow this 3-step copy-modify rule:

1. Copy/modify evaluation entry scripts in [FM_v3_test](../../FM_v3_test).
2. Copy/modify rollout ODE logic in [flow_matcher_v3](../../flow_matcher_v3).
3. Inject exactly 2 selection parameters in [config/avoiding-d3il.py](../../config/avoiding-d3il.py).

Required naming style:

1. Use core-style names based on `flow_matching_v3`.
2. New variant name is `flow_matching_v3_ode_selectable`.

Required two parameters:

1. `ode_solver_backend_v3` default `legacy_euler`
2. `ode_solver_method_v3` default `euler`

Selection rule:

1. Choose method only from config values in [config/avoiding-d3il.py](../../config/avoiding-d3il.py).
2. No CLI method selection in gen3v2.

---

## 4) Solver Scope Clarification

Two solver classes remain strictly separated:

1. ODE integrator for flow rollout.
2. Optimization solver for projection/safety constraints.

Optimization examples (not ODE integrator replacement):

- [flow_matcher_v3/sampling/projection.py](../../flow_matcher_v3/sampling/projection.py#L138)
- [../../../SafeFlowMPC/safe_flow_mpc/SafetyFilter/SafetyFilterAcados.py](../../../SafeFlowMPC/safe_flow_mpc/SafetyFilter/SafetyFilterAcados.py#L157)

---

## 5) Reuse Audit Result

### 5.1 DPCC direct reuse for FM ODE integrator

No direct plug-and-play FM ODE integrator module was found.

Evidence:

1. DPCC diffusion model is diffusion denoising path:
	- [../../../dpcc/diffuser/models/diffusion.py](../../../dpcc/diffuser/models/diffusion.py)
2. DPCC projection module is optimization-focused:
	- [../../../dpcc/diffuser/sampling/projection.py](../../../dpcc/diffuser/sampling/projection.py#L145)

### 5.2 Open-source package reuse candidate in workspace

`torchdiffeq` is already used in vendored d3il code:

1. Imports:
	- [../../d3il/agents/models/beso/agents/diffusion_agents/k_diffusion/gc_sampling.py](../../d3il/agents/models/beso/agents/diffusion_agents/k_diffusion/gc_sampling.py#L8)
2. ODE call example:
	- [../../d3il/agents/models/beso/agents/diffusion_agents/k_diffusion/gc_sampling.py](../../d3il/agents/models/beso/agents/diffusion_agents/k_diffusion/gc_sampling.py#L492)
3. Dependency installation trace:
	- [../../d3il/install.sh](../../d3il/install.sh#L53)

---

## 6) Locked Rank Policy for Gen3v2

### Rank-1: Open-source package first

Primary candidate: `torchdiffeq` backend integration for FM-v3 ODE step path.

### Rank-2: Paid solver options only if rank-1 fails

Evaluate paid options only if measurable target is not met with rank-1.

### Rank-3: Build own solver only as last fallback

Allowed only after rank-1 and rank-2 are evaluated and rejected with evidence.

---

## 7) Compatibility Rule

1. Existing FM-v3 explicit Euler behavior stays default.
2. New package methods are additive and config-gated.
3. Missing new keys must still run the legacy Euler path.

---

## 8) Next Step

Use 02 to define the implementation details with:

1. package-first implementation sequence,
2. paid-option trigger criteria,
3. strict fallback gate before any custom solver code,
4. exact injection branch in rollout code with `legacy_euler` default and config-only selection.
