# Guide: Who Uses the 2 New Gen4 Config Files

This note documents the usage map for the two planned Gen4 config files:
1. `config/avoiding-d3il-gen4-visual.py`
2. `config/projection_eval_gen4_visual.yaml`

It answers: who consumes them, how they are loaded, and what must be changed.

---

## 1) File A: `config/avoiding-d3il-gen4-visual.py`

## 1.1 Primary consumers (Gen4 scripts)

This Python config module is consumed by all three Gen4 major scripts:
1. `FM_gen4_avoiding_visual_test/train_FM_gen4_avoiding_visual.py`
2. `FM_gen4_avoiding_visual_test/eval_FM_gen4_avoiding_visual.py`
3. `FM_gen4_avoiding_visual_test/load_results_FM_gen4_avoiding_visual.py`

Each script must define a parser like:
```python
class Parser(utils.Parser):
    dataset: str = 'avoiding-d3il-gen4-visual'
    config: str = 'config.avoiding-d3il-gen4-visual'
```

## 1.2 Runtime loading chain

The loading chain is:
1. script sets `config: str = 'config.avoiding-d3il-gen4-visual'`
2. `utils.Parser.parse_args(...)` calls `read_config(...)`
3. `read_config(...)` executes `importlib.import_module(args.config)`
4. module `base[experiment]` is read (plus dataset override block if present)

So if this module name changes, all three Gen4 scripts must be updated consistently.

## 1.3 Required experiment keys inside file A

At minimum, file A must contain:
1. `base['flow_matching_gen4_avoiding_visual']` for training
2. `base['plan_fm_gen4_avoiding_visual']` for eval/load

If these keys are missing, parser lookup fails at runtime.

---

## 2) File B: `config/projection_eval_gen4_visual.yaml`

## 2.1 Primary consumers (Gen4 scripts)

This YAML config is consumed by:
1. `FM_gen4_avoiding_visual_test/eval_FM_gen4_avoiding_visual.py`
2. `FM_gen4_avoiding_visual_test/load_results_FM_gen4_avoiding_visual.py`

Typical usage pattern:
```python
with open('config/projection_eval_gen4_visual.yaml', 'r') as file:
    config = yaml.safe_load(file)
```

## 2.2 Runtime usage in scripts

From this YAML, scripts read keys like:
1. `exps`
2. `seeds`
3. `projection_variants`
4. `avoiding_halfspace_variants`
5. `n_trials`
6. `plot_how_many`
7. `constraint_types`

If key names differ from existing eval/load expectations, evaluation scripts break.

---

## 3) Is changing only 2 copied folders enough?

Short answer: No.

Besides copying/modifying the two major folders (`FM_gen4_avoiding_visual_test/` and `flow_matcher_gen4_avoiding_visual/`), you must also:
1. create these two config files in existing `config/` folder,
2. update Gen4 script parser module string to `config.avoiding-d3il-gen4-visual`,
3. update Gen4 eval/load scripts to open `config/projection_eval_gen4_visual.yaml`,
4. ensure file A includes required Gen4 experiment keys.

Without these config-side changes, the copied folders cannot run.

---

## 4) Legacy safety

This two-file plan is backward-safe if you keep additive rules:
1. do not rename existing config folder,
2. do not remove old config files,
3. do not rewrite old experiment keys.

Then old baseline and Gen4 can coexist.

---

## 5) Verdict (Current Repository State)

Current verdict from repository inspection:
1. the two new Gen4 config files are planned targets and not yet loaded by runtime scripts,
2. current references exist in documentation/planning only,
3. once Gen4 folder scripts are created, these two config files should be bound only there,
4. old baseline scripts remain on old config files.
