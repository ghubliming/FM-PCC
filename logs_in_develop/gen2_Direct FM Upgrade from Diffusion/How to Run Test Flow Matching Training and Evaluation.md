# How to Run Flow Matching Training and Evaluation

This guide explains exactly what you need to change to train and evaluate the new Flow Matching (FM) model using your existing scripts.

---

## 1. Change Model Import in Training/Evaluation Scripts

**Where:**
- Any script that imports and instantiates the diffusion model (e.g., `scripts/train.py`, `scripts/eval.py`, or your custom runner).

**What to change:**

**Before (original diffusion):**
```python
from diffuser.models.diffusion import GaussianDiffusion
# ...
model = GaussianDiffusion(...)
```

**After (Flow Matching):**
```python
from flow_matcher.models.diffusion import GaussianDiffusion
# ...
model = GaussianDiffusion(...)
```

---

## 2. (Optional) Update Config or Model Class String

If your config or script uses a string to specify the model class, update it:

**Before:**
```python
'diffusion': 'models.GaussianDiffusion'
```
**After:**
```python
'diffusion': 'flow_matcher.models.diffusion.GaussianDiffusion'
```

---

## 3. No Other Code Changes Needed
- The Trainer, dataset, optimizer, and all other training/eval logic remain unchanged.
- The FM model is API-compatible with the original diffusion model.

---

## 4. Run Training/Evaluation as Usual
- Launch your training or evaluation script as you normally would (e.g., `python scripts/train.py ...`).
- The FM model will be used in place of the diffusion model.

---

## 5. (Optional) Revert to Diffusion
- To switch back, simply restore the original import/class string.

---

## Summary Table
| Step | What to Change | Where |
|------|----------------|-------|
| 1    | Import path    | All scripts that use the model |
| 2    | Model class string (if used) | Config files or script |
| 3    | Nothing else   | Trainer, dataset, etc. |

---

**You do NOT need to rewrite your training or evaluation logic. Just change the model import/class and everything else will work as before, now using Flow Matching.**
