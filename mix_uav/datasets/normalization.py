import os

import numpy as np
import scipy.interpolate as interpolate
import pdb

#-----------------------------------------------------------------------------#
#-------------- Gen15 Fix_16 — degenerate (constant) dimensions --------------#
#-----------------------------------------------------------------------------#
# A dimension whose data is CONSTANT (min == max) carries no scale information and no
# gradient signal. `SafeLimitsNormalizer` has to widen it to avoid 0/0 = NaN, and the
# width it picks silently becomes that channel's physical output scale:
#
#     unnormalize(y) = (y+1)/2 * (maxs-mins) + mins = y*eps + c
#
# With the inherited `eps=1` and c=0 that is the IDENTITY — the model's raw normalized
# output is emitted verbatim in data units, and `LimitsNormalizer`'s clip to [-1,1]
# becomes a +/-1 *data-unit* ceiling. On D3IL/maze (actions O(1)) that is harmless. On the
# UAV (actions O(0.02 m), and `pillars` has dz == 0 exactly) it gives the vertical action
# ~23x the gain of every other channel, and makes the projector's `action_bounds:'auto'`
# cap for that channel +/-1 m — i.e. no cap at all.
#
# See logs_in_develop/Gen15/Study/STUDY_20260901_mf_unguided_failure_uav_pillars.md.
#
# FIX: derive the widening from the scale of the NON-constant dimensions instead of using a
# fixed 1.0. `normalize()` is UNCHANGED for any eps (constant data always maps to the
# midpoint 0), so this is CHECKPOINT-COMPATIBLE — no retrain is required.
#
#   FMPCC_SAFE_EPS_MODE = 'scaled' (default) | 'legacy'   ('legacy' restores eps=1.0 exactly,
#                                                          for A/B against pre-fix runs)
#   FMPCC_SAFE_EPS_FRAC = fraction of the median non-constant half-width (default 1e-3)
SAFE_EPS_MODE = os.environ.get('FMPCC_SAFE_EPS_MODE', 'scaled').strip().lower()
SAFE_EPS_FRAC = float(os.environ.get('FMPCC_SAFE_EPS_FRAC', '1e-3'))
SAFE_EPS_FLOOR = 1e-8

POINTMASS_KEYS = ['observations', 'actions', 'next_observations', 'deltas']

#-----------------------------------------------------------------------------#
#--------------------------- multi-field normalizer --------------------------#
#-----------------------------------------------------------------------------#

class DatasetNormalizer:

    def __init__(self, dataset, normalizer, path_lengths=None):
        dataset = flatten(dataset, path_lengths)

        self.observation_dim = dataset['observations'].shape[1]
        self.action_dim = dataset['actions'].shape[1]

        if type(normalizer) == str:
            normalizer = eval(normalizer)

        self.normalizers = {}
        for key, val in dataset.items():
            # Gen15 Fix_16 — forward the field name when the normalizer accepts one, so the
            # degenerate-dimension diagnostic can say `actions[2]` and not just `2`. Falls
            # back to the original call for any normalizer that does not take `key`.
            try:
                self.normalizers[key] = normalizer(val, key=key)
                continue
            except TypeError:
                pass
            except Exception:
                print(f'[ utils/normalization ] Skipping {key} | {normalizer}')
                continue
            try:
                self.normalizers[key] = normalizer(val)
            except Exception:
                print(f'[ utils/normalization ] Skipping {key} | {normalizer}')

    def __repr__(self):
        string = ''
        for key, normalizer in self.normalizers.items():
            string += f'{key}: {normalizer}]\n'
        return string

    def __call__(self, *args, **kwargs):
        return self.normalize(*args, **kwargs)

    def normalize(self, x, key):
        return self.normalizers[key].normalize(x)

    def unnormalize(self, x, key):
        return self.normalizers[key].unnormalize(x)

    def get_field_normalizers(self):
        return self.normalizers

def flatten(dataset, path_lengths):
    '''
        flattens dataset of { key: [ n_episodes x max_path_lenth x dim ] }
            to { key : [ (n_episodes * sum(path_lengths)) x dim ]}
    '''
    flattened = {}
    for key, xs in dataset.items():
        assert len(xs) == len(path_lengths)
        flattened[key] = np.concatenate([
            x[:length]
            for x, length in zip(xs, path_lengths)
        ], axis=0)
    return flattened

#-----------------------------------------------------------------------------#
#------------------------------- @TODO: remove? ------------------------------#
#-----------------------------------------------------------------------------#

class PointMassDatasetNormalizer(DatasetNormalizer):

    def __init__(self, preprocess_fns, dataset, normalizer, keys=POINTMASS_KEYS):

        reshaped = {}
        for key, val in dataset.items():
            dim = val.shape[-1]
            reshaped[key] = val.reshape(-1, dim)

        self.observation_dim = reshaped['observations'].shape[1]
        self.action_dim = reshaped['actions'].shape[1]

        if type(normalizer) == str:
            normalizer = eval(normalizer)

        self.normalizers = {
            key: normalizer(reshaped[key])
            for key in keys
        }

#-----------------------------------------------------------------------------#
#-------------------------- single-field normalizers -------------------------#
#-----------------------------------------------------------------------------#

class Normalizer:
    '''
        parent class, subclass by defining the `normalize` and `unnormalize` methods
    '''

    def __init__(self, X, key=None):
        self.X = X.astype(np.float32)
        self.mins = X.min(axis=0)
        self.maxs = X.max(axis=0)
        # Gen15 Fix_16 — the field this normalizer belongs to ('observations'/'actions'),
        # so diagnostics can name the offending channel instead of a bare index.
        self.key = key

    def __repr__(self):
        return (
            f'''[ Normalizer ] dim: {self.mins.size}\n    -: '''
            f'''{np.round(self.mins, 2)}\n    +: {np.round(self.maxs, 2)}\n'''
        )

    def __call__(self, x):
        return self.normalize(x)

    def normalize(self, *args, **kwargs):
        raise NotImplementedError()

    def unnormalize(self, *args, **kwargs):
        raise NotImplementedError()


class DebugNormalizer(Normalizer):
    '''
        identity function
    '''

    def normalize(self, x, *args, **kwargs):
        return x

    def unnormalize(self, x, *args, **kwargs):
        return x


class GaussianNormalizer(Normalizer):
    '''
        normalizes to zero mean and unit variance
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.means = self.X.mean(axis=0)
        self.stds = self.X.std(axis=0)
        self.z = 1

    def __repr__(self):
        return (
            f'''[ Normalizer ] dim: {self.mins.size}\n    '''
            f'''means: {np.round(self.means, 2)}\n    '''
            f'''stds: {np.round(self.z * self.stds, 2)}\n'''
        )

    def normalize(self, x):
        return (x - self.means) / self.stds

    def unnormalize(self, x):
        return x * self.stds + self.means


class LimitsNormalizer(Normalizer):
    '''
        maps [ xmin, xmax ] to [ -1, 1 ]
    '''

    def normalize(self, x):
        ## [ 0, 1 ]
        x = (x - self.mins) / (self.maxs - self.mins)
        ## [ -1, 1 ]
        x = 2 * x - 1
        return x

    def unnormalize(self, x, eps=1e-4):
        '''
            x : [ -1, 1 ]
        '''
        if x.max() > 1 + eps or x.min() < -1 - eps:
            # 🔴 Gen15 Fix_16 — this clip used to be SILENT (the warning was commented out).
            # It is not cosmetic: a saturating channel is clipped to the dimension's own
            # `maxs`, and for a widened constant dimension that is a full data-unit ceiling.
            # A whole eval batch could sit on that ceiling for >50% of its steps and no log
            # line anywhere said so. Warn ONCE per normalizer (batch logs must stay quiet —
            # see Slurm logging rules) and keep running counters for the caller to report.
            self._clip_events = getattr(self, '_clip_events', 0) + 1
            self._clip_max = max(getattr(self, '_clip_max', 0.0),
                                 float(max(abs(x.max()), abs(x.min()))))
            if self._clip_events == 1:
                print(f'[ utils/normalization ] ⚠ Fix_16: sample out of range on '
                      f'{self.key or "?"} — clipping to [-1,1] | '
                      f'({float(x.min()):.4f}, {float(x.max()):.4f}). '
                      f'Further occurrences silent; see `_clip_events` / `_clip_max`.')
            x = np.clip(x, -1, 1)

        ## [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.

        return x * (self.maxs - self.mins) + self.mins

class SafeLimitsNormalizer(LimitsNormalizer):
    '''
        functions like LimitsNormalizer, but can handle data for which a dimension is constant
    '''

    def __init__(self, *args, eps=None, **kwargs):
        super().__init__(*args, **kwargs)

        const = np.asarray(self.mins == self.maxs).reshape(-1)
        self.degenerate_dims = [int(i) for i in np.nonzero(const)[0]]
        self.degenerate_eps = {}
        if not self.degenerate_dims:
            return

        width = self._resolve_eps(const, eps)
        for i in self.degenerate_dims:
            # Widen ONLY this constant dimension so it maps to the midpoint (0)
            # instead of 0/0=NaN. Must index [i] — adjusting the whole mins/maxs
            # array would corrupt every other (non-constant) dimension's scale.
            self.mins[i] -= width
            self.maxs[i] += width
            self.degenerate_eps[i] = float(width)
            print(f'[ utils/normalization ] Constant data in '
                  f'{self.key or "?"}[{i}] | max = min = {self.maxs[i] - width} '
                  f'→ Fix_16 widened by eps={width:.3e} (mode={SAFE_EPS_MODE}). '
                  f'unnormalize scale for this channel = {width:.3e} data-units per unit output.')

    def _resolve_eps(self, const, eps):
        '''Gen15 Fix_16 — pick the widening for a constant dimension.

        `normalize()` maps constant data to the midpoint 0 for ANY eps, so this choice does
        not change the training signal and does not invalidate a checkpoint. What it DOES set
        is the physical scale of that channel at `unnormalize` time, and therefore:
          • how large a command an unconstrained model output becomes, and
          • the `action_bounds:'auto'` cap the projector derives from `mins`/`maxs`.

        Default ('scaled'): a small fraction of the MEDIAN half-width of the non-constant
        dimensions, so the degenerate channel is quieter than every real one instead of
        louder. 'legacy' restores the inherited eps=1.0 byte-for-byte for A/B.
        '''
        if eps is not None:
            return float(eps)
        if SAFE_EPS_MODE == 'legacy':
            return 1.0
        halfw = (np.asarray(self.maxs) - np.asarray(self.mins))[~const] / 2.0
        halfw = halfw[np.isfinite(halfw) & (halfw > 0)]
        if halfw.size == 0:
            # Every dimension is constant — there is no scale to reference. Fall back to the
            # inherited behaviour rather than invent one, and say so.
            print('[ utils/normalization ] ⚠ Fix_16: ALL dimensions constant — '
                  'no reference scale; falling back to eps=1.0.')
            return 1.0
        return max(float(np.median(halfw)) * SAFE_EPS_FRAC, SAFE_EPS_FLOOR)

#-----------------------------------------------------------------------------#
#------------------------------- CDF normalizer ------------------------------#
#-----------------------------------------------------------------------------#

class CDFNormalizer(Normalizer):
    '''
        makes training data uniform (over each dimension) by transforming it with marginal CDFs
    '''

    def __init__(self, X):
        super().__init__(atleast_2d(X))
        self.dim = self.X.shape[1]
        self.cdfs = [
            CDFNormalizer1d(self.X[:, i])
            for i in range(self.dim)
        ]

    def __repr__(self):
        return f'[ CDFNormalizer ] dim: {self.mins.size}\n' + '    |    '.join(
            f'{i:3d}: {cdf}' for i, cdf in enumerate(self.cdfs)
        )

    def wrap(self, fn_name, x):
        shape = x.shape
        ## reshape to 2d
        x = x.reshape(-1, self.dim)
        out = np.zeros_like(x)
        for i, cdf in enumerate(self.cdfs):
            fn = getattr(cdf, fn_name)
            out[:, i] = fn(x[:, i])
        return out.reshape(shape)

    def normalize(self, x):
        return self.wrap('normalize', x)

    def unnormalize(self, x):
        return self.wrap('unnormalize', x)

class CDFNormalizer1d:
    '''
        CDF normalizer for a single dimension
    '''

    def __init__(self, X):
        assert X.ndim == 1
        self.X = X.astype(np.float32)
        quantiles, cumprob = empirical_cdf(self.X)
        self.fn = interpolate.interp1d(quantiles, cumprob)
        self.inv = interpolate.interp1d(cumprob, quantiles)

        self.xmin, self.xmax = quantiles.min(), quantiles.max()
        self.ymin, self.ymax = cumprob.min(), cumprob.max()

    def __repr__(self):
        return (
            f'[{np.round(self.xmin, 2):.4f}, {np.round(self.xmax, 2):.4f}'
        )

    def normalize(self, x):
        x = np.clip(x, self.xmin, self.xmax)
        ## [ 0, 1 ]
        y = self.fn(x)
        ## [ -1, 1 ]
        y = 2 * y - 1
        return y

    def unnormalize(self, x, eps=1e-4):
        '''
            X : [ -1, 1 ]
        '''
        ## [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.

        if (x < self.ymin - eps).any() or (x > self.ymax + eps).any():
            print(
                f'''[ dataset/normalization ] Warning: out of range in unnormalize: '''
                f'''[{x.min()}, {x.max()}] | '''
                f'''x : [{self.xmin}, {self.xmax}] | '''
                f'''y: [{self.ymin}, {self.ymax}]'''
            )

        x = np.clip(x, self.ymin, self.ymax)

        y = self.inv(x)
        return y

def empirical_cdf(sample):
    ## https://stackoverflow.com/a/33346366

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample.size

    return quantiles, cumprob

def atleast_2d(x):
    if x.ndim < 2:
        x = x[:,None]
    return x

