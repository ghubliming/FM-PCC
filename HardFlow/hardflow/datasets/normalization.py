import numpy as np
import scipy.interpolate as interpolate
import torch

POINTMASS_KEYS = ["observations", "actions", "next_observations", "deltas"]


class DatasetNormalizer:

    def __init__(self, dataset, normalizer, path_lengths=None):
        dataset = flatten(dataset, path_lengths)

        self.observation_dim = dataset["observations"].shape[1]
        self.action_dim = dataset["actions"].shape[1]

        if type(normalizer) == str:
            normalizer = eval(normalizer)

        self.normalizers = {}
        for key, val in dataset.items():
            try:
                self.normalizers[key] = normalizer(val)
            except Exception as e:
                print(f"[ utils/normalization ] Skipping {key} | {normalizer} | {e}")

    def __repr__(self):
        string = ""
        for key, normalizer in self.normalizers.items():
            string += f"{key}: {normalizer}]\n"
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
    flattened = {}
    for key, xs in dataset.items():
        assert len(xs) == len(path_lengths)
        flattened[key] = np.concatenate(
            [x[:length] for x, length in zip(xs, path_lengths)], axis=0
        )
    return flattened


class PointMassDatasetNormalizer(DatasetNormalizer):

    def __init__(self, preprocess_fns, dataset, normalizer, keys=POINTMASS_KEYS):

        reshaped = {}
        for key, val in dataset.items():
            dim = val.shape[-1]
            reshaped[key] = val.reshape(-1, dim)

        self.observation_dim = reshaped["observations"].shape[1]
        self.action_dim = reshaped["actions"].shape[1]

        if type(normalizer) == str:
            normalizer = eval(normalizer)

        self.normalizers = {key: normalizer(reshaped[key]) for key in keys}


class Normalizer:

    def __init__(self, X):
        self.X = X.astype(np.float32)
        self.mins = X.min(axis=0)
        self.maxs = X.max(axis=0)

    def __repr__(self):
        return (
            f"""[ Normalizer ] dim: {self.mins.size}\n    -: """
            f"""{np.round(self.mins, 2)}\n    +: {np.round(self.maxs, 2)}\n"""
        )

    def __call__(self, x):
        return self.normalize(x)

    def normalize(self, *args, **kwargs):
        raise NotImplementedError()

    def unnormalize(self, *args, **kwargs):
        raise NotImplementedError()


class DebugNormalizer(Normalizer):

    def normalize(self, x, *args, **kwargs):
        return x

    def unnormalize(self, x, *args, **kwargs):
        return x


class GaussianNormalizer(Normalizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.means = torch.tensor(self.X, device="cuda").mean(dim=0)
        self.stds = torch.tensor(self.X, device="cuda").std(dim=0)
        self.z = 1

    def __repr__(self):
        return (
            f"""[ Normalizer ] dim: {self.mins.size}\n    """
            f"""means: {np.round(self.means, 2)}\n    """
            f"""stds: {np.round(self.z * self.stds, 2)}\n"""
        )

    def normalize(self, x):
        x_gpu = torch.tensor(x, device="cpu")
        result_gpu = (x_gpu - self.means.cpu()) / self.stds.cpu()
        result = result_gpu.cpu().numpy()

        del x_gpu
        del result_gpu
        torch.cuda.empty_cache()

        return result

    def unnormalize(self, x):
        x = torch.tensor(x, device="cpu")
        result = (x * self.stds.cpu() + self.means.cpu()).cpu().numpy()

        del x
        torch.cuda.empty_cache()

        return result


class LimitsNormalizer(Normalizer):

    def normalize(self, x):
        x = (x - self.mins) / (self.maxs - self.mins)
        x = 2 * x - 1
        return x

    def unnormalize(self, x, eps=1e-4):
        if x.max() > 1 + eps or x.min() < -1 - eps:
            x = np.clip(x, -1, 1)

        x = (x + 1) / 2.0

        return x * (self.maxs - self.mins) + self.mins
