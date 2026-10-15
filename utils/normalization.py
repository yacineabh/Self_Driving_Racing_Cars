import numpy as np

class Normalizer:
    def __init__(self, mean=None, std=None):
        self.mean, self.std = mean, std
    def fit(self, x):
        self.mean, self.std = x.mean(0), x.std(0) + 1e-6
        return self
    def __call__(self, x):
        return (x - self.mean) / self.std
    def inverse(self, x):
        return x * self.std + self.mean
    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std)
    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["mean"], d["std"])