
import numpy as np


class Normalizer:

    def __init__(self, method="zscore"):
        self.method = method
        self.stat_a = None  
        self.stat_b = None 

    def fit(self, X):
        if self.method == "zscore":
            self.stat_a = X.mean()
            self.stat_b = X.std()
            if self.stat_b == 0:
                self.stat_b = 1.0
        elif self.method == "minmax":
            self.stat_a = X.min()
            self.stat_b = X.max() - X.min()
            if self.stat_b == 0:
                self.stat_b = 1.0
        elif self.method in (None, "none"):
            pass
        else:
            raise ValueError(f"Unknown normalization_method: {self.method}")
        return self

    def transform(self, X):
        if self.method in (None, "none"):
            return X.astype(np.float32)
        return ((X - self.stat_a) / self.stat_b).astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def build_normalizer(dataset_cfg):
    """Create a :class:`Normalizer` from ``dataset.yaml`` preprocessing settings."""
    pre = dataset_cfg.get("preprocessing", {})
    if not pre.get("normalize", False):
        return Normalizer(method="none")
    return Normalizer(method=pre.get("normalization_method", "zscore"))
