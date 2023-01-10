import numpy as np


class Standardizer:
    def __init__(self, fp=None):
        self.center = None
        self.radius = None
        self.fp = fp

    def initialize_from_data(self, data):
        ub = np.max(data, axis=0)
        lb = np.min(data, axis=0)

        self.center = (ub + lb) / 2
        self.radius = (ub - lb) / 2

    def initialize_from_file(self, fp=None):
        if fp is not None:
            self.fp = fp

        with open(self.fp, "rb") as f:
            self.center, self.radius = np.load(f)

    def write_to_file(self, fp=None):
        assert self.center is not None
        assert self.radius is not None

        if fp is not None:
            self.fp = fp

        with open(self.fp, "wb") as f:
            np.save(f, [self.center, self.radius])

    def standardize(self, data):
        return (data - self.center) / self.radius

    def destandardize(self, data):
        return (data * self.radius) + self.center
