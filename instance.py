import numpy as np

class Instance:
    def __init__(self, A: np.ndarray, p: np.ndarray) -> None:
        self.m, self.n = A.shape
        self.A = A
        self.p = p