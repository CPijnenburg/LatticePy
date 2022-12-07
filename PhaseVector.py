import numpy as np

class PhaseVector:
    def __init__(self, rng = None):
        if rng is not None:
            self.phases = 2 * np.pi * rng.random(4)
        else:
            self.phases = np.zeros(4)