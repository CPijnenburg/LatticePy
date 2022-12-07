import numpy as np
from PhaseVector import PhaseVector


class Lattice:
    UNIT_VECTORS = np.eye(4)
    ACCEPTANCE_CONSTANT = (np.sqrt(np.pi ** 2 - 4) - 2 * np.arccos(2 / np.pi)) / np.pi

    def __init__(self, size, random = True, rng = np.random.default_rng()):

        self.size = size
        self.rng = rng
        self.lattice = np.zeros((self.size, self.size, self.size, self.size, 4))
        if random:
            for i in range(self.size):
                for j in range(self.size):
                    for k in range(self.size):
                        for l in range(self.size):
                            self.lattice[i, j, k, l] = PhaseVector(rng).phases

    def n_m(self, i):
        if i < 0:
            return int(self.size - 1)
        else:
            return int(i%self.size)


    def plaquettes_without_link(self, i, j, k, l, m):
        lambda_sum = 0j

        for n in range(4):
            if n != m:
                phase1 = self.lattice[int((i + self.UNIT_VECTORS[m, 0]) % self.size), 
                                      int((j + self.UNIT_VECTORS[m, 1]) % self.size),
                                      int((k + self.UNIT_VECTORS[m, 2]) % self.size),
                                      int((l + self.UNIT_VECTORS[m, 3]) % self.size), n]

                phase2 = self.lattice[int((i + self.UNIT_VECTORS[n, 0]) % self.size), 
                                      int((j + self.UNIT_VECTORS[n, 1]) % self.size),
                                      int((k + self.UNIT_VECTORS[n, 2]) % self.size),
                                      int((l + self.UNIT_VECTORS[n, 3]) % self.size), m]

                phase3 = self.lattice[i, j, k, l, n]


                lambda_sum += np.exp((phase1 - phase2 - phase3) * 1j)


                phase4 = self.lattice[self.n_m(i - self.UNIT_VECTORS[n, 0]), 
                                      self.n_m(j - self.UNIT_VECTORS[n, 1]),
                                      self.n_m(k - self.UNIT_VECTORS[n, 2]),
                                      self.n_m(l - self.UNIT_VECTORS[n, 3]), m]

                phase5 = self.lattice[self.n_m(i - self.UNIT_VECTORS[n, 0] + self.UNIT_VECTORS[m, 0]), 
                                      self.n_m(j - self.UNIT_VECTORS[n, 1] + self.UNIT_VECTORS[m, 1]),
                                      self.n_m(k - self.UNIT_VECTORS[n, 2] + self.UNIT_VECTORS[m, 2]),
                                      self.n_m(l - self.UNIT_VECTORS[n, 3] + self.UNIT_VECTORS[m, 3]), n]

                phase6 = self.lattice[self.n_m(i - self.UNIT_VECTORS[n, 0]), 
                                      self.n_m(j - self.UNIT_VECTORS[n, 1]),
                                      self.n_m(k - self.UNIT_VECTORS[n, 2]),
                                      self.n_m(l - self.UNIT_VECTORS[n, 3]), n]

                lambda_sum += np.exp((-phase4 - phase5 + phase6) * 1j)

        return lambda_sum

    def average_action(self):
        num_plaq = 6 * self.size ** 4
        sum = 0

        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.size):
                    for l in range(self.size):

                            for m in range(3):
                                n = m + 1
                                while n < 4:
                                    phase1 = self.lattice[i, j, k, l, m]
                                    phase2 = self.lattice[int((i + self.UNIT_VECTORS[m, 0]) % self.size),
                                                          int((j + self.UNIT_VECTORS[m, 1]) % self.size),
                                                          int((k + self.UNIT_VECTORS[m, 2]) % self.size), 
                                                          int((l + self.UNIT_VECTORS[m, 3]) % self.size), n]

                                    phase3 = self.lattice[int((i + self.UNIT_VECTORS[n, 0]) % self.size),
                                                          int((j + self.UNIT_VECTORS[n, 1]) % self.size),
                                                          int((k + self.UNIT_VECTORS[n, 2]) % self.size), 
                                                          int((l + self.UNIT_VECTORS[n, 3]) % self.size), m]
                                    phase4 = self.lattice[i, j, k, l, n]

                                    sum += 1 - np.cos(phase1 + phase2 - phase3 - phase4)
                                    n += 1

        return sum / num_plaq

    def acceptance_probability(self, x, prefactor):
        return np.exp(prefactor * (np.cos((np.pi / 2) * (1 - x)) - x)) / np.exp(self.ACCEPTANCE_CONSTANT * prefactor)

    def sample_theta(self, alpha, beta):
        prefactor = alpha * beta

        while True:
            sample_x = -1 + (1 / prefactor) * np.log(1 + (np.exp(2 * prefactor) - 1) * self.rng.random())
            if (self.rng.random() < self.acceptance_probability(sample_x, prefactor)):
                theta = (np.pi / 2) * (1 - sample_x)
                
                return self.rng.choice([-1, 1]) * theta

    def heathbath_update(self, beta):
        random_i = self.rng.integers(0, self.size)
        random_j = self.rng.integers(0, self.size)
        random_k = self.rng.integers(0, self.size)
        random_l = self.rng.integers(0, self.size)
        random_m = self.rng.integers(0, 4)

        other_plaquettes = self.plaquettes_without_link(random_i, random_j, random_k, random_l, random_m)
        alpha = np.abs(other_plaquettes)
        theta_0 = -np.angle(other_plaquettes)

        new_theta = self.sample_theta(alpha, beta)

        self.lattice[random_i, random_j, random_k, random_l, random_m] = new_theta + theta_0