import random

import random

class CovPolicy:
    def __init__(self, delta=0.1):
        self.delta = delta

    def __call__(self, Q_values, beta_probs):
        actions = list(Q_values.keys())

        if all(beta_probs[a] > self.delta for a in actions):
            return max(actions, key=lambda a: Q_values[a])

        low_coverage_actions = [a for a in actions if beta_probs[a] <= self.delta]
        return random.choice(low_coverage_actions)

class CorPolicy:
    def __init__(self, alpha, epsilon):
        self.alpha = alpha
        self.epsilon = epsilon

    def __call__(self, Q, beta):
        min_q = min(Q.values())
        return max(
            Q,
            key=lambda a: self.alpha * Q[a]
            + (1 - self.alpha) * (Q[a] if beta[a] > self.epsilon else min_q)
        )

