import random
import torch

class CovPolicy:
    def __init__(self, delta):
        self.delta = delta

    def __call__(self, Q_values, beta_probs, epsilon):
        actions = torch.argsort(Q_values)

        if all(beta_probs[a] > self.delta for a in actions):
            return max(actions, key=lambda a: Q_values[a]).item(), False

        low_coverage_actions = [a for a in actions if beta_probs[a] <= self.delta]
        return random.choice(low_coverage_actions).item(), True
    
    def __str__(self):
        return f"CovPolicy delta={self.delta}"

class CorPolicy:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, Q, beta, epsilon):
        actions = torch.argsort(Q)
        min_q = torch.min(Q)
        action = max(
            actions,
            key=lambda a: self.alpha * Q[a]
            + (1 - self.alpha) * (Q[a] if beta[a] > epsilon else min_q)
        )

        return action.item(), self.is_exploration(action, actions, Q, beta, epsilon)

    def is_exploration(self, action, actions, Q, beta, epsilon):
        return action != max(actions, key=lambda a: Q[a] if beta[a] > epsilon else 0)
    
    def __str__(self):
        return f"CorPolicy alpha={self.alpha}"
