from collections import deque
import numpy as np

class MetaController:
    def __init__(self, num_policies, window_size=1000):
        self.num_policies = num_policies
        self.window_size = window_size
        self.history = deque(maxlen=window_size)  # automatycznie usuwa ostatnie 
        # Przechowuje (policy_idx, reward, exploration_ratio)

    def select_policy(self):
        used_indices = [h[0] for h in self.history]
        for i in range(self.num_policies):
            if i not in used_indices:
                return i

        best_value = -float("inf")
        best_policy = 0

        for i in range(self.num_policies):
            value = self.count_mean(i) + self.count_exploration_bonus(i)
            if value > best_value:
                best_value = value
                best_policy = i

        return best_policy


    def update(self, policy_idx, reward, exploration_ratio):
        self.history.append((policy_idx, reward, exploration_ratio))

    def count_mean(self, i):
        rewards = [reward for policy_idx, reward, _ in self.history if policy_idx == i]
        if not rewards:
            return 0.0
        return sum(rewards) / len(rewards)
    
    def count_exploration_bonus(self, i): # bk(pi, L)
        n = self.count_policy_occurance(i)
        if n == 0:
            return float("inf")
        return (1 / n) * self.count_sum_of_exploration_ratio(i)


    def count_sum_of_exploration_ratio(self, i): # E Bm(pii)
        return sum(exploration_ratio for policy_idx, _, exploration_ratio in self.history if policy_idx == i)

    def count_policy_occurance(self, i): # Nk(pi, L)
        return sum(1 for policy_idx, _, _ in self.history if policy_idx == i)