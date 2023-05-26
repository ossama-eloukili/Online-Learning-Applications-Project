from Learner import Learner

import numpy as np


class UCB1_Learner(Learner):
    def __init__(self, arms):
        self.values = {}
        self.counts = {}
        for arm in arms:
            self.values[arm] = 0
            self.counts[arm] = 0

    def pull_arm(self):
        """
        returns the value of the arm to pull
        """
        for arm in self.values:
            if self.counts[arm] == 0:
                return arm

        ucb_values = {}
        total_counts = sum(self.counts.values())
        for arm in self.values:
            ucb_values[arm] = self.values[arm] + np.sqrt( (2 * np.log(total_counts)) / (self.counts[arm]) )
        return max(ucb_values, key=ucb_values.get)

    def update(self, arm, reward):
        """
        updates the values given the reward and the arm pulled
        """
        self.counts[arm] += 1
        n = self.counts[arm]

        value = self.values[arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[arm] = new_value

    def get_values(self):
        """
        returns a dictionary with the values for each arm
        """
        return self.values


# for testing
if __name__ == "__main__":
    learner = UCB1_Learner([0.1, 0.2])
    learner.update(0.2, 0.4)
    learner.update(0.1, 0.8)
    learner.update(0.2, 0.2)
    print(learner.pull_arm())
    print(learner.get_values())