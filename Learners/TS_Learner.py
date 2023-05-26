from Learner import Learner

import numpy as np


class TS_Learner(Learner):
    def __init__(self, arms):
        self.values = {}
        self.counts = {}
        self.beta = {}
        for arm in arms:
            self.values[arm] = 0
            self.counts[arm] = 0
            self.beta[arm] = [1, 1]

    def pull_arm(self):
        """
        returns the value of the arm to pull
        """
        samples = {}
        for arm in self.beta:
            samples[arm] = np.random.beta(self.beta[arm][0], self.beta[arm][1])
        # print(samples)
        return max(samples, key=samples.get)
    
    def update(self, arm, reward):
        """
        updates the values given the reward and the arm pulled
        """
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[arm] = new_value

        self.beta[arm][0] += reward
        self.beta[arm][1] += 1 - reward

    def get_values(self):
        """
        returns a dictionary with the values for each arm
        """
        return self.values
    

if __name__ == "__main__":
    learner = TS_Learner([0.1, 0.2])
    print(learner.pull_arm())
    learner.update(0.2, 0.2)
    learner.update(0.1, 0.1)
    learner.update(0.2, 0.3)
    print(learner.get_values())
    print(learner.pull_arm())