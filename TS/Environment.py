import numpy as np

class Environment:

    def __init__(self, n_arms, probabilities):
        self.n_arms = n_arms
        self.probabilities = probabilities

    def round(self, n_daily_clicks, margin, cum_daily_costs, pulled_arm):
        # Reward as reported in clayrvoiant optimization algorithm
        # The conversion probability is the probability of buying the item given a price, thus is a Bernoulli probabilty
        reward = n_daily_clicks * np.random.binomial(1, self.probabilities[pulled_arm]) * margin - cum_daily_costs
        return reward