from .Learner import Learner

import numpy as np
import random as rnd
from sklearn.gaussian_process import GaussianProcessRegressor

class GP_UCB_Learner(Learner):
    def __init__(self, arms, kernel=None, alpha=1):
        self.arms = list(arms)
        self.values = None
        self.sigma = None
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha = alpha, normalize_y = False)
        self.x = np.array([])
        self.y = np.array([])


    def pull_arm(self):
        """
        returns the value of the arm to pull
        """
        if len(self.x) == 0:
            return rnd.choice(self.arms)
        
        ucb = self.values + 2*self.sigma
        i = np.argmax(ucb)

        return self.arms[i]
    
    
    def update(self, arm, reward):
        """
        updates the values given the reward and the arm pulled
        """
        self.x = np.append(self.x, arm)
        self.y = np.append(self.y, reward)

        X = np.atleast_2d(self.x).T
        Y = self.y.ravel()
        self.gp.fit(X, Y)

        self.values, self.sigma = self.gp.predict(np.atleast_2d(self.arms).T, return_std = True)


    def get_values(self):
        """
        returns a dictionary with the values for each arm
        """
        dict = {}
        for i, arm in enumerate(self.arms):
            dict[arm] = self.values[i]

        return dict
    

if __name__ == "__main__":
    pass