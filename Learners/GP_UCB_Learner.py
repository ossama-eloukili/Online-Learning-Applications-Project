from .Learner import Learner

import numpy as np
import random as rnd
from sklearn.gaussian_process import GaussianProcessRegressor

class GP_UCB_Learner(Learner):
    def __init__(self, arms, kernel=None, alpha=1):
        self.values = {}
        self.ucb = {}
        for arm in arms:
            self.values[arm] = 0
            self.ucb[arm] = 0
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha = alpha, normalize_y = False)
        self.x = np.array([])
        self.y = np.array([])


    def pull_arm(self):
        """
        returns the value of the arm to pull
        """
        if len(self.x) == 0:
            return rnd.choice(list(self.ucb.keys()))    #TODO don't know if this is enough

        return max(self.ucb, key=self.ucb.get)
    
    
    def update(self, arm, reward):
        """
        updates the values given the reward and the arm pulled
        """
        self.x = np.append(self.x, arm)
        self.y = np.append(self.y, reward)
        X = np.atleast_2d(self.x).T
        Y = self.y.ravel()

        self.gp.fit(X, Y)

        for key in self.values:
            y_pred, sigma = self.gp.predict(key, return_std = True)
            self.values[key] = y_pred
            self.ucb[key] = y_pred + sigma  #TODO don't knoiw if this is coerrect


    def get_values(self):
        """
        returns a dictionary with the values for each arm
        """
        return self.values
    

if __name__ == "__main__":
    pass