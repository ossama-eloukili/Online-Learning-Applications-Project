from .GP_UCB_Learner import GP_UCB_Learner

import numpy as np
import random as rnd

class GP_TS_Learner(GP_UCB_Learner):
    def pull_arm(self):
        """
        returns the value of the arm to pull
        """
        if len(self.x) == 0:
            return rnd.choice(self.arms)
        
        samples = np.empty(len(self.arms))
        for i in range(len(self.arms)):
            samples[i] = np.random.normal(self.values[i], self.sigma[i])

        i = np.argmax(samples)

        return self.arms[i]
    

if __name__ == "__main__":
    pass