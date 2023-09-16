import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct


class Learner:
    """
    Atributes:
    (....)
    rewards_per_arm: a list of list of dimension: number_of_arms x self.t

    (....)
    """
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.t = 0 #current round
        self.rewards_per_arm = x = [[] for i in range(n_arms)]
        self.collected_rewards = []


    def pull_arm(self):
        pass


    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards.append(reward)




####################                          Basic MABs                          ####################

class UCB1_Learner(Learner):
    def __init__(self, n_arms, support = (0,1)):
        """
        support: tuple
            the support interval of what we are learning
            e.g. in the case we are learning a probability p -> support = (0,1)
        """
        super().__init__(n_arms)
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.full(n_arms, np.inf)
        self.c = (support[1] - support[0]) * np.sqrt(2)


    def pull_arm(self):
        upper_confidence = self.empirical_means + self.confidence
        idx = np.random.choice(np.where(upper_confidence == upper_confidence.max())[0])
        return idx
    

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        
        #Model Updating:
        #-update mean
        self.empirical_means[pulled_arm] += (reward - self.empirical_means[pulled_arm]) / len(self.rewards_per_arm[pulled_arm])

        #COM: previously_
        # self.empirical_means[pulled_arm] += (reward - self.empirical_means[pulled_arm]) / self.t
        # it's the same as: self.empirical_means[pulled_arm] = ((self.t - 1) * self.empirical_means[pulled_arm] + reward) / self.t

        #-update confidences
        for a in range(self.n_arms):
            n_samples = len(self.rewards_per_arm[a])
            if n_samples > 0:
                self.confidence[a] = self.c * np.sqrt(np.log(self.t)/n_samples)


    def update_from_batch(self, pulled_arms, rewards, time):
        temp_empirical_means = np.zeros(self.n_arms)
        temp_cnt= np.zeros(self.n_arms)

        # Filling observations
        for pulled_arm, reward in zip(pulled_arms, rewards):
            self.update_observations(pulled_arm, reward)

            # updating temptemporary variables
            temp_cnt[pulled_arm] += 1
            temp_empirical_means[pulled_arm] += (reward - temp_empirical_means[pulled_arm]) / temp_cnt[pulled_arm]

        # Setting time
        self.t += time

        for a in range(self.n_arms):
            n_samples = len(self.rewards_per_arm[a])
            if n_samples > 0:
                # self.empirical_means[a] = sum(self.rewards_per_arm[a]) / n_samples # --- it should be the same as
                self.empirical_means[a] += (temp_empirical_means[a] - self.empirical_means[a]) * temp_cnt[a] / n_samples
                self.confidence[a] = self.c * np.sqrt(np.log(self.t)/n_samples)




class Binomial_TS_Learner(Learner):
    """
    Notes:
        - This algorithm learns p parameter of a binomial distribution (for each arm)
          assuming n is known (obs: n is the maximum value k can assume)
        - In this scenario the reward we observe is k

        Bayes:
            prior: P(p) = Beta(alpha, beta)
            likelihood: P(k|p;n) =  Binomial(n,p)
            ==> posterior: P(p|k;n) = Beta(alpha+k, beta+n-k)
    """
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.beta_parameters = np.ones((n_arms, 2))
        #self.k_max = k_max


    def pull_arm(self):
        idx = np.argmax(np.random.beta( self.beta_parameters[:,0], self.beta_parameters[:,1] ))
        return idx


    def update(self, pulled_arm, reward, k_max):
        self.t += 1
        self.update_observations(pulled_arm, reward)

        self.beta_parameters[pulled_arm,0] = self.beta_parameters[pulled_arm,0] + reward
        self.beta_parameters[pulled_arm,1] = self.beta_parameters[pulled_arm,1] + k_max - reward


    def update_from_batch(self, pulled_arms, rewards, k_maxs, time):
        # Filling observations and updating parameters
        for pulled_arm, reward, k_max in zip(pulled_arms, rewards, k_maxs):
            self.update_observations(pulled_arm, reward)
            self.beta_parameters[pulled_arm,0] = self.beta_parameters[pulled_arm,0] + reward
            self.beta_parameters[pulled_arm,1] = self.beta_parameters[pulled_arm,1] + k_max - reward

        # Setting time
        self.t += time



####################                          GP-MABs                          ####################

class GPUCB_Learner(Learner):
    def __init__(self, arms, kernel=None, alpha = 0.025):
        n_arms = len(arms)
        super().__init__(n_arms)

        self.arms = arms

        if kernel == None:
            kernel = ConstantKernel(1.0, (1e-4, 1e4)) * RBF(1.0, (1e-4, 1e4))#Obs: the hyperparamters will be optimized (changed) during learning by GaussianProcessRegressor class

        self.means = np.zeros(n_arms)
        self.sigmas = np.ones(n_arms)#np.full(n_arms, 1e3)#np.ones(n_arms)

        self.pulled_arms_x = []#here we will save the values of the pulled arms (not the indexes)
        #obs: actually, the information of the (arms pulled, reward) is there
        # e.g. [[i, self.rewards_per_arm[i]] for i in arms]
        # but creating another list is more processor efficient

        self.gp = GaussianProcessRegressor(kernel=kernel,
                                           alpha = alpha,
                                           n_restarts_optimizer = 5,# the higher the slower the run, try 4
                                           normalize_y=True)
        

    def update_observations(self, arm_idx, reward):
        super().update_observations(arm_idx, reward)
        self.pulled_arms_x.append([self.arms[arm_idx]])


    def pull_arm(self):
        upper_confidence = self.means + self.sigmas
        idx = np.random.choice(np.where(upper_confidence == upper_confidence.max())[0])
        return idx


    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)

        # update of the model:
        if len(self.collected_rewards) > 1:#at least 2 elements in our training set. (not sure why.. to update the kernel hyperparameters??)
            self.gp.fit(self.pulled_arms_x,self.collected_rewards)

        self.means, self.sigmas = self.gp.predict(np.reshape(self.arms,(-1,1)), return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-5)


    def update_from_batch(self, pulled_arms, rewards, time):
        # Filling observations
        for pulled_arm, reward in zip(pulled_arms, rewards):
            # fill observations context:
            self.update_observations(pulled_arm, reward)

        # Setting time
        self.t += time

        # Updating model learner:
        if len(self.collected_rewards) > 1:
            self.gp.fit(self.pulled_arms_x,self.collected_rewards)
            self.means, self.sigmas = self.gp.predict(np.reshape(self.arms,(-1,1)), return_std=True)
            self.sigmas = np.maximum(self.sigmas, 1e-5)




class GPTS_Learner(Learner):
    def __init__(self, arms, kernel = None, alpha = 0.025):
        n_arms = len(arms)
        super().__init__(n_arms)
        self.arms = arms

        self.means = np.zeros(n_arms)#np.full(n_arms, 1e3)
        self.sigmas = np.ones(n_arms)#np.full(n_arms, 1e3)#np.ones(n_arms)

        self.pulled_arms_x = []#here we will save the values of the pulled arms (not the indexes)
        #obs: actually, the information of the (arms pulled, reward) is there
        # e.g. [[i, self.rewards_per_arm[i]] for i in arms]
        # but creating another list is more processor efficient

        if kernel == None:
            kernel = ConstantKernel(1.0, (1e-4, 1e4)) * RBF(1.0, (1e-4, 1e4))#Obs: the hyperparamters will be optimized (changed) during learning

        self.gp = GaussianProcessRegressor(kernel=kernel,
                                           alpha = alpha,
                                           n_restarts_optimizer = 5,#the higher the slower the run
                                           normalize_y=True)


    def update_observations(self, arm_idx, reward):
        super().update_observations(arm_idx, reward)

        self.pulled_arms_x.append([self.arms[arm_idx]])


    def pull_arm(self):
        sampled_values = np.random.normal(self.means, self.sigmas)
        idx = np.argmax(sampled_values)
        return idx


    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)

        # update of the model:
        if len(self.collected_rewards) > 1:#at least 2 elements in our training set. (not sure why.. to update the kernel hyperparameters??)
            self.gp.fit(self.pulled_arms_x, self.collected_rewards)

        self.means, self.sigmas = self.gp.predict(np.reshape(self.arms,(-1,1)), return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-3)


    def update_from_batch(self, pulled_arms, rewards, time):
        # Filling observations
        for pulled_arm, reward in zip(pulled_arms, rewards):
            # fill observations context:
            self.update_observations(pulled_arm, reward)

        # Setting time
        self.t += time

        # Updating model learner:
        if len(self.collected_rewards) > 1:
            self.gp.fit(self.pulled_arms_x,self.collected_rewards)
            self.means, self.sigmas = self.gp.predict(np.reshape(self.arms,(-1,1)), return_std=True)
            self.sigmas = np.maximum(self.sigmas, 1e-5)




