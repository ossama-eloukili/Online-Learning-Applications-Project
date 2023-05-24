import numpy as np
import matplotlib.pyplot as plt
from Environment import *
from TS_Learner import *

#Fixed parameters
n_arms = 5
T = 365

#Parameters to be set (temporary chosen randomly)
p = np.array([0.15, 0.1, 0.1, 0.35])
opt = p[3] #the last arm is the optimal one
n_daily_clicks = 1000
margin = 90
cum_daily_costs = 200


n_experiments = 1000
ts_rewards_per_experiment = []


for e in range(0, n_experiments):
    env = Environment(n_arms = n_arms, probabilities = p)
    ts_learner = TS_Learner(n_arms)
    for t in range(0, T):
        pulled_arm = ts_learner.pull_arm()
        reward = env.round(n_daily_clicks, margin, cum_daily_costs, pulled_arm)
        ts_learner.update(pulled_arm, reward)
        
    ts_rewards_per_experiment.append(ts_rewards_per_experiment)


#Plot Cumulative Regret
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis = 0)), 'r')
plt.legend(["TS"])
plt.show()




