from UserClass import UserClass

import numpy as np


def fixed_dev(x):
        return 1


class GenericUserClass(UserClass):
    def __init__(self, click_function, cost_function, conversion, click_dev=fixed_dev, cost_dev=fixed_dev):
        self.click_function = click_function
        self.click_dev = click_dev
        self.cost_function = cost_function
        self.cost_dev = cost_dev
        self.conversion = conversion

    def n_clicks(self, bid, noise=True):
        pull = self.click_function(bid)
        if noise:
            pull += np.random.normal(0, self.click_dev(bid))
        if pull > 0:
            return int(pull)
        else:
            return 0
        
    def cum_click_cost(self, bid, noise=True):
        pull = self.cost_function(bid)
        if noise:
            pull += np.random.normal(0, self.cost_dev(bid))
        if pull>0:
            return pull
        else:
            return 0

    def pull_user(self, price, n_users=1):
        try:
            p = self.conversion[price]
            return np.random.binomial(n_users, p)
        except:
            raise ValueError("The price is not one of the available values")
        
    def clarvoyant_solution(self, prod_cost, bid_range):
        profit_per_click = {}
        for price in self.conversion:
            profit_per_click[price] = self.conversion[price] * (price - prod_cost)
        best_price = max(profit_per_click, key=profit_per_click.get)

        reward = {}
        for bid in bid_range:
            reward[bid] = profit_per_click[best_price] * self.click_function(bid) - self.cost_function(bid)
        best_bid = max(reward, key=reward.get)

        return best_price, best_bid


# Testing for the class
if __name__ == "__main__":
    def clicks(x):
        return 10
    
    def dev(x):
        return 0

    def cost(x):
        return x + 10
    
    conversion = {
        3 : 1,
        4 : 0.5
    }

    usr_class = GenericUserClass(clicks, cost, conversion, dev, dev)

    for i in range(10):
        print(usr_class.n_clicks(i), usr_class.cum_click_cost(i))

    for i in range(10):
        print(usr_class.pull_user(3, 100), usr_class.pull_user(4, 100))

    print(usr_class.pull_user(5))   # This outputs an error