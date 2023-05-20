from UserClass import UserClass

import numpy as np


class GenericUserClass(UserClass):
    def __init__(self, click_function, cost_function, conversion, click_dev=1, cost_dev=1):
        self.click_function = click_function
        self.click_dev = click_dev
        self.cost_function = cost_function
        self.cost_dev = cost_dev
        self.conversion = conversion

    def n_clicks(self, bid):
        pull = self.click_function(bid) + np.random.normal(0, self.click_dev)
        if pull > 0:
            return int(pull)
        else:
            return 0
        
    def cum_click_cost(self, bid):
        pull = self.cost_function(bid) + np.random.normal(0, self.cost_dev)
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


# Testing for the class
if __name__ == "__main__":
    def clicks(x):
        return 10

    def cost(x):
        return x + 10
    
    conversion = {
        3 : 1,
        4 : 0.5
    }

    usr_class = GenericUserClass(clicks, cost, conversion, 1, 1)

    for i in range(10):
        print(usr_class.n_clicks(i), usr_class.cum_click_cost(i))

    for i in range(10):
        print(usr_class.pull_user(3, 10), usr_class.pull_user(4, 10))

    print(usr_class.pull_user(5))   # This outputs an error