from GenericUserClass import GenericUserClass

import numpy as np


def exp_clicks(x, y_max, x_scale, offset):
    y = y_max * (1 - np.exp(-((x-offset)*(y_max/x_scale))))
    if y < 0:
        y=0
    return y

def exp_cost(x, click_function, scale):
    return click_function(x) * scale * x**0.5     # The 0.8 makes the cost per click decrease with the nomber of clicks

def conversion_func(price, min_price, drop_rate):
    return np.exp(-(price*drop_rate - min_price))


class ExpUserClass(GenericUserClass):
    def __init__(self, min_bid=0.05, max_clicks=10000, clicks_scaling=2000,
                 cost_scaling=10, 
                 price_values=[20, 22, 24, 26, 28], min_price=0, conversion_drop_rate=0.1, conversion_dict=None,
                 prop_dev_clicks=0.1, prop_dev_cost=0.1):
        clicks = lambda bid: exp_clicks(bid, max_clicks, clicks_scaling, min_bid)
        clicks_dev = lambda bid: clicks(bid) * prop_dev_clicks
        cost = lambda bid: exp_cost(bid, clicks, cost_scaling)
        cost_dev = lambda bid: cost(bid) * prop_dev_cost
        if conversion_dict==None:
            conversion_dict = {}
            for price in price_values:
                conversion_dict[price] = conversion_func(price, min_price, conversion_drop_rate)
        super().__init__(clicks, cost, conversion_dict, clicks_dev, cost_dev)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    clicks = lambda x: exp_clicks(x, 10000, 5000, 1)

    usr_class = ExpUserClass()

    bid = []
    y_clicks = []
    click_samples = []
    y_cost = []
    cost_samples = []
    for i in range(100):
        x = i/100
        bid.append(x)
        y_clicks.append(usr_class.click_function(x))
        click_samples.append(usr_class.n_clicks(x))
        y_cost.append(usr_class.cost_function(x))
        cost_samples.append(usr_class.cum_click_cost(x))

    price = []
    conversion = []
    for value in usr_class.conversion:
        price.append(value)
        conversion.append(usr_class.conversion[value])

    
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))

    axs[0].plot(bid, y_clicks)
    axs[0].plot(bid, click_samples, "o")
    axs[0].set_title("n_clicks")
    axs[1].plot(bid, y_cost)
    axs[1].plot(bid, cost_samples, "o")
    axs[1].set_title("cum_cost")
    axs[2].bar(price, conversion)
    axs[2].set_title("conversion")

    plt.show()

    