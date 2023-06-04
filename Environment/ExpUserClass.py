from .GenericUserClass import GenericUserClass

import numpy as np


def exp_clicks(x, y_max, x_scale, offset):
    y = y_max * (1 - np.exp(-((x-offset)*(y_max/x_scale))))
    if y < 0:
        y=0
    return y

def exp_cost(x, click_function, scale):
    return click_function(x) * scale * x**0.5

def conversion_func(price, min_price, drop_rate):
    return np.exp(-((price - min_price)*drop_rate))


class ExpUserClass(GenericUserClass):
    """
    This UserClass uses an exponential function to medel the internal clicks, cost and conversion curves.
    The noise added to the curves has proportional std deviation to the y value of the function.
    It has multiple parameters to control the shape of these curves.
    All the parameters have default values, so it's not necessary to use all of them, just set the ones that are needed.

    ### parameters for the n_clicks curve:
    - min_bid: The minimum bid needed to start winning any auction and start getting clicks;
    - max_clicks: The asymptotic maximum number of clicks obtainable with infinite bid;
    - clicks_scaling: A scaling factor that models how fast number of clicks rises as the bid varies;

    ### parameters for the cost curve:
    - cost_scaling: a parameter that controls how fast the cost rises as the bid increases;

    ### parameters for the conversion:
    - price_values: the valid values for the price of the product;
    - min_price: a theoretical price where the conversion rate is 100%;
    - conversion_drop_rate: this parameter controls how fast the conversion % drops as the price increases,
                            higher values make the function drop faster;
    - conversion_dict: instead of giving the last parameters one can directly imput a dictionary with custom prices 
                        and conversions for each, if this is given the other parameters for the conversion are ignored;

    ### noise parameters:
    - prop_dev_clicks: the standard deviation of the gaussian noise added to the clicks curve;
    - prop_dev_cost: the standard deviation of the gaussian noise added to the cost curve;
    """
    def __init__(self, min_bid=0.05, max_clicks=10000, clicks_scaling=2000,
                 cost_scaling=0.1, 
                 price_values=[20, 22, 24, 26, 28], min_price=5, conversion_drop_rate=0.2, conversion_dict=None,
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
    prod_cost = 18
    bid_range = np.arange(0, 1, 0.01)

    bid = []
    y_clicks = []
    click_samples = []
    y_cost = []
    cost_samples = []
    exp_reward = {}
    for price in usr_class.conversion:
        exp_reward[price] = []
    for x in bid_range:
        bid.append(x)
        clicks = usr_class.click_function(x)
        y_clicks.append(clicks)
        click_samples.append(usr_class.n_clicks(x))
        cost = usr_class.cost_function(x)
        y_cost.append(cost)
        cost_samples.append(usr_class.cum_click_cost(x))
        for price in usr_class.conversion:
            reward = (price - prod_cost) * usr_class.conversion[price] * clicks - cost
            exp_reward[price].append(reward)

    price = []
    conversion = []
    for value in usr_class.conversion:
        price.append(value)
        conversion.append(usr_class.conversion[value])

    best_price, best_bid = usr_class.clarvoyant_solution(18, bid_range)

    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()

    axs[0].plot(bid, click_samples, "o")
    axs[0].plot(bid, y_clicks)
    axs[0].set_title("n_clicks")
    axs[1].plot(bid, cost_samples, "o")
    axs[1].plot(bid, y_cost)
    axs[1].set_title("cum_cost")
    axs[2].bar(price, conversion)
    axs[2].set_title("conversion")
    for price in exp_reward:
        axs[3].plot(bid, exp_reward[price], label=price)
    axs[3].axvline(best_bid)
    axs[3].set_title("expected reward")
    axs[3].legend()

    plt.show()

    