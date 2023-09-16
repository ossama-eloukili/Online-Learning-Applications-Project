class UserClass:
    """
    F1 and F2: in {0,1}
        are the binary features
    n_daily_clicks_function: it is a function
        represents the number of clicks given a bid
    cum_daily_costs: it is a function
        represents the cost given a bid
    conversion_rate: a function with image in [0,1] (com: usually descendent)
        it represent how a user is likely to buy the product given a price
        Obs:
        |   in the text says to consider just 5 prices, maybe we can consider this as an array instead of a function
        |__

    """

    def __init__(self, F1, F2, n_daily_clicks_function, cum_daily_costs_function, conversion_rate_function):
        self.F1 = F1
        self.F2 = F2
        self.n_daily_clicks_function = n_daily_clicks_function
        self.cum_daily_costs_function = cum_daily_costs_function
        self.conversion_rate_function = conversion_rate_function