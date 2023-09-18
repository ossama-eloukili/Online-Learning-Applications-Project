import numpy as np


class PricingEnvironment:
    def __init__(self, prices, margin_param, user_class):
        """
        PARAMETERS:
            prices: list or array of numbers
                it is the set of prices we will use, since we are discretizing the problem
                eg: an array of 5

            user_calss: UserClass type

        OBS:
            - we have changed the margin -> self.prices[pulled_arm] - self.margin_param
            - other than the reward, we are returning the converted_clicks too
            - name change: convertion_probabilities -> convertion_rates
        """
        self.n_arms = len(prices)
        self.prices = prices
        self.margin_param = margin_param
        # self.user_class = user_class
        self.convertion_rates = [user_class.conversion_rate_function(p) for p in prices]


    def round(self, pulled_arm, n_daily_clicks, cum_daily_costs):
        alpha = self.convertion_rates[pulled_arm]
        converted_clicks = np.random.binomial(n_daily_clicks, alpha)
        reward =  converted_clicks * (self.prices[pulled_arm] - self.margin_param) - cum_daily_costs
        return reward
    
    def round_step3(self, pulled_arm, n_daily_clicks, cum_daily_costs):
        alpha = self.convertion_rates[pulled_arm]
        converted_clicks = np.random.binomial(n_daily_clicks, alpha)
        reward =  converted_clicks * (self.prices[pulled_arm] - self.margin_param) - cum_daily_costs
        return converted_clicks, reward
    



class BiddingEnvironment:#or maybe call it AdvertisingEnvironment
    def __init__(self, bids, sigma_clicks, sigma_costs, user_class):
        """
        n_arms: integer

        prices: array
            it is the set of prices we will use, since we are discretizing the problem
            eg: an array of 5
        """
        self.bids = bids
        # self.n_arms = len(bids)
        # self.user_class = user_class

        self.mean_clicks = [user_class.n_daily_clicks_function(b) for b in bids]
        self.mean_costs = [user_class.cum_daily_costs_function(b) for b in bids]

        self.sigmas_clicks = np.ones(len(bids)) * sigma_clicks
        self.sigmas_costs = np.ones(len(bids)) * sigma_costs


    def round(self, pulled_arm):
        n_daily_clicks = np.random.normal(self.mean_clicks[pulled_arm], self.sigmas_clicks[pulled_arm])
        n_daily_clicks = max(int(n_daily_clicks), 0)#because non integer or negative number of clicks has no sense
        
        cum_daily_costs = np.random.normal(self.mean_costs[pulled_arm], self.sigmas_costs[pulled_arm])

        return n_daily_clicks, cum_daily_costs