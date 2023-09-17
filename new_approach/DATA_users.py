import numpy as np
from UserClass import *


#### User class C1 ####
F1 = 0
F2 = 0

def click_f(bid):
    bid_0 = 2
    bid_1 = 6
    max_clicks = 100

    # if bid < bid_0:
    #     return 0
    # elif bid < bid_1:
    #     return n_max * (bid - bid_0)/(bid_1-bid_0)
    # return n_max
    return (bid < bid_1) * (bid > bid_0) * (bid - bid_0)/(bid_1-bid_0) * max_clicks + (bid >= bid_1) * max_clicks


def cost_f(bid):
    return bid * 60


def conversion_rate_f(price):
    return (((np.exp(-0.05*price + 50))/np.exp(-0.05*50+50))*0.9)


C1 = UserClass(F1, F2, click_f, cost_f, conversion_rate_f)



#### User class C2 ####
F1 = 0
F2 = 1

def click_f(bid):
    bid_0 = 2
    bid_1 = 6
    max_clicks = 100

    # if bid < bid_0:
    #     return 0
    # elif bid < bid_1:
    #     return n_max * (bid - bid_0)/(bid_1-bid_0)
    # return n_max
    return (bid < bid_1) * (bid > bid_0) * (bid - bid_0)/(bid_1-bid_0) * max_clicks + (bid >= bid_1) * max_clicks


def cost_f(bid):
    return bid * 60


def conversion_rate_f(price):
    return (((np.exp(-0.1*price + 50))/np.exp(-0.1*50+50))*0.9)


C2 = UserClass(F1, F2, click_f, cost_f, conversion_rate_f)



#### User class C3 ####
F1 = 1
F2 = 0

def click_f(bid):
    bid_0 = 2
    bid_1 = 6
    max_clicks = 100

    # if bid < bid_0:
    #     return 0
    # elif bid < bid_1:
    #     return n_max * (bid - bid_0)/(bid_1-bid_0)
    # return n_max
    return (bid < bid_1) * (bid > bid_0) * (bid - bid_0)/(bid_1-bid_0) * max_clicks + (bid >= bid_1) * max_clicks


def cost_f(bid):
    return bid * 60


def conversion_rate_f(price):
    return ((((-np.power((price - 70), 4))/(np.power((50-70), 4))+1.2))*0.75)


C3 = UserClass(F1, F2, click_f, cost_f, conversion_rate_f)



#### User class C1 for step 5 ####
F1 = 1
F2 = 0

def click_f(bid):
    bid_0 = 2
    bid_1 = 6
    max_clicks = 100

    # if bid < bid_0:
    #     return 0
    # elif bid < bid_1:
    #     return n_max * (bid - bid_0)/(bid_1-bid_0)
    # return n_max
    return (bid < bid_1) * (bid > bid_0) * (bid - bid_0)/(bid_1-bid_0) * max_clicks + (bid >= bid_1) * max_clicks


def cost_f(bid):
    return bid * 60

def conversion_rate_f(price):
    return np.array([(np.exp(-0.1*price + 50))/np.exp(-0.1*50+50)*0.9, \
                    ((-np.power((price - 70), 4))/(np.power((50-70), 4))+1.2)*0.75, \
                    (np.exp(-0.1*price + 50))/np.exp(-0.1*50+50)*0.9])

'''def conversion_rate_f(price):
    return np.array([(np.exp(-0.1*price + 50))/np.exp(-0.1*50+50)*0.9, \
                    ((-np.power((price - 70), 4))/(np.power((50-70), 4))+1.2)*0.75, \
                    (np.exp(-0.1*price + 50))/np.exp(-0.1*50+50)*0.9])'''


C1_NS = UserClass(F1, F2, click_f, cost_f, conversion_rate_f)


