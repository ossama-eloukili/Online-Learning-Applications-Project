from abc import ABC, abstractmethod

class UserClass(ABC):
    @abstractmethod
    def n_clicks(self, bid):
        pass
    
    @abstractmethod
    def cum_click_cost(self, bid):
        pass

    @abstractmethod
    def pull_user(self, price, n_users=1):
        pass