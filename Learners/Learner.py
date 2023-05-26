from abc import ABC, abstractmethod

class Learner(ABC):
    @abstractmethod
    def pull_arm(self):
        """
        returns the value of the arm to pull
        """
        pass

    @abstractmethod
    def update(self, arm, reward):
        """
        updates the values given the reward and the arm pulled
        """
        pass

    @abstractmethod
    def get_values(self):
        """
        returns a dictionary with the values for each arm
        """
        pass