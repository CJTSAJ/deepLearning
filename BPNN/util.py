import random
import numpy as np

class util:
    def __init__(self):
        pass

    def rand(self, a, b):
        return (b - a) * random.random() + a

    def logistic(self, x):
        return 1.0 / (1 + np.exp(-x))

    def logistic_derivative(self, x):
        return x * (1 - x)

    def tanh(self, x):
        return 2 * self.logistic(2*x) - 1

    def tanh_derivative(self, x):
        return (1 + x)*(1 - x)