import numpy as np

from perceptrons.perceptron import Perceptron
from perceptrons.activations import heaviside


class ANDPerceptron(Perceptron):
    """Interactive AND Perceptron."""
    def __init__(self):
        super().__init__()
        self.x_test = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.y_test  = [0, 0, 0, 1]


class ORPerceptron(Perceptron):
    """Interactive OR Perceptron."""
    def __init__(self):
        super().__init__()
        self.x_test = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.y_test  = [0, 1, 1, 1]


class NOTPerceptron(Perceptron):
    """Interactive NOT Perceptron. Ignores The First Input."""
    def __init__(self):
        super().__init__()
        self.params['weight1'] = (0, 0)
        self.x_test = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.y_test  = [1, 0, 1, 0]


class XORPerceptron(Perceptron):
    """Interactive XOR Perceptron."""
    def __init__(self):
        super().__init__()
        self.WRONG_TITLE = 'Spoiler! You Will Never Solve It!'
        self.x_test = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.y_test  = [0, 1, 1, 0]


class XOR2LayerPerceptron(Perceptron):
    """XOR Multi-Layer Perceptron.
    l1n1_w1: Layer1 Node1 Weight1
    l1n1_w2: Layer1 Node1 Weight2
    l1n1_b:  Layer1 Node1 Bias
    l1n2_w1: Layer1 Node2 Weight1
    l1n2_w2: Layer1 Node2 Weight2
    l1n2_b:  Layer1 Node2 Bias
    l2_w1: Layer2 Weight1
    l2_w2: Layer2 Weight2
    l2_b:  Layer2 Bias
   """

    def __init__(self):
        super().__init__()
        self.params = {'l1n1_w1': self.param_range,
                       'l1n1_w2': self.param_range,
                       'l1n1_b': self.param_range,
                       'l1n2_w1': self.param_range,
                       'l1n2_w2': self.param_range,
                       'l1n2_b': self.param_range,
                       'l2_w1': self.param_range,
                       'l2_w2': self.param_range,
                       'l2_b': self.param_range,
                       }
        self.x_test = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.y_test  = [0, 1, 1, 0]

    def forward(self, X):
        """XOR Multi-Layer Perceptron forward."""
        l1n1 = X @ self.l1n1_W + self.l1n1_b  # Layer1 Node1
        l1n2 = X @ self.l1n2_W + self.l1n2_b  # Layer1 Node2
        l1n1 = heaviside(l1n1)  # Layer1 Node1
        l1n2 = heaviside(l1n2)  # Layer1 Node2
        l1 = np.dstack([l1n1, l1n2])  # Layer1 output
        l2 = l1 @ self.l2_W + self.l2_b  # Layer2
        l2 = heaviside(l2)  # Layer2 (Output) Single Node
        return l2

    def plot(self, **kwargs):
        self.l1n1_W = np.array([kwargs['l1n1_w1'], kwargs['l1n1_w2']])
        self.l1n1_b = kwargs['l1n1_b']
        self.l1n2_W = np.array([kwargs['l1n2_w1'], kwargs['l1n2_w2']])
        self.l1n2_b = kwargs['l1n2_b']
        self.l2_W = np.array([kwargs['l2_w1'], kwargs['l2_w2']])
        self.l2_b = kwargs['l2_b']
        self.plot_surface()
        self.plot_test_points()
        self.adjust_plots()

    def __str__(self):
        return ' 2'.join(self.__class__.__name__.split('2'))
