import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from ipywidgets import interactive


def heaviside(z):
    """Heaviside step function."""
    a = np.where(z >= 0, 1, 0)
    return a


class Perceptron:
    """Perceptron superclass.
    """
    def __init__(self):
        self.param_range = (-10.0, 10.0)
        self.params = {'weight1': self.param_range,
                       'weight2': self.param_range,
                       'bias': self.param_range,
                       }
        self.main_title = {'t': f'Interactive {str(self)}',
                           'fontsize': 'xx-large',
                           'ha': 'center',
                           }
        self.figure_params = {'num': None,
                              'figsize': (5, 5),
                              'dpi': 100,
                              'facecolor': 'w',
                              'edgecolor': None,
                              }
        self.title = {'color': 'w',
                      'fontsize': 'large',
                      'verticalalignment': 'top',
                      }
        self.WRONG = 'Don\'t Give Up! You Can Do It!'
        self.CORRECT = 'Nice! You Did It!'
        self.RESOLUTION = 200
        self.X1_RANGE = [-0.05, 1.2]
        self.X2_RANGE = [-0.05, 1.2]
        self.XTICKS = [0, 1]
        self.YTICKS = [0, 1]
        self.LEVELS = np.linspace(0, 1, 3)  # number of colormap levels
        self.CMAP = cm.RdBu  # color map

        # Test points:
        self.x_test = NotImplementedError  # expected to be [(x1, x2), ... ]
        self.y_test = NotImplementedError  # expected to be [y, ...]

    def forward(self, x):
        """Forward propagation."""
        z = x @ self.W.T + self.b
        y_hat = heaviside(z)
        return y_hat

    def __call__(self, print_help=False):
        self.interactive_plot = interactive(self.plot, **self.params)
        self.output = self.interactive_plot.children[-1]
        self.output.layout.height = '500px'
        if print_help:
            print(self.__doc__)
        return self.interactive_plot

    def plot(self, **kwargs):
        self.W = np.array([kwargs['weight1'], kwargs['weight2']])
        self.b = kwargs['bias']
        self.plot_surface()
        self.plot_test_points()
        self.adjust_plots()

    def plot_surface(self):
        X1 = np.linspace(self.X1_RANGE[0], self.X1_RANGE[1], self.RESOLUTION)
        X2 = np.linspace(self.X2_RANGE[0], self.X2_RANGE[1], self.RESOLUTION)
        XX1, XX2 = np.meshgrid(X1, X2)
        X = np.dstack([XX1, XX2])
        Y = self.forward(X)

        plt.figure(**self.figure_params)
        plt.suptitle(**self.main_title)

        ax = plt.contourf(XX1, XX2, Y, cmap=self.CMAP, levels=self.LEVELS)
        # plt.colorbar(ax, orientation='vertical')  # add colorbar

    def plot_test_points(self):
        y_hat = []

        for i, j in zip(self.x_test, self.y_test):
            y_hat_i = int(self.forward(i))
            y_hat.append(y_hat_i)
            c = 'b' if j else 'r'
            plot = plt.plot([i[0]], [i[1]], marker='o', markersize=10, color=c)

        result, c = (self.CORRECT, 'g') if (y_hat == self.y_test) else (self.WRONG, 'r')

        plt.title(f'{result}', backgroundcolor=c, **self.title)

    def adjust_plots(self):
        plt.xticks(self.XTICKS)
        plt.yticks(self.YTICKS)
        plt.xlim(self.X1_RANGE)
        plt.ylim(self.X2_RANGE)

    def __str__(self):
        return ' P'.join(self.__class__.__name__.split('P'))


class ANDPerceptron(Perceptron):
    """Interactive AND Perceptron."""
    def __init__(self):
        super().__init__()
        self.x_test = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.y_test = [0, 0, 0, 1]


class ORPerceptron(Perceptron):
    """Interactive OR Perceptron."""
    def __init__(self):
        super().__init__()
        self.x_test = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.y_test = [0, 1, 1, 1]


class NOTPerceptron(Perceptron):
    """Interactive NOT Perceptron. Ignores The First Input."""
    def __init__(self):
        super().__init__()
        self.params['weight1'] = (0, 0)
        self.x_test = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.y_test = [1, 0, 1, 0]


class XORPerceptron(Perceptron):
    """Interactive XOR Perceptron."""
    def __init__(self):
        super().__init__()
        self.WRONG_TITLE = 'Spoiler! You Will Never Solve It!'
        self.x_test = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.y_test = [0, 1, 1, 0]


class XOR2LayerPerceptron(Perceptron):
    """XOR Multi-Layer Perceptron.
    l1n1_w1: Layer1 Node1 Weight1
    l1n1_w2: Layer1 Node1 Weight2
    l1n1_b: Layer1 Node1 Bias
    l1n2_w1: Layer1 Node2 Weight1
    l1n2_w2: Layer1 Node2 Weight2
    l1n2_b: Layer1 Node2 Bias
    l2_w1: Layer2 Weight1
    l2_w2: Layer2 Weight2
    l2_b: Layer2 Bias
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
        self.y_test = [0, 1, 1, 0]

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
