import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from ipywidgets import fixed, interactive
from IPython.display import display

##########################################
# CONFIGS
##########################################

PARAM_RANGE = (-10.0, 10.0)  # range value for weights and biases
RESOLUTION = 100  # number of points for surface plot (a kind of resolution)
X1_RANGE = [-0.25, 1.25]  # axis range
X2_RANGE = [-0.25, 1.25]  # axis range
FIGURE_PARAMS = {'num': None,
                 'figsize': (5, 5),
                 'dpi': 100,
                 'facecolor': 'w',
                 'edgecolor': None,
                 }
MAIN_TITLE = {'fontsize': 'xx-large',
              'ha': 'center',
              }
CORRECT_COLOR = 'g'
WRONG_COLOR = 'r'
ZEROS_COLOR = 'r'
ONES_COLOR = 'b'
LEVELS = np.linspace(0, 1, 3)  # number of colormap levels
CMAP = cm.RdBu  # color mapp; other candidates: bwr_r seismic_r
POINT_SIZE = 200  # size of the test points
EDGECOLORS = 'w'  # test points edge color (white)
TITLE = {'color': 'w',
         'fontsize': 'large',
         'verticalalignment': 'top',
         }
WRONG = 'Don\'t Give Up! You Can Do It!'  # title for wrong solution
CORRECT = 'Nice! You Did It!'  # title for correct solution
XTICKS = [0, 1]  # ticks for x axis
YTICKS = [0, 1]  # ticks for y axis
# titles, x test values and y ground truth values used for perceptrons:
ANDCONFIG = ['ANDPerceptron', [(0, 0), (0, 1), (1, 0), (1, 1)], [0, 0, 0, 1]]
ORCONFIG = ['ORPerceptron', [(0, 0), (0, 1), (1, 0), (1, 1)], [0, 1, 1, 1]]
NOTCONFIG = ['NOTPerceptron', [(0, 0), (0, 1), (1, 0), (1, 1)], [1, 0, 1, 0]]
XORCONFIG = ['XORPerceptron', [(0, 0), (0, 1), (1, 0), (1, 1)], [0, 1, 1, 0]]
XOR2CONFIG = ['XOR2LayerPerceptron', XORCONFIG[1], XORCONFIG[2]]

##########################################
# All the functional stuff starts here
##########################################


def heaviside(z):
    """Heaviside step function."""
    a = np.where(z >= 0, 1, 0)
    return a


def neuron(x, w, b, activation=heaviside):
    """Simple forward propagation: linear + heaviside (by default).
    Inputs:
    x: input;
    w: weights;
    b: bias;
    activation: activation function (default: heaviside).
    """
    return activation(x @ w.T + b)


def transform():
    """Prepares input data for neuron."""
    X1 = np.linspace(X1_RANGE[0], X1_RANGE[1], RESOLUTION)
    X2 = np.linspace(X2_RANGE[0], X2_RANGE[1], RESOLUTION)
    XX1, XX2 = np.meshgrid(X1, X2)
    X = np.dstack([XX1, XX2])
    return X


def plot_results(x, yhat):
    """Simply plots the results.
    Inputs:
    x: input;
    yhat: predicted values of y"""
    plt.contourf(x[:, :, 0], x[:, :, 1], yhat, cmap=CMAP, levels=LEVELS)


def plot_test_points(x, y, wb, mlp=False) -> None:
    """Plots test points, showing the required solution (result).
    Inputs:
    x: input;
    y: ground truth values;
    wb: list of weights and bias(es);
    mlp: type of propagation (default: single neuron).
    """
    gt = list()
    for i, j in zip(x, y):
        gt_i = int(neuron(i, wb[0], wb[1])) if not mlp else int(propagate_mlp(i, wb))
        gt.append(gt_i)
        c = ONES_COLOR if j else ZEROS_COLOR
        plt.scatter([i[0]], [i[1]], s=POINT_SIZE, edgecolors=EDGECOLORS, c=c)

    result, c = (CORRECT, CORRECT_COLOR) if (gt == y) else (WRONG, WRONG_COLOR)
    TITLE['label'] = f'{result}'  # sets the corrct title
    TITLE['backgroundcolor'] = c  # sets the right title color
    plt.title(**TITLE)


def prepare_plot(t: str) -> None:
    """Sets title, init fig, sets ticks and axis limits.
    Inputs:
    t: plot title.
    """
    # set the plot title
    MAIN_TITLE['t'] = f'  {t}'
    # prepare figure
    plt.figure(**FIGURE_PARAMS)
    plt.suptitle(**MAIN_TITLE)
    # add ticks
    plt.xticks(XTICKS)
    plt.yticks(YTICKS)
    # set the axis limits
    plt.xlim(X1_RANGE)
    plt.ylim(X2_RANGE)


def plot(x, y, weight1, weight2, bias):
    """Propagates and plots the results for the simple neuron.
    Inputs:
    x: input;
    y: ground truth values;
    weight[], bias: weights and bias of the neuron.
    """
    w = np.array([weight1, weight2])
    X = transform()
    yhat = neuron(X, w, bias)

    plot_results(X, yhat)
    plot_test_points(x, y, [w, bias])


def run(t, x, y, weight1=PARAM_RANGE, weight2=PARAM_RANGE, bias=PARAM_RANGE):
    """This function will be interactive.
    Inputs:
    t: main title;
    x: input;
    y: ground truth values;
    weight[], bias: weights and bias of the neuron.
    """
    prepare_plot(t)
    plot(x, y, weight1, weight2, bias)


def perceptron(t: str, x, y, **kwargs):
    """Base function for single neuron perceptrons. Returns ipython widget.
    Inputs:
    t: main title;
    x: input;
    y: ground truth values.
    """
    return interactive(run, t=fixed(t), x=fixed(x), y=fixed(y), **kwargs)


def propagate_mlp(x, wb: list):
    """Forward propagation for 2LayerPerceptron.
    Inputs:
    x: input;
    wb: weights and biases values.
    """
    l1n1 = neuron(x, wb[0], wb[1])  # Layer 1 Neuron 1
    l1n2 = neuron(x, wb[2], wb[3])  # Layer 1 Neuron 2
    l1 = np.dstack([l1n1, l1n2])  # Layer 1 output
    l2 = neuron(l1, wb[4], wb[5])  # Layer 2 output
    return l2


def plot_mlp(x, y, wb: list):
    """Propagates and plot the mlp results.
    Inputs:
    x: input;
    y: ground truth values;
    wb: weights and biases values.
    """
    X = transform()
    yhat = propagate_mlp(X, wb)
    plot_results(X, yhat)
    plot_test_points(x, y, wb, mlp=True)


def run_mlp(t,
            x,
            y,
            l1n1_w1=PARAM_RANGE,
            l1n1_w2=PARAM_RANGE,
            l1n1_b=PARAM_RANGE,
            l1n2_w1=PARAM_RANGE,
            l1n2_w2=PARAM_RANGE,
            l1n2_b=PARAM_RANGE,
            l2_w1=PARAM_RANGE,
            l2_w2=PARAM_RANGE,
            l2_b=PARAM_RANGE):
    """This function will be interactive."""
    l1n1_W, l1n2_W = np.array([l1n1_w1, l1n1_w2]), np.array([l1n2_w1, l1n2_w2])
    l2_W = np.array([l2_w1, l2_w2])
    wb = [l1n1_W, l1n1_b, l1n2_W, l1n2_b, l2_W, l2_b]

    prepare_plot(t)
    plot_mlp(x, y, wb)


def mlp(t, x, y, **kwargs):
    """Base function for interactive MLPs."""
    return interactive(run_mlp, t=fixed(t), x=fixed(x), y=fixed(y), **kwargs)


##########################################
# Here's final functions for export :)
##########################################


def and_perceptron():
    """Interactive AND Perceptron."""
    display(perceptron(*ANDCONFIG))


def or_perceptron():
    """Interactive OR Perceptron."""
    display(perceptron(*ORCONFIG))


def not_perceptron():
    """Interactive NOT Perceptron."""
    display(perceptron(*NOTCONFIG, weight1=(0, 0)))


def xor_perceptron():
    """Unsolvable Interactive XOR Perceptron."""
    display(perceptron(*XORCONFIG))


def xor_mlp():
    """Interactive XOR 2-Layer Perceptron."""
    display(mlp(*XOR2CONFIG))
