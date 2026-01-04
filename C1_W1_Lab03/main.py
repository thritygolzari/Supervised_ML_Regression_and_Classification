import numpy as np
import matplotlib.pyplot as plt

from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
plt.style.use('./deeplearning.mplstyle')

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480, 430, 630, 730])

# computing cost
def compute_cost(x, y, w, b):
    # computes cost function for linear regression
    #returns total cost as float

    # number of training examples
    m = x.shape[0]

    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b # find predicted value
        cost = (f_wb - y[i]) ** 2 # squared cost
        cost_sum += cost
    total_cost = (1 / (2*m)) * cost_sum

    return total_cost

# simple model
plt_intuition(x_train, y_train)

# contour plot
plt.close('all')
fig, ax, dyn_items = plt_stationary(x_train, y_train)
updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)

# another 3d perspective
soup_bowl()