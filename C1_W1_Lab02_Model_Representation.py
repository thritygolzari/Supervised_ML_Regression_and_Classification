import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')

x_train = np.array([1.0, 2.0]) # x_train is the input in 1000 sqft
y_train = np.array([300.0, 500.0]) # y_train is the output in 1000 dollars

print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

# use .shape to find m
print(f"x_train.shape: {x_train.shape}") # returns tuple with entry of each dimension
m = x_train.shape[0] # gives length of array
print(f"Number of training examples is: {m}")

# use len() to find m
m = len(x_train)
print(f"Number of training examples is: {m}")

# indexing training examples
i = 0 

x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")

# plotting the data
plt.scatter(x_train, y_train, marker='x', c='r') # plot points, c (color of markers)
plt.title("Housing Prices") # set title
plt.ylabel('Price (in 1000s of dollars)') # set y axis label
plt.xlabel('Size (1000 sqft)') # set x axis label
plt.show()

# f(x) = wx + b
w = 200
b = 100
print(f"w: {w}")
print(f"b: {b}")

def compute_model_output(x, w, b):
    # computes the prediction of a linear model arguments
    m = x.shape[0]
    f_wb = np.zeros(m) # creates numpy array with size m
    for i in range(m):
        f_wb[i] = x[i] * w + b
    return f_wb # returns array with predicted points

# using f_wb to plot predicted points now
tmp_f_wb = compute_model_output(x_train, w, b,)

# plot our model prediction
plt.plot(x_train, tmp_f_wb, c="b", label='Our Prediction') # .plot = line

plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values') # plot points, c (color of markers)
plt.title("Housing Prices") # set title
plt.ylabel('Price (in 1000s of dollars)') # set y axis label
plt.xlabel('Size (1000 sqft)') # set x axis label
plt.legend()
plt.show()

# now that w and b are set, predict the price of a house with 1200 sqft
# w = 200
# b = 100
x_i = 1.2
cost_1200sqft = w * x_i + b
print(f"${cost_1200sqft} thousand dollars")