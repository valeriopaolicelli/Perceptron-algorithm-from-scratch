#%% imports
import numpy as np
import matplotlib.pyplot as plt
from get_data import get_data

#%% functions

# Function to normalize the data
def normalize_data(x):
    mean= np.mean(x, axis= 0)
    std= np.std(x, axis= 0)
    x = (x - mean) / std
    return x

# Function to predict value, given x, y, w and b
def predict(x, w, b):
    # if <w,x> if greater than b -> 1, otherwise -> 0
    f = np.dot(w, x)
    if f >= b:
        return 1
    else:
        return 0

# Function to train the model
def train_perceptron(x, y, epochs, lr):
    # initialize the bias parameter and the weights
    b = 0
    n_features = X_train.shape[1]
    w = np.zeros((n_features))

    #for all epochs
    for epoch in range(epochs):
        print("-> Epoch: %d" % (epoch+1))
        for i in range(len(x)):                      
            y_pred = predict(x[i], w, b)

            # updating the weights
            for k in range(len(w)):             
                w[k] = w[k] + lr * (y[i]-y_pred) * x[i][k]
                b = b + lr * (y[i]-y_pred)
    return (w, b)

def linear_kernel(x, xi):
    # x is the vector of weights
    # xi is the sample
    return np.dot(x, xi)

def gaussian_rbf_kernel(x, xi):
    # x is the vector of weights
    # xi is the sample
    std = np.std(x)
    norm = np.linalg.norm(x - xi)
    k = np.exp((-1) * norm**2 / (2 * std))
    return k

def laplacian_rbf_kernel(x, xi):
    # x is the vector of weights
    # xi is the sample
    std = np.std(x)
    norm = np.linalg.norm(x - xi)
    k = np.exp((-1) * norm / (2 * std))
    return k

# Function to test the model
def test_perceptron(x, y, w, b):
    y_pred = []
    n_correct = 0
    n_sample = x.shape[0]
    for i in range(n_sample):
        f = linear_kernel(w, x[i])
        #f = laplacian_rbf_kernel(w, x[i])
        #f = gaussian_rbf_kernel(w, x[i])

        # activation function
        if f >= b:                               
            yhat = 1                             
        else:                                   
            yhat = 0
        if yhat == y[i]:
            n_correct += 1
        
        y_pred.append(yhat)
    accuracy = 100 * n_correct/n_sample
    print("Accuracy: %.2f%%" % accuracy)
    return accuracy

#%% main
############ MAIN ##############

# Retrieve data
path_train = "./breast-train.csv"
path_test = "./breast-test.csv"
X_train, y_train, X_test, y_test = get_data(path_train, path_test)

# Normalize samples
# X_train = normalize_data(X_train)
# X_test = normalize_data(X_test)

#%% Train
# Given the training set (data + labels), train the model.
# It returns the weights and bias parameter, that classify correctly all train data samples
print( "Training perceptron... wait please")
w, b = train_perceptron(X_train, y_train, epochs= 1000, lr= 10**-9)

print("Weights:")
print(w) # only to see the weights calculated

#%% Test on trainset
#print("Accuracy trainset:")
#test_perceptron(X_train, y_train, w, b)

#%% Test on testset
# Now, test the model with new data
print("Accuracy testset:")
test_perceptron(X_test, y_test, w, b)