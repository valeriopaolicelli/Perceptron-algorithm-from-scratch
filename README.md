# Perceptron-algorithm-from-scratch
Classification task solved by means of the perceptron algorithm in python language, by using only the numpy library.
There is one dataset about cancer/healthy patients, already splitted in two .cvs file, to train (breast-train.csv) and test (breast-test.csv) the perceptron.
The file get_data.py allows to import the data, throw a fuction that receives the file names of the train and test set, and returns:
  - matrix X with the samples of the train set
  - array y with the grounth-truth of the train set
  - matrix X_test with the samples of the test set, to evaluate the model
  - array y_test containing the ground-truth of the test set
This is only a 'toy-example' where the several library offered by python are not allowed.
The core of the repo is inside the sol.py file, where the get_data.py is invoked, and the different functions to perform the model are explained and developed.
There are three applicable kernels:
  - linear
  - gaussian-rbf
  - laplacian-rbf
The programmer can choose the proper one, by commenting and decommenting the proper lines of code.
