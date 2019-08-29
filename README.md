# Perceptron-algorithm-from-scratch
Classification task solved by means of the perceptron algorithm in python language, by using only the numpy library.
There is one dataset about cancer/healthy patients, already splitted in two .cvs file, to train (breast-train.csv) and test (breast-test.csv) the perceptron.
The file get_data.py allows to import the data, throw a fuction that receives the file names of the train and test set, and returns:
  - matrix X with the samples of the train set
  - array y with the grounth-truth of the train set
  - matrix X_test with the samples of the test set, to evaluate the model
  - array y_test containing the ground-truth of the test set
This is only a 'toy-example' where the several library offered by python are not allowed.
