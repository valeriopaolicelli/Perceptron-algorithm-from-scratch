# to see the detail of the breast-cancer dataset, see here https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

def get_data(filetrain, filetest):
    data = np.genfromtxt(filetrain, delimiter=",", skip_header=1)

    X = data[:, :-1]
    y = data[:, -1]

    data = np.genfromtxt(filetest, delimiter=",", skip_header=1)

    X_test = data[:, :-1]
    y_test = data[:, -1]

    print(X)
    
    return X, y, X_test, y_test

X, y, X_test, y_test = get_data("breast-train.csv", "breast-test.csv")