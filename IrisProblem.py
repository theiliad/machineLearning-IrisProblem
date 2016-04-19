import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from tabulate import tabulate

iris = load_iris()
test_idx = [0, 90, 60, 10, 45, 5, 70, 20, 100]

#training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis = 0)

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

test_data_finalArray = test_data.tolist()

accum = 0
print "The \"Species\" column in the table below, is guesses made for each row based on the other 4 columns of data for the flower\n"

for i in clf.predict(test_data):
    if (int(i) == 0):
        test_data_finalArray[accum].append("Satosa")
    elif (int(i) == 1):
        test_data_finalArray[accum].append("Versicolor")
    elif (int(i) == 2):
        test_data_finalArray[accum].append("Virginica")
    
    accum += 1

print tabulate(test_data_finalArray, headers=["Sepal Length","Sepal Width", "Petal Length", "Petal Width", "Species"])
