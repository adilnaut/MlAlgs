from sklearn.metrics import accuracy_score
import numpy as np
from statistics import mode
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def return_class(str):
    if str == "Iris-setosa":
        return 1.0
    elif str == "Iris-versicolor":
        return 2.0
    elif str == "Iris-virginica":
        return 3.0

def predict_class(X,Y, x, num_neigh):
    distances = []
    for i in range(len(X)):
        sum = 0
        for j in range(len(x)):
            sum = sum + (X[i][j] - x[j])**2
        distances.append(sum**0.5)
    sort_i = np.argsort(distances)
    classes = []
    for arg in np.argsort(distances)[:num_neigh]:
        classes.append(Y[arg])
    label = mode(classes)
    return label

f = open('iris.data','r')
text = f.read()
lines = text.split('\n')
lines = lines[:-1]

f.close()

X = [[float(x.split(',')[0]), float(x.split(',')[1]),float(x.split(',')[2]), float(x.split(',')[3])] for x in lines]
Y = [return_class(x.split(',')[4]) for x in lines]

X_train, X_test,Y_train, Y_test = train_test_split(X,Y, test_size=0.33, random_state = 42)

Y_pred = [predict_class(X_train, Y_train, point, 3) for point in X_test]
print(accuracy_score(Y_test, Y_pred))

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train,Y_train)

print(accuracy_score(Y_test, clf.predict(X_test)))
