import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

f = open('exam.txt','r')
text = f.read()
X = [[float(x.split(',')[0]), float(x.split(',')[1])] for x in text.split('\n')]
Y = [float(x.split(',')[2]) for x in text.split('\n')]
f.close()

X = np.array(X)
Y = np.array(Y)

X_train, X_test,Y_train, Y_test = train_test_split(X,Y, test_size=0.33, random_state = 42)

model = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None,
                                        random_state=None, solver='liblinear', max_iter=100, multi_class='ovr',
                                         verbose=0, warm_start=False, n_jobs=1)
model.fit(X_train,Y_train)

print(accuracy_score(Y_test, model.predict(X_test)))
coef =  model.coef_[0]


X_1 = []
X_2 = []
for i in range(len(Y)):
    if Y[i] == 2:
        X_1.append(X[i])
    else:
        X_2.append(X[i])
X_1 = np.array(X_1)
X_2 = np.array(X_2)
#
plt.scatter(X_1[:,0], X_1[:,1], color="red")
plt.scatter(X_2[:,0], X_2[:,1], color="blue")
plt.show()

new_point = [[0, 1, 2]]
model.predict(new_point)


#
# plt.scatter(X[:,0],X[:,1])
# plt.plot(X[:,2],hyp_(X, theta))
# plt.show()
