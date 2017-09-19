import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
f = open('exam.txt','r')
text = f.read()
X = [[float(x.split(',')[0]), float(x.split(',')[1])] for x in text.split('\n')]
Y = [float(x.split(',')[2]) for x in text.split('\n')]
f.close()

X = np.array(X)
Y = np.array(Y)
# X = preprocessing.normalize(X, norm='l2')

X_train, X_test,Y_train, Y_test = train_test_split(X,Y, test_size=0.33, random_state = 42)


model = clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(10, 15), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=3, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)

model.fit(X_train,Y_train)

print(accuracy_score(Y_test, model.predict(X_test)))
coef =  model.coefs_[0]


X_1 = []
X_2 = []
for i in range(len(Y)):
    if Y[i] == 1:
        X_1.append(X[i])
    else:
        X_2.append(X[i])
X_1 = np.array(X_1)
X_2 = np.array(X_2)
#
#
# plt.scatter(X_1[:,0], X_1[:,1], color="red")
# plt.scatter(X_2[:,0], X_2[:,1], color="blue")
# plt.show()

new_point = [[0, 1]]
model.predict(new_point)


#
# plt.scatter(X[:,0],X[:,1])
# plt.plot(X[:,2],hyp_(X, theta))
# plt.show()
