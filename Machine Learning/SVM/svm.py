from sklearn.metrics import accuracy_score
import numpy as np
from statistics import mode
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
def predict_class(X,Y):
    return true

f = open('cancer.data','r')
text = f.read()
f.close()
lines = text.split('\n')
lines = lines[:-1]
X = []
Y = []
for x in text.split('\n'):
    try:
        X.append([float(x.split(',')[1]), float(x.split(',')[2]), float(x.split(',')[3]), float(x.split(',')[4]), float(x.split(',')[5]), float(x.split(',')[6]), float(x.split(',')[7]), float(x.split(',')[8]), float(x.split(',')[9]) ])
        Y.append(float(x.split(',')[10]))
    except:
        pass
X = np.array(X)
Y = np.array(Y)


X_train, X_test,Y_train, Y_test = train_test_split(X,Y, test_size=0.33, random_state = 42)
# Y_pred = predict_class(X_train, Y_train)
# print(accuracy_score(Y_test, Y_pred))
clf = svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf.fit(X_train, Y_train)


clf_2 = linear_model.LogisticRegression()
clf_2.fit(X_train, Y_train)

clf_3 = KNeighborsClassifier(n_neighbors=3)
clf_3.fit(X_train,Y_train)

model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
a = model.fit_transform(X)
X_1 = []
X_2 = []
for i in range(len(Y)):
    if Y[i] == 2:
        X_1.append(a[i])
    else:
        X_2.append(a[i])
X_1 = np.array(X_1)
X_2 = np.array(X_2)
#
plt.scatter(X_1[:,0], X_1[:,1], color="red")
plt.scatter(X_2[:,0], X_2[:,1], color="blue")
plt.show()

print(accuracy_score(Y_test, clf.predict(X_test)))
print(accuracy_score(Y_test, clf_2.predict(X_test)))
print(accuracy_score(Y_test, clf_3.predict(X_test)))

#
# a_train, a_test, y_train, y_test = train_test_split(a, Y, test_size=0.33, random_state = 42)
#
#
# clf_4 = svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
#     max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=False)
# clf_4.fit(a_train, y_train)
# print(accuracy_score(y_test, clf_4.predict(a_test)))
#

# print(X)
# print(clf.support_vectors_)
# for vector in clf.support_vectors_:
#     print(vector)
#     plt.plot(vector[0], vector[1])

# plt.show()
