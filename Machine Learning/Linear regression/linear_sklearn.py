import matplotlib.pyplot as mp
from sklearn import  linear_model
from sklearn.linear_model import Lasso


import numpy as np
f = open('data.txt','r')
text = f.read()
X = [[float(x.split(',')[0])] for x in text.split('\n')]
Y = [float(x.split(',')[1]) for x in text.split('\n')]
f.close()

model = linear_model.LinearRegression()
model_2 = linear_model.Ridge (alpha = .5)
lasso = Lasso(alpha=0.1)


model.fit(X, Y)
model_2.fit(X, Y)

print(model.coef_)

print(model.predict(X))
print(Y)

mp.scatter(X,Y, color="black")
mp.plot(X, lasso.fit(X, Y).predict(X))
mp.show()
