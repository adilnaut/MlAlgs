import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
# import sklearn

f = open('data.txt','r')
text = f.read()
X = [float(x.split(',')[0]) for x in text.split('\n')]
Y = [float(x.split(',')[1]) for x in text.split('\n')]
f.close()
