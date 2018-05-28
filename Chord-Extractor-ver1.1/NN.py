import numpy as np
from PCP import PCP_Extractor as pcp
import pickle
import os
from sklearn.linear_model import LogisticRegression as logReg
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier as SGDc
from sklearn.metrics import accuracy_score
print('Current directoy is ',os.pwd())
print('Enter target directory')
tar_dir = input()
while not os.path.isdir(tar_dir):
    print('Wrong path or directory doesn\'t exist. Try again')
    tar_dir=input()
X = pcp(tar_dir)
#print(X.shape)
y = np.zeros((X.shape)[0])
counter = 0
value = 1
for i in range((X.shape)[0]) :
    counter%=200
    y[i] = value
    counter += 1
#logRegr = logReg()
#logRegr.fit(X, y)
#print(logRegr.predict(X))
#model = MLPClassifier(solver='lbfgs',activation='logistic',
 #                      alpha=1e-2, hidden_layer_sizes=(50,),
 #                     momentum=0.25, random_state=1)
print('Enter Maximum Iterations')
maxIter=input()
model = SGDc(learning_rate='constant', eta0=0.1, max_iter=maxIter, tol = 1e-3)
print('Fitting model to data...')
model.fit(X, y)
print(model.predict(X))
filename = 'trained_NN_ver2.sav'
pickle.dump(model, open(filename, 'wb'))
myModel = pickle.load(open('trained_NN_ver2.sav', 'rb'))
pred = myModel.predict(X)
print('The model accuracy for this run was',accuracy_score(y, pred))
