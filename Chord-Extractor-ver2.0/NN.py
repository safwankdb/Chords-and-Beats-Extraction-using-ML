import numpy as np
from PCP import PCP_Extractor
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
tar_dir = "A:/ML/Chords-and-Beats-Extraction-using-ML-master/Ver1/Training Set/Guitar_Only/completeSet"
X = PCP_Extractor(tar_dir)
#print(X.shape)
y = np.zeros((X.shape)[0])
counter = 0
value = 1
for i in range(0, (X.shape)[0]) :
    if counter == 200 :
        value += 1
        counter = 0
    y[i] = value
    counter += 1

#logRegr = LogisticRegression()
#logRegr.fit(X, y)
#print(logRegr.predict(X))
#model = MLPClassifier(solver='sgd',activation='logistic',
 #                      alpha=1e-2, hidden_layer_sizes=(50,),
    #                   momentum=0.25, random_state=1)
model = SGDClassifier(learning_rate='constant', eta0=0.1, max_iter=200, tol = 1e-3)
model.fit(X, y)
#print(model.predict(X))
filename = 'trained_NN_ver3.sav'
pickle.dump(model, open(filename, 'wb'))
myModel = pickle.load(open('trained_NN_ver3.sav', 'rb'))
pred = myModel.predict(X)
print(accuracy_score(y, pred))
