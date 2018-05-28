import numpy as np
import pickle
from NC import CtoN
from NC import NtoC
from PCP import pcp

#To use the trainer keep the .wav file in the same folder as trainer
#Then enter filename (and not path of file) in first field and
#the true value of chord in second field
#Beware to provide correct true value
#Since otherwise it can lead to bugs in the model
#Also refrain from overtraining with one particular kind of data/chord
#because it can lead to overfitting of data

file = str(input("Enter filename: "))
t_chord = input("Enter true chord of the wav file: ")
myModel = pickle.load(open('trained_NN_ver2.sav', 'rb'))
X = pcp(file)
X = np.array([X])
pred = myModel.predict(X)
print("The model predicted chord to be: ", NtoC(pred[0]))
true_value = np.array([CtoN(t_chord)])
if true_value != pred :
    myModel.partial_fit(X, true_value)
pickle.dump(myModel, open('trained_NN_ver2.sav', 'wb'))

