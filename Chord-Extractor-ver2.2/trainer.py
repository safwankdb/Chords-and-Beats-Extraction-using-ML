import numpy as np
import pickle
from utilities import CtoN
from utilities import NtoC
from PCP import pcp
import os
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.kernel_approximation import RBFSampler
#To use the trainer keep the .wav file in the same folder as trainer
#Then enter filename (and not path of file) in first field and
#the true value of chord in second field
#Beware to provide correct true value
#Since otherwise it can lead to bugs in the model
#Also refrain from overtraining with one particular kind of data/chord
#because it can lead to overfitting of data

file = str(input("Enter filename: "))
t_chord = input("Enter true chord of the wav file: ")

#Three stable versions of models are avaliable
#trained_ML_model_ver1.sav
#trained_ML_model_ver2.sav
#trained_ML_model_ver3.sav
#Model 1 is not further trainable so refrain from using it in trainer
#Model 2 is further trainable but sometimes compromises with results
#Model 3 is further trainable and gives best results currently
myModel = pickle.load(open('trained_ML_model_ver3.sav', 'rb'))
#kernels used to increase features to get better results
#You can use two kernels for this purpose Radial Basis Function Kernel
#and Additive Chi Squared Kernel
#We have refrained ourselves from using other kernels becuase
#they do not provide satisfactory results with our model
sampler = AdditiveChi2Sampler()
#Comment the above sampler and uncomment the lower one to change kernels

#sampler = RBFSampler(gamma=1, random_state=1)

# If file is not a wav file than convert to .wav format
if (file[-3:] != "wav"):
    cmd = "C:/ffmpeg/bin/ffmpeg.exe -i " + file + " " + file[:-3] + ".wav"
    os.system(cmd)
    file = file[:-3] + "wav"
X = pcp(file)
X = np.array([X])
#Change the features using sampler
X = sampler.fit_transform(X)
#predicts the chord of the file using the model you provide it with
pred = myModel.predict(X)
print("The model predicted chord to be: ", NtoC(pred[0]))
true_value = np.array([CtoN(t_chord)])
#Checks if there is any error with prediction and actual output
#And if they differ it fits the true data with the PCP vector
#The changed model is then resaved in the current model
if true_value != pred :
    myModel.partial_fit(X, true_value)
    pickle.dump(myModel, open('trained_ML_model_ver3.sav', 'wb'))

