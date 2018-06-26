from utilities import chord_sequence
import pickle
import os
from removeVocals import remove

model = pickle.load(open('trained_ML_model_ver3.sav', 'rb'))
file = input("Enter the file name: ")
if (file[-3:] != "wav"):
    cmd = "C:/ffmpeg/bin/ffmpeg.exe -i " + file + " " + file[:-3] + "wav"
    os.system(cmd)
    file = file[:-3] + "wav"

#remove(file)
print("For each 0.9 second interval the chord in the music is ", chord_sequence(model, file, 1))