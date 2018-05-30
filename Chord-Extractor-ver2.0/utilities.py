import os
import soundfile as sf
from PCP import mPCP
import scipy.io.wavfile
import pickle
import numpy as np

def make_part(file, start, time, name) :
    cmd = "C:/ffmpeg/bin/ffmpeg.exe -ss " +  start + " -t " + time + " -i " + file + " " + name
    os.system(cmd)
    return

def all_part(file) :
    f = sf.SoundFile(file)
    duration = len(f)/f.samplerate
    i = 0
    while i + 0.1 < duration :
        output_name = "output" + str(int(int(i*100)/10)) + ".wav"
        make_part(file, str(i), "0.1", output_name)
        i += 0.1
    return

def find_chord(model, file) :
    fs, y = scipy.io.wavfile.read(file)
    X = mPCP(y, fs)
    X = np.array([X])
    pred = model.predict(X)
    return NtoC(pred[0])
myModel = pickle.load(open('trained_NN_ver2.sav', 'rb'))

def analyse(file, model) :
    f = sf.SoundFile(file)
    duration = len(f)/f.samplerate
    i = 0
    all_chords = []
    while i + 0.1 < duration:
        o_name = "output.wav"
        make_part(file, str(i), "0.1", o_name)
        i += 0.1
        all_chords.append(find_chord(model, o_name))
        cmd = "del /f output.wav"
        os.system(cmd)
    return all_chords

def chord_sequence(model, file) :
    f = sf.SoundFile(file)
    duration = len(f)/f.samplerate
    i = 0
    final_chords = []
    while duration > 0 :
        o_name = "foo.wav"
        if duration > 1 :
            make_part(file, str(i), "1", o_name)
        else :
            if duration > 0.5 :
                make_part(file, str(i), str(duration), o_name)
            else :
                final_chords.append("null")
                break
        analysis = analyse(o_name, model)
        final_chords.append(max(set(analysis), key= analysis.count))
        i += 1
        duration -= 1
        cmd = "del /f foo.wav"
        os.system(cmd)
    return final_chords

def NtoC(n) :
    if (n == 1) :
        return "A"
    elif (n == 2) :
        return "Am"
    elif (n == 3):
        return "Bm"
    elif (n == 4):
        return "C"
    elif (n == 5):
        return "D"
    elif (n == 6):
        return "Dm"
    elif (n == 7):
        return "E"
    elif (n == 8):
        return "Em"
    elif (n == 9):
        return "F"
    elif (n == 10) :
        return "G"

def CtoN(c) :
    if (c == "A") :
        return 1
    elif (c == "Am") :
        return 2
    elif (c == "Bm") :
        return 3
    elif (c == "C") :
        return 4
    elif (c == "D") :
        return 5
    elif (c == "Dm") :
        return 6
    elif (c == "E") :
        return 7
    elif (c == "Em") :
        return 8
    elif (c == "F") :
        return 9
    elif (c == "G") :
        return 10
