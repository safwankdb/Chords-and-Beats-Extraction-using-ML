import os
import soundfile as sf
from PCP import mPCP
import scipy.io.wavfile
import pickle
import numpy as np

N_to_C={1:'A',2:'Am',3:'Bm',4:'C',5:'D',6:'Dm',7:'E',8:'Em',9:'F',10:'G'}
C_to_N = {v: k for k, v in N_to_C.items()}

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
    duration = len(f) / f.samplerate
    i = 0
    all_chords = []
    while i + 0.2 <= duration:
        o_name = "output.wav"
        make_part(file, str(i), "0.2", o_name)
        i += 0.2
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
        if duration > 0.6 :
            make_part(file, str(i), "0.6", o_name)
        else :
            if duration > 0.3 :
                make_part(file, str(i), str(duration), o_name)
            else :
                final_chords.append("null")
                break
        analysis = analyse(o_name, model)
        final_chords.append(max(set(analysis), key= analysis.count))
        i += 0.6
        duration -= 0.6
        cmd = "del /f foo.wav"
        os.system(cmd)
    return final_chords

def NtoC(n) :
    if n in range(1,11):
    return N_to_C[n]    

def CtoN(c) :
    if c in C_to_N.keys:
        return C_to_N[c]