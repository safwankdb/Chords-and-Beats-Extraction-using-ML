import numpy as np
import scipy.io.wavfile
from os import listdir
from os.path import isfile, join

def convStoM(y) :
    y = y.astype(float)
    mono_y = y[:,0]/2 + y[:,1]/2
    return mono_y

def pcp(path) :
    fs,y = scipy.io.wavfile.read(path)
    if len(y.shape) == 2 :
        y = convStoM(y)
    n = np.size(y)
    k = int(n/2)
    y = (np.square(abs(np.fft.rfft(y))[:k]))
    pcp = np.zeros(12, dtype=float)
    fref = 130.8
    M = np.zeros(k)
    M[0] = -1
    for l in range(1, k) :
        M[l] = round(12*np.log2((fs/fref)*(l/n)))%12
    for i in range(0, 12) :
        pcp[i] = np.dot(y, M==i)
    pcp = pcp/sum(pcp)
    return pcp

def mPCP(y, fs) :
    if len(y.shape) == 2 :
        y = convStoM(y)
    n = np.size(y)
    k = int(n/2)
    y = (np.square(abs(np.fft.rfft(y))[:k]))
    pcp = np.zeros(12, dtype=float)
    fref = 130.8
    M = np.zeros(k)
    M[0] = -1
    for l in range(1, k) :
        M[l] = round(12*np.log2((fs/fref)*(l/n)))%12
    for i in range(0, 12) :
        pcp[i] = np.dot(y, M==i)
    pcp = pcp/sum(pcp)
    return pcp


#print(path + "/" + all_files[0])
def PCP_Extractor(tar_dir) :
    #tar_dir = "A:/ML/Chords-and-Beats-Extraction-using-ML-master/Ver1/Training Set/Guitar_Only/test"
    all_files = [f for f in listdir(tar_dir) if isfile(join(tar_dir, f))]
    PCP = np.zeros((len(all_files), 12))
    i = 0
    for file in all_files :
        PCP[i] = pcp(tar_dir + "/" + file)
        i += 1
    return PCP
