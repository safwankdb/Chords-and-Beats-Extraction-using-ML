import numpy as np
import scipy.io.wavfile
from os import listdir
from os.path import isfile, join

def pcp(path) :
    fs,y = scipy.io.wavfile.read(path)
    n = np.size(y)
    k = int(n/2)
    y = np.transpose(np.square(np.absolute(np.fft.rfft(y))[:k]))
    if ((y.shape()))
    pcp = np.zeros(12)
    fref = 130.8
    M = np.zeros(k)
    M[0] = -1
    for l in range(1, k) :
        M[l] = round(12*np.log2((fs/fref)*(l/n)))%12
    for i in range(12) :
        pcp[i] = np.dot(y, (M==(i*np.ones(k))))
    pcp = pcp/sum(pcp)
    return pcp

#print(path + "/" + all_files[0])
def PCP_Extractor(tar_dir) :
    #tar_dir = "A:/ML/Chords-and-Beats-Extraction-using-ML-master/Ver1/Training Set/Guitar_Only/test"
    all_files = [f for f in listdir(tar_dir) if isfile(join(tar_dir, f))]
    PCP=np.array([pcp(tar_dir+"/"+all_files[i]) for i in range(len(all_files))])
    return PCP
