import matplotlib.pyplot as plot
import numpy as np
import scipy.io.wavfile

#Enter filename below
filename = 'a1.wav'
fs,y = scipy.io.wavfile.read(filename)

time=np.linspace(0, len(y)/fs, num=len(y))

plot.figure(1)
plot.plot(time, y)
plot.show()
