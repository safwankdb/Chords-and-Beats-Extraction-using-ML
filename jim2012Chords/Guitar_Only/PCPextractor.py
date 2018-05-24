import numpy, scipy.io.wavfile
from matplotlib import pyplot
print("Enter file path")
path=input()
fs, y = scipy.io.wavfile.read(path)
n=numpy.size(y)
k=int(n/2)
y=numpy.transpose(numpy.square(abs(numpy.fft.fft(y))[:k]))
pcp=numpy.zeros(k)
fref=130.8
M=numpy.zeros(k)
M[0]=-1
for l in range(1,k):
	M[l] = round(12*numpy.log2((fs/fref)*(l/n)))%12
pcp=numpy.zeros(12)
for i in range(12):
	pcp[i]=numpy.dot(y,(M==(i*numpy.ones(k))))
pcp/=numpy.sum(pcp)
with pyplot.xkcd():
	pyplot.plot(pcp)
pyplot.show()