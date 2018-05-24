import os, numpy, scipy.io.wavfile
from matplotlib import pyplot
print("Enter file path")
c,path,fref=0,input(),130.8
def PCPextract(path):
	fs, y = scipy.io.wavfile.read(path)
	n=numpy.size(y)
	k=int(n/2)
	y=numpy.square(abs(numpy.fft.fft(y)))[:k]
	pcp=numpy.zeros(k)
	M=numpy.zeros(k)
	M[0]=-1
	for l in range(1,k):
		M[l] = round(12*numpy.log2((fs/fref)*(l/n)))%12
	pcp=numpy.zeros(12)
	for i in range(12):
		pcp[i]=numpy.dot(y,(M==(i*numpy.ones(k))))
	return pcp/sum(pcp)
files=os.listdir(path)
for file in files:
	c+=1
	print(c,file)
	pyplot.plot(PCPextract(os.path.join(path,file)))
pyplot.show()