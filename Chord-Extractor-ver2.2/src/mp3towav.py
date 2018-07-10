import pydub
def convert(file,path='./'):
	input = pydub.AudioSegment.from_mp3(file)
	#file='.'.join(file.rsplit('.')[:-1])
	input = input.set_channels(1)
	input.export(path+file.rsplit('.')[0]+'.wav', format='wav')
file=input()
convert(file)