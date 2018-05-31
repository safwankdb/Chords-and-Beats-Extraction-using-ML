import pydub
def convert(file,path='./'):
	input = pydub.AudioSegment.from_mp3(file)
	if file[-4:] == '.mp3':
		file=file[:-4]
	input = input.set_channels(1)
	input.export(path+file+'.wav', format='wav')