from pydub import AudioSegment
def remove(myAudioFile):
	if myAudioFile[:-4] == '.mp3':
		fmt='mp3'
	else:
		import filetype
		fmt=filetype.guess(myAudioFile).extension
	sound_stereo = AudioSegment.from_file(myAudioFile, format=fmt)
	sound_monoL = sound_stereo.split_to_mono()[0]
	sound_monoR = sound_stereo.split_to_mono()[1]
	sound_monoR_inv = sound_monoR.invert_phase()
	sound_CentersOut = sound_monoL.overlay(sound_monoR_inv)
	sound_CentersOut.export(myAudioFile+'NoLyrics', format='mp3')