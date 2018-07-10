from pydub import AudioSegment
import os
files=os.listdir()
for file in files:
	AudioSegment.from_wav(file)[:500].export('X'+file,format='wav')