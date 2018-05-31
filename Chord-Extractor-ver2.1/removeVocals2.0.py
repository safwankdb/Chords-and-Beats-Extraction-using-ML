import os
from time import sleep
from pydub.playback import play
from pydub import AudioSegment

SECONDS = 1000

def get_mp3(path):
    return AudioSegment.from_mp3(path)

def num_samples(music):
    return len(music.get_array_of_samples())

def remove_lyrics(music):
    split = music.split_to_mono()
    final_song = AudioSegment.empty()

    left = split[0]
    right = split[1]

    original_arr = music.get_array_of_samples()
    left_arr = left.get_array_of_samples()
    right_arr = right.get_array_of_samples()

    for index in range(int(num_samples(music) / 2)):
        left_val = left_arr[index]
        right_val = right_arr[index]

        modified = int((left_val - right_val) / 2)
        sample = original_arr[modified]

        final_song = final_song + sample

    return final_song


song = get_mp3("sample2")
lyrics_gone = remove_lyrics(song)
#play(lyrics_gone)
f = lyrics_gone.export("output.mp3", format="mp3")