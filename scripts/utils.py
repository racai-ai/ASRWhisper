import wave
from scipy.io import wavfile
import io
import numpy as np

def getAudioLength(fname):
    with wave.open(fname, "rb") as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration

def readAudioBytes(fname, start, duration):
    with wave.open(fname, "rb") as f:
        # get file data
        rate = f.getframerate()
        frames = f.getnframes()
        fduration = frames / float(rate)
        # set position in wave to start of segment
        f.setpos(int(start * rate))
        # extract data
        return f.readframes(int(min(duration,fduration-start) * rate))

def getAudio(fname, start, duration):
    mem=io.BytesIO()
    with wave.open(fname, "rb") as f:
        # get file data
        nchannels = f.getnchannels()
        sampwidth = f.getsampwidth()
        rate = f.getframerate()
        frames = f.getnframes()
        fduration = frames / float(rate)
        # set position in wave to start of segment
        f.setpos(int(start * rate))
        # extract data
        data=f.readframes(int(min(duration,fduration-start) * rate))
    with wave.open(mem, 'w') as outfile:
        outfile.setnchannels(nchannels)
        outfile.setsampwidth(sampwidth)
        outfile.setframerate(rate)
        outfile.setnframes(int(len(data) / sampwidth))
        outfile.writeframes(data)
    mem.seek(0)
    audio=wavfile.read(mem)[1]
    audio=np.frombuffer(audio, np.int16).flatten().astype(np.float32) / 32768.0
    return audio

def extractAudio(fname, start, duration, fnameout):
    with wave.open(fname, "rb") as f:
        # get file data
        nchannels = f.getnchannels()
        sampwidth = f.getsampwidth()
        rate = f.getframerate()
        frames = f.getnframes()
        fduration = frames / float(rate)
        # set position in wave to start of segment
        f.setpos(int(start * rate))
        # extract data
        data=f.readframes(int(min(duration,fduration-start) * rate))
    with wave.open(fnameout, 'w') as outfile:
        outfile.setnchannels(nchannels)
        outfile.setsampwidth(sampwidth)
        outfile.setframerate(rate)
        outfile.setnframes(int(len(data) / sampwidth))
        outfile.writeframes(data)


#print(getAudioLength("3006_1.wav"))

#print(getAudio("3006_1.wav",10,30))
