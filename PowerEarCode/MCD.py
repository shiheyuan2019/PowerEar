import numpy as np
import librosa
from scipy.io import wavfile
import pysptk
from scipy.spatial.distance import euclidean
import os
from fastdtw import fastdtw


def readmgc(filename):
    sr, x = wavfile.read(filename)
    x = x.astype(np.float64)
    frame_length = 1024
    hop_length = 256
    frames = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T
    frames *= pysptk.blackman(frame_length)
    var = frames.shape[1] == frame_length
    order = 25
    alpha = 0.41
    stage = 5
    gamma = -1.0 / stage

    mgc = pysptk.mgcep(frames, order, alpha, gamma)
    mgc = mgc.reshape(-1, order + 1)
    return mgc


if __name__ == '__main__':
    natural_folder = "./MCD/oppo/real/"
    synth_folder = "./MCD/oppo/fake/"

    _logdb_const = 10.0 / np.log(10.0) * np.sqrt(2.0)
    s = 0.0
    framesTot = 0

    files = os.listdir(natural_folder)
    files2 = os.listdir(synth_folder)
    print(len(files))
    print(len(files2))
    i =0
    for wavID in files2:
        i += 1
        filename1 = natural_folder + wavID
        mgc1 = readmgc(filename1)
        print("Processing -----------{}".format(wavID))

        filename2 = synth_folder + wavID
        print("Processing -----------{}".format(wavID))
        mgc2 = readmgc(filename2)

        x = mgc1
        y = mgc2

        distance, path = fastdtw(x, y, dist=euclidean)

        distance /= (len(x) + len(y))
        pathx = list(map(lambda l: l[0], path))
        pathy = list(map(lambda l: l[1], path))
        x, y = x[pathx], y[pathy]

        frames = x.shape[0]
        framesTot += frames

        z = x - y
        s += np.sqrt((z * z).sum(-1)).sum()
        MCD_value = _logdb_const * float(np.sqrt((z * z).sum(-1)).sum()) / float(frames)
        print(i)
        print("MCD = : {:f}".format(MCD_value))

    MCD_value = _logdb_const * float(s) / float(framesTot)

    print("MCD = : {:f}".format(MCD_value))
