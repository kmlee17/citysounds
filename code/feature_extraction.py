# Feature extraction of wav files

from setup import LOCAL_REPO_DIR
import numpy as np
import pandas as pd
import librosa as lr
import aubio as ab
from scipy.stats import skew, kurtosis
from sklearn.manifold import MDS
import scipy.io.wavfile as wav

# variables for the mfcc extraction
window = 1024                # fft window
hop = window / 2             # step size
n_filters = 40
n_coeffs = 25                # mfcc coefficients
sample_rate = 44100

df_path = LOCAL_REPO_DIR + 'metadata/citysound.csv'
df = pd.read_csv(df_path)

# testing first fold only
# df_f1 = df[df['fold'] == 1]

X = []
y = []

# aubio
# for i, row in df.iterrows():
#     wavfile = LOCAL_REPO_DIR + 'mono/fold' + str(row['fold']) + '/' + row['slice_file_name']
#     print wavfile
#     print i
#     wavmat, Fs = lr.load(wavfile, sr=44100)
#     s = ab.source(wavfile, Fs, hop)
#     sample_rate = s.samplerate
#     phase = ab.pvoc(window, hop)
#     mag = ab.mfcc(window, n_filters, n_coeffs, sample_rate)
#     mfcc = np.zeros([n_coeffs,])
#     frames_read = 0
#     while True:
#         samples, read = s()
#         spec = phase(samples)
#         mfcc_out = mag(spec)
#         mfcc = np.vstack((mfcc, mfcc_out))
#         frames_read += read
#         if read < hop: break
#     # mfcc = lr.feature.mfcc(wav_mat, sr, n_mfcc=13)
#     mfcc_delta = lr.feature.delta(mfcc.T)
#     mfcc_delta = mfcc_delta.T
#     mfcc_delta2 = lr.feature.delta(mfcc.T, order=2)
#     mfcc_delta2 = mfcc_delta2.T
#     # mfcc = mfcc.T
#     total_mfcc = np.column_stack((mfcc, mfcc_delta, mfcc_delta2))
#     avg_mfcc = np.mean(total_mfcc, axis=0)
#     var_mfcc = np.var(total_mfcc, axis=0)
#     max_mfcc = np.max(mfcc, axis=0)
#     min_mfcc = np.min(mfcc, axis=0)
#     med_mfcc = np.median(mfcc, axis=0)
#     skew_mfcc = skew(mfcc, axis=0)
#     kurt_mfcc = skew(mfcc, axis=0)
#     features = np.concatenate((avg_mfcc, var_mfcc, max_mfcc, min_mfcc, med_mfcc, skew_mfcc, kurt_mfcc))
#     X.append(features)
#     y.append(row['class'])

# librosa
for i, row in df.iterrows():
    wavfile = LOCAL_REPO_DIR + 'audio/fold' + str(row['fold']) + '/' + row['slice_file_name']

    # print statements to update the progress of the processing
    print wavfile
    print i

    wav_mat, sr = lr.load(wavfile, sr=sample_rate)
    S = lr.feature.melspectrogram(wav_mat, sr=sr, n_mels=n_filters, fmax=sr/2, n_fft=window, hop_length=hop)
    mfcc = lr.feature.mfcc(S=lr.logamplitude(S), n_mfcc=25)
    # mfcc = lr.feature.mfcc(wav_mat, sr, n_mfcc=25, n_fft=1024, hop_length=512, n_mels=40, fmax=22050)
    # mfcc = lr.feature.mfcc(wav_mat, sr, n_mfcc=13)
    mfcc_delta = lr.feature.delta(mfcc)
    mfcc_delta = mfcc_delta.T
    mfcc_delta2 = lr.feature.delta(mfcc, order=2)
    mfcc_delta2 = mfcc_delta2.T
    mfcc = mfcc.T
    total_mfcc = np.column_stack((mfcc, mfcc_delta, mfcc_delta2))
    avg_mfcc = np.mean(total_mfcc, axis=0)
    var_mfcc = np.var(total_mfcc, axis=0)
    max_mfcc = np.max(mfcc, axis=0)
    min_mfcc = np.min(mfcc, axis=0)
    med_mfcc = np.median(mfcc, axis=0)
    skew_mfcc = skew(mfcc, axis=0)
    kurt_mfcc = skew(mfcc, axis=0)
    features = np.concatenate((avg_mfcc, var_mfcc, max_mfcc, min_mfcc, med_mfcc, skew_mfcc, kurt_mfcc))
    X.append(features)
    y.append(row['class'])

X = np.asarray(X)
y = np.asarray(y)

mono = pd.DataFrame(X)
mono['class'] = y
mono['fold'] = df['fold']

mono.to_csv('csv/urbansound_stereo.csv')


