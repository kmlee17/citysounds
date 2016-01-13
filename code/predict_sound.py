from setup import LOCAL_REPO_DIR
import numpy as np
import pandas as pd
import cPickle
import librosa as lr
# import aubio as ab
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import normalize, StandardScaler

with open('model/svm.pkl', 'rb') as f1:
    svm = cPickle.load(f1)

with open('model/lda.pkl', 'rb') as f2:
    lda = cPickle.load(f2)

with open('model/ss.pkl', 'rb') as f3:
    ss = cPickle.load(f3)

# filepath = LOCAL_REPO_DIR + 'test/drill.wav'
# wav_mat, sr = lr.load(filepath, sr=44100)
# S = lr.feature.melspectrogram(wav_mat, sr=sr, n_mels=40, fmax=22500, n_fft=1024, hop_length=512)
# mfcc = lr.feature.mfcc(S=lr.logamplitude(S), n_mfcc=25)
# # mfcc = lr.feature.mfcc(wav_mat, sr, n_mfcc=25, n_fft=1024, hop_length=512, n_mels=40, fmax=22050)
# # mfcc = lr.feature.mfcc(wav_mat, sr, n_mfcc=13)
# mfcc_delta = lr.feature.delta(mfcc)
# mfcc_delta = mfcc_delta.T
# mfcc_delta2 = lr.feature.delta(mfcc, order=2)
# mfcc_delta2 = mfcc_delta2.T
# mfcc = mfcc.T
# total_mfcc = np.column_stack((mfcc, mfcc_delta, mfcc_delta2))
# avg_mfcc = np.mean(total_mfcc, axis=0)
# var_mfcc = np.var(total_mfcc, axis=0)
# max_mfcc = np.max(mfcc, axis=0)
# min_mfcc = np.min(mfcc, axis=0)
# med_mfcc = np.median(mfcc, axis=0)
# skew_mfcc = skew(mfcc, axis=0)
# kurt_mfcc = skew(mfcc, axis=0)
# X = np.concatenate((avg_mfcc, var_mfcc, max_mfcc, min_mfcc, med_mfcc, skew_mfcc, kurt_mfcc))

# X = np.asarray(X)
# X = X.reshape(1, 275)

# X = ss.transform(X)

# X = lda.transform(X)

# y_pred = svm.predict(X)

# print y_pred

def single_file_featurization(filepath):
    wav_mat, sr = lr.load(filepath, sr=44100)
    S = lr.feature.melspectrogram(wav_mat, sr=sr, n_mels=40, fmax=22050, n_fft=1024, hop_length=512)
    mfcc = lr.feature.mfcc(S=lr.logamplitude(S), n_mfcc=25)
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
    X = np.concatenate((avg_mfcc, var_mfcc, max_mfcc, min_mfcc, med_mfcc, skew_mfcc, kurt_mfcc))
    X = np.asarray(X)
    X = X.reshape(1, 275)
    X = ss.transform(X)
    X = lda.transform(X)
    return X

filepath = LOCAL_REPO_DIR + 'test/drill.wav'
X = single_file_featurization(filepath)

y_pred = svm.predict(X)

print y_pred
