# Feature extraction of wav files

from setup import LOCAL_REPO_DIR
import numpy as np
import pandas as pd
import librosa as lr
from scipy.stats import skew, kurtosis
import scipy.io.wavfile as wav

# variables for the mfcc extraction

# fft window size: ex. sample rate is 44.1 kHz, a window of 1024 is 0.23 ms
window = 1024
# hop size: how far to jump ahead to the next window
hop = window / 2
# number of mel frequency triangular filters
n_filters = 40
# number of mfcc coefficients to return
n_coeffs = 25
# sample rate of the audio before transformation into the frequency domain
sample_rate = 44100


def extract_features(csv_path):
    '''
    INPUT:
    - A csv file with the following labeled columns:
    - 'slice_file_name': filename of the audio sample
    - 'class': class of audio sample
    - 'fold': numerical folder name of where the file is stored (ie '1' -> in folder 'fold1')

    OUTPUT:
    - Saves features in a csv with 'class' and 'fold' data included in location specified.  Each
      row is a feature vector (275 features) representing a single audio sample

    Reads in a csv and iterates through each row, extracting the features from each audio
    sample.  The resulting features are saved in a .csv in the 'csv' folder for future use
    '''
    
    df = pd.read_csv(csv_path)
    X = []
    y = []

    # iterate through every row in the dataframe, each row represents an audio sample
    for i, row in df.iterrows():

        # construct the path of the file
        wavfile = LOCAL_REPO_DIR + 'audio/fold' + str(row['fold']) + '/' + row['slice_file_name']
    
        # print statements to update the progress of the processing
        print wavfile
        print i
        
        # load the raw audio .wav file as a matrix using librosa
        wav_mat, sr = lr.load(wavfile, sr=sample_rate)

        # create the spectrogram using the predefined variables for mfcc extraction
        S = lr.feature.melspectrogram(wav_mat, sr=sr, n_mels=n_filters, fmax=sr/2, n_fft=window, hop_length=hop)

        # using the pre-defined spectrogram, extract the mfcc coefficients
        mfcc = lr.feature.mfcc(S=lr.logamplitude(S), n_mfcc=25)
        
        # calculate the first and second derivatives of the mfcc coefficients to detect changes and patterns
        mfcc_delta = lr.feature.delta(mfcc)
        mfcc_delta = mfcc_delta.T
        mfcc_delta2 = lr.feature.delta(mfcc, order=2)
        mfcc_delta2 = mfcc_delta2.T
        mfcc = mfcc.T

        # combine the mfcc coefficients and their derivatives in a column stack for analysis
        total_mfcc = np.column_stack((mfcc, mfcc_delta, mfcc_delta2))

        # use the average of each column to condense into a feature vector
        # this makes each sample uniform regardless of the length of original the audio sample
        # the following features are extracted
        # - avg of mfcc, first derivative, second derivative
        # - var of mfcc, first derivative, second derivative
        # - max of mfcc
        # - min of mfcc
        # - median of mfcc
        # - skew of mfcc
        # - kurtosis of mfcc
        avg_mfcc = np.mean(total_mfcc, axis=0)
        var_mfcc = np.var(total_mfcc, axis=0)
        max_mfcc = np.max(mfcc, axis=0)
        min_mfcc = np.min(mfcc, axis=0)
        med_mfcc = np.median(mfcc, axis=0)
        skew_mfcc = skew(mfcc, axis=0)
        kurt_mfcc = skew(mfcc, axis=0)

        # combine into one vector and append to the total feature matrix
        features = np.concatenate((avg_mfcc, var_mfcc, max_mfcc, min_mfcc, med_mfcc, skew_mfcc, kurt_mfcc))
        X.append(features)

    X = np.asarray(X)
    features = pd.DataFrame(X)

    # extract the class and fold of the sample
    features['class'] = df['class']
    features['fold'] = df['fold']

    # output to a csv file for later usage
    features.to_csv('csv/citysounds_test.csv')



if __name__ == '__main__':
    csv_path = LOCAL_REPO_DIR + 'metadata/citysounds_meta.csv'
    extract_features(csv_path)    

