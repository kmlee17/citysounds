# Preprocessing of audio data
# - Removing background sound
# - Downsampling audio files to a consistent 44.1 kHz
# - Converting stereo to mono for analysis and featurization

from setup import LOCAL_REPO_DIR
import os
import subprocess
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav

def convert_to_16bits(wavfile, newfile):
   pass

n_folds = 10
metadata_path = LOCAL_REPO_DIR + 'metadata/citysound.csv'

md = pd.read_csv(metadata_path)

# save just the foreground sounds in a .csv file
fg = md[md['salience'] == 1]
fg_csv_path = LOCAL_REPO_DIR + 'metadata/foreground_only.csv'
fg.to_csv(fg_csv_path)

# filter out background sounds ('salience' = 2)
bg = md[md['salience'] == 2]

# remove wav files with 'salience' = 2 from dataset
# for _, row in bg.iterrows():
#     del_file_path = LOCAL_REPO_DIR + 'audio/fold' + str(row['fold']) + '/' + row['slice_file_name']
#     # print del_file_path
#     # check if file exists, if it does, delete it
#     if os.path.exists(del_file_path):
#         os.remove(del_file_path)

# create folders to store new audio files
mono_folder_path = LOCAL_REPO_DIR + 'mono'
os.mkdir(mono_folder_path)
for i in xrange(1,11):
	folder_path = mono_folder_path + '/fold' + str(i)
	os.mkdir(folder_path)

# downsampling/upsampling to a consistent 44.1 kHz
# converting stereo to mono
# converting to 16 bit
for _, row in md.iterrows():
    wavfile = LOCAL_REPO_DIR + 'audio/fold' + str(row['fold']) + '/' + row['slice_file_name']
    newfile = mono_folder_path + '/fold' + str(row['fold']) + '/' + row['slice_file_name']
    if os.path.exists(wavfile):
        command = "sox " + wavfile + " -r 44100 -c 1 -b 16 " + newfile
        print command
        subprocess.call(command, shell=True)
    