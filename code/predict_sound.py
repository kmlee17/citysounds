from setup import LOCAL_REPO_DIR
import numpy as np
import pandas as pd
import cPickle
from feature_extraction import single_file_featurization

def open_pickle(pickle_path):
    '''
    INPUT:
    path of pickled model

    OUTPUT:
    unpickled model

    Pass in the location of the pickled model and the function will unpickle it and return
    the opened model
    '''

    with open(pickle_path, 'rb') as f:
        return cPickle.load(f)


if __name__ == '__main__':
    # location of test file
    filepath = LOCAL_REPO_DIR + 'test/drill.wav'

    # open pickled models for processing new data
    svm = open_pickle('model/svm.pkl')
    lda = open_pickle('model/lda.pkl')
    ss = open_pickle('model/ss.pkl')

    # featurization of audio file
    X = single_file_featurization(filepath)

    # run through models for classification
    X = np.asarray(X)
    X = X.reshape(1, 275)
    X = ss.transform(X)
    X = lda.transform(X)

    y_pred = svm.predict(X)

    print y_pred
