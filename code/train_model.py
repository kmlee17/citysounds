from setup import LOCAL_REPO_DIR
import numpy as np
import pandas as pd
import cPickle
from sklearn.svm import SVC
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report


data_path = LOCAL_REPO_DIR + 'csv/citysounds.csv'
data = pd.read_csv(data_path)

# test = data[(data['class'] != 'car_horn') & (data['class'] != 'jackhammer') & (data['class'] != 'siren')]
# test = data[(data['class'] == 'dog_bark') | (data['class'] == 'children_playing')]
# data = data[(data['class'] != 'engine_idling')]
X = data.drop(['class', 'fold', 'Unnamed: 0'], axis=1).values
y = data['class'].values

# X = normalize(X)

# feature matrix has many different scales, need to standardize
ss = StandardScaler()
X = ss.fit_transform(X)

lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(X, y)

svm = SVC(C=1, gamma=0.04)
svm.fit(X_lda, y)
y_pred_svm = svm.predict(X_lda)
# kf_accuracy_svm.append(accuracy_score(y_test, y_pred_svm))

print 'model accuracy: ', accuracy_score(y, y_pred_svm)

with open('svm.pkl', 'wb') as f:
    cPickle.dump(svm, f)

with open('lda.pkl', 'wb') as f:
    cPickle.dump(lda, f)

with open('ss.pkl', 'wb') as f:
    cPickle.dump(ss, f)