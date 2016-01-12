from setup import LOCAL_REPO_DIR
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_validation import LeaveOneLabelOut

data_path = LOCAL_REPO_DIR + 'csv/citysounds.csv'
data = pd.read_csv(data_path)

# data = data[(data['class'] != 'drilling')]
X = data.drop(['class', 'fold', 'Unnamed: 0'], axis=1).values
y = data['class'].values

# feature matrix has many different scales, need to standardize
ss = StandardScaler()
X = ss.fit_transform(X)

X = LinearDiscriminantAnalysis().fit_transform(X, y)

folds = data['fold']
lolo = LeaveOneLabelOut(folds)

# params_svm = [
#   {'C': [10, 5, 1, 0.5], 'kernel': ['linear']},
#   {'C': [10, 5, 1, 0.5], 'gamma': [0.5, 0.1, 0.01, 0.005, 0.001], 'kernel': ['rbf']}]

params_svm = [{'C': [1, 0.75, 0.5, 0.4, 0.3, 0.25], 'gamma': [0.1, 0.075, 0.05, 0.04, 0.03, 0.025, 0.01]}]

svm = SVC()
gs = GridSearchCV(svm, params_svm, cv=lolo, verbose=3, scoring='accuracy')
gs.fit(X, y)

print gs.best_params_
print gs.best_score_

# params_rf = [{'n_estimators': [10, 50, 100, 250, 500], 'criterion': ['gini', 'entropy']}]

# rf = RandomForestClassifier()
# gs = GridSearchCV(rf, params_rf, cv=lolo, verbose=3, scoring='accuracy')
# gs.fit(X, y)

# print gs.best_params_
# print gs.best_score_

# params_gb = [{'learning_rate': [0.1, 0.01, 0.25, 0.5], 'n_estimators': [100], 'max_depth': [3, 5]}]

# gb = GradientBoostingClassifier()
# gs = GridSearchCV(gb, params_gb, cv=lolo, verbose=3, scoring='accuracy')
# gs.fit(X, y)

# print gs.best_params_
# print gs.best_score_

# params_knn = [{'n_neighbors': [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]}]

# knn = KNeighborsClassifier()
# gs = GridSearchCV(knn, params_knn, cv=lolo, verbose=3, scoring='accuracy')
# gs.fit(X, y)

# print gs.best_params_
# print gs.best_score_