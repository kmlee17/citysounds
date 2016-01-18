from setup import LOCAL_REPO_DIR
import argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_validation import LeaveOneLabelOut

'''
USAGE:
- run from command line (i.e. 'python code/grid_search.py knn')
- choose between 'randomforest', 'svm', 'knn', 'gradientboost'
'''

data_path = LOCAL_REPO_DIR + 'csv/citysounds.csv'

def get_features_classes():
    data = pd.read_csv(data_path)
    # extract feature matrix
    X = data.drop(['class', 'fold', 'Unnamed: 0'], axis=1).values
    # extract class labels
    y = data['class'].values

    # feature matrix has many different scales, need to standardize
    ss = StandardScaler()
    X = ss.fit_transform(X)

    # LDA for dimensionality reduction
    X = LinearDiscriminantAnalysis().fit_transform(X, y)

    # creating custom kfold index structure from metadata fold data
    folds = data['fold']
    lolo = LeaveOneLabelOut(folds)
    return X, y, lolo

def svm_grid_search():
    params_svm = [{'C': [1, 0.75, 0.5, 0.4, 0.3, 0.25], 'gamma': [0.1, 0.075, 0.05, 0.04, 0.03, 0.025, 0.01]}]
    svm = SVC()
    grid_search(svm, params)


def knn_grid_search():
    params = [{'n_neighbors': [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]}]
    knn = KNeighborsClassifier()
    grid_search(knn, params)


def rf_grid_search():
    params = [{'n_estimators': [10, 50, 100, 250, 500], 'criterion': ['gini', 'entropy']}]
    rf = RandomForestClassifier()
    grid_search(rf, params)


def gb_grid_search():
    params = [{'learning_rate': [0.1, 0.01, 0.25, 0.5], 'n_estimators': [100], 'max_depth': [3, 5]}]
    gb = GradientBoostingClassifier()
    grid_search(gb, params)


def grid_search(model, params):
    X, y, lolo = get_features_classes()
    gs = GridSearchCV(model, params, cv=lolo, verbose=3, scoring='accuracy')
    gs.fit(X, y)
    print gs.best_params_
    print gs.best_score_

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='grid search')
    parser.add_argument("model", help="choose between randomforest, svm, knn, gradientboost")
    args = parser.parse_args()
    if args.model == 'svm':
        svm_grid_search()
    elif args.model == 'knn':
        knn_grid_search()
    elif args.model == 'randomforest':
        rf_grid_search()
    elif args.model == 'gradientboost':
        gb_grid_search()
    else:
    	print 'invalid model argument, use randomforest, svm, knn, or grandientboost'