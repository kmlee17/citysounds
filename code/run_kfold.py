from setup import LOCAL_REPO_DIR
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, f_classif
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_validation import LeaveOneLabelOut
import matplotlib.pyplot as plt
import seaborn as sns


def run_kfold(csv_path, model='svm'):
    '''
    INPUT: 
    audio features csv with 'class' and 'fold' data included
    model to use, choose from 'svm', 'knn', 'randomforest'

    OUTPUT:
    - print statements on accuracy of each fold, overall accuracy
    - overall classification report on test data
    - seaborn heatmap of confusion matrix
    
    Uses the 'label' column to build the kfold using sklearn's 'LeaveOneLabelOut' function
    '''

    csv = LOCAL_REPO_DIR + csv_path
    df = pd.read_csv(csv_path)

    # extracts X, y for training model from dataframe
    X = df.drop(['class', 'fold', 'Unnamed: 0'], axis=1).values
    y = df['class'].values

    # feature matrix has many different scales, need to standardize
    X = StandardScaler().fit_transform(X)

    X = LinearDiscriminantAnalysis().fit_transform(X, y)

    # X = PCA(n_components=0.999, whiten=True).fit_transform(X)

    # runs MDS to reduce dimensionality to 2D for an approximation of clusters
    # mds = MDS()
    # points = mds.fit_transform(X)

    # df = pd.DataFrame(dict(x=points[:,0], y=points[:,1], label=y))
    # groups = df.groupby('label')
    # fig, ax = plt.subplots()
    # ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    # for name, group in groups:
    # ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=name)
    # ax.legend()

    # plt.show()

    kf_accuracy = []

    folds = df['fold']
    lolo = LeaveOneLabelOut(folds)
    class_reports = []
    kf = KFold(len(X), n_folds=10)
    for train_index, test_index in lolo:
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if model == 'knn':
            knn = KNeighborsClassifier(n_neighbors=21)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            kf_accuracy.append(accuracy_score(y_test, y_pred))

        elif model == 'randomforest':
            rf = RandomForestClassifier(n_estimators=500, criterion='entropy')
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            kf_accuracy.append(accuracy_score(y_test, y_pred))

        else:
            svm = SVC(C=1, gamma=0.04)
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            kf_accuracy.append(accuracy_score(y_test, y_pred))

        print classification_report(y_test, y_pred)
    class_reports.append(pd.crosstab(pd.Series(y_test), pd.Series(y_pred), rownames=['True'], colnames=['Predicted'], margins=True))

    print 'number of samples: ', len(X)
    print 'kfold accuracy: ', kf_accuracy
    print 'kfold accuracy overall: ', sum(kf_accuracy) / float(len(kf_accuracy))

    # combine classification reports from each kfold for an overall report
    confusion_mat = sum(class_reports)
    print 'final crosstab:'
    print confusion_mat

    # save results of classification report to csv
    # conf_mat.to_csv('csv/kfold_results.csv')

    # convert confusion matrix to percentages rather than absolute values
    per_confusion_mat = confusion_mat / confusion_mat['All']
    per_confusion_mat.drop('All', axis=1, inplace=True)
    per_confusion_mat.drop('All', axis=0, inplace=True)

    # heatmap of confusion matrix
    # note, there is a bug with matplotlib/seaborn where the annotation doesn't show up other than bottom left
    # all annotation shows up when you savefig to a image file
    sns.heatmap(per_confusion_mat, annot=True, fmt=".2f", linewidths=.5, cbar=False)
    plt.show()

if __name__ == '__main__':
    run_kfold('csv/citysounds.csv')