from setup import LOCAL_REPO_DIR
import numpy as np
import pandas as pd
import cPickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, f_classif
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_validation import LeaveOneLabelOut

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

X = LinearDiscriminantAnalysis().fit_transform(X, y)

# X = PCA(n_components=0.999, whiten=True).fit_transform(X)

# mds = MDS()
# points = mds.fit_transform(X)

# df = pd.DataFrame(dict(x=points[:,0], y=points[:,1], label=y))
# groups = df.groupby('label')
# fig, ax = plt.subplots()
# ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
# for name, group in groups:
#     ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=name)
# ax.legend()

# plt.show()

kf_accuracy_knn = []
kf_accuracy_svm = []
kf_accuracy_rf = []
kf_accuracy_rf1 = []

folds = data['fold']
lolo = LeaveOneLabelOut(folds)
class_reports = []
kf = KFold(len(X), n_folds=10)
for train_index, test_index in lolo:
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    knn = KNeighborsClassifier(n_neighbors=21)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    kf_accuracy_knn.append(accuracy_score(y_test, y_pred))

    # rf = RandomForestClassifier(n_estimators=500, criterion='entropy')
    # rf.fit(X_train, y_train)
    # y_pred_rf = rf.predict(X_test)
    # kf_accuracy_rf.append(accuracy_score(y_test, y_pred_rf))

    # model = SelectFromModel(rf, prefit=True)
    # X_train_new = model.transform(X_train)
    # X_test_new = model.transform(X_test)

    # rf1 = RandomForestClassifier()
    # rf1.fit(X_train_new, y_train)
    # y_pred_rf1 = rf1.predict(X_test_new)
    # kf_accuracy_rf1.append(accuracy_score(y_test, y_pred_rf1))

    svm = SVC(C=0.3, gamma=0.05)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    kf_accuracy_svm.append(accuracy_score(y_test, y_pred_svm))

    print classification_report(y_test, y_pred_svm)
    class_reports.append(pd.crosstab(pd.Series(y_test), pd.Series(y_pred_svm), rownames=['True'], colnames=['Predicted'], margins=True))

print 'number of samples: ', len(X)
print 'knn accuracy: ', kf_accuracy_knn
# print 'knn accuracy after: ', kf_accuracy_knn1
# print 'rf accuracy: ', kf_accuracy_rf
# print 'rf accuracy overall: ', sum(kf_accuracy_rf) / float(len(kf_accuracy_rf))
# print 'rf accuracy after: ', kf_accuracy_rf1
print 'svm accuracy: ', kf_accuracy_svm
print 'svm accuracy overall: ', sum(kf_accuracy_svm) / float(len(kf_accuracy_svm))
print 'final crosstab:'
conf_mat = sum(class_reports)
print conf_mat
per_conf_mat = conf_mat / conf_mat['All']
per_conf_mat.drop('All', axis=1, inplace=True)
per_conf_mat.drop('All', axis=0, inplace=True)

sns.heatmap(per_conf_mat, annot=True, fmt=".2f", linewidths=.5, cbar=False)