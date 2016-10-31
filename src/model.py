from data_load_transform import DataLoadTransform
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


class Model(object):
    """
    Tune and  build a model
    """

    def __init__(self, X_train, y_train, X_test, y_test, clf=LogisticRegression()):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.classifier = clf

    def train_model(self, clfs=[]):
        """
        Cross validate and train on best model
        :param clf:
        :return:
        """
        clf_scores = []
        if not len(clfs):
            clfs = [self.classifier]

        for idx, clf in enumerate(clfs):
            scores = cross_val_score(clf, self.X_train, self.y_train, cv=10, scoring='accuracy')
            clf_scores += [np.mean(scores)]

        best_model = clfs[clf_scores.index(max(clf_scores))]
        return best_model

    def tune_model(self, clfs=None, params=None):
        """
        Tunes different models with a few parameter combinations and returns a list of best performing models
        :param clfs:
        :param params:
        :return:
        """
        tuned_models = []
        if not (clfs and params):
            clfs = [
                LogisticRegression(), RandomForestClassifier(n_estimators=20), SVC()
            ]
            params = [
                {'solver': ['liblinear'], 'penalty': ['l1', 'l2'], 'C': [.001, .01, .5]},
                {"max_depth": [3, None],
                 "bootstrap": [True, False],
                 "criterion": ["gini", "entropy"]}
                ,{'kernel': ['rbf'], 'C': [0.1, 1, 10]}
            ]
        for clf, param in zip(clfs, params):
            tuned_models += [GridSearchCV(clf, param_grid=param)]

        return tuned_models

if __name__ == '__main__':
    dlt = DataLoadTransform()
    raw_data = dlt.init_load_data()
    X, y, class_desc = DataLoadTransform.prepare_data_for_modeling(raw_data)
    X_train, y_train, X_test, y_test = DataLoadTransform.split_test_train(X, y)
    model = Model(X_train, y_train, X_test, y_test)

    # params_grid = {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}
    clfs_tuned = model.tune_model()
    best_clf = model.train_model(clfs=clfs_tuned)
    best_clf.fit(X_train, y_train)
    y_pred = best_clf.predict(X_test)
    print(confusion_matrix(y_test, y_pred, labels=[0,1,2]))
    print(classification_report(y_true=y_test, y_pred=y_pred, target_names=class_desc))

