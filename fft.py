import numpy as np
from scipy.signal import find_peaks, correlate
from hw3_utils import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator, ClassifierMixin

from classifier import *

NUM_FEATS = 30

class CrazyClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.forest = RandomForestClassifier(n_estimators=100)
        self.knn = KNeighborsClassifier(n_neighbors=3)

    def fit(self, X, y):
        self.forest.fit(X,y)
        self.knn.fit(X,y)
        return self

    def predict(self, samples):
        return self.forest.predict(samples) & self.knn.predict(samples)

def get_feats(X):
    peaks, _ = find_peaks(X)
    missing = NUM_FEATS - len(peaks)
    peaks = peaks[:NUM_FEATS]
    return np.concatenate([peaks,[0]*missing ,X[peaks],[0]*missing])

class transformer():
    def __init__(self):
        self.clf = RandomForestClassifier(n_estimators=100)

    def fit(self, X, y):
        self.clf = self.clf.fit(X, y)
        self.model = SelectFromModel(self.clf, prefit=True, max_features=50, threshold=-np.inf)

    def transform_features(self, feats):
        new_feats = []
        fft = np.fft.fft(feats, axis=1)
        fft = np.abs(fft[:,:feats.shape[1] // 2])

        for i in range(len(fft)):
            fft_feats = get_feats(fft[i])
            new_feats.append(fft_feats)

        new_feats = np.array(new_feats)
        new_feats = normalize(new_feats)

        X_new = self.model.transform(feats)
        return np.concatenate([X_new, new_feats], axis=1)

def main():
    orig_feats, labels, test = load_data()
    labels = np.array(labels, dtype=bool)

    trans = transformer()
    trans.fit(orig_feats, labels)
    X_new = trans.transform_features(orig_feats)

    print(X_new.shape)
    # tuning

    clf = RandomForestClassifier(n_estimators=100)
    print("Result2:", cross_val_score(clf, X_new, labels, cv=2).mean())
    print("Result3:", cross_val_score(clf, X_new, labels, cv=3).mean())
    crazy = CrazyClassifier()
    print("Crazy2", cross_val_score(crazy, X_new, labels, cv=2).mean())
    print("Crazy3", cross_val_score(crazy, X_new, labels, cv=3).mean())

    clf = CrazyClassifier()
    clf = clf.fit(X_new, labels)

    test_new = trans.transform_features(test)

    classes = clf.predict(test_new)
    print(classes)
    write_prediction(classes)

if __name__ == '__main__':
    main()
