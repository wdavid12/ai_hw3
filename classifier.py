from hw3_utils import  *

def euclidean_distance(a,b):
    """Euclidean distance for numpy arrays.
    For a pair of vectors, return the scalar distance.
    For a vector and a matrix, return the distance of the vector
    from each row of the matrix (useful for KNN).
    For a pair of matrices, return the distance between each pair
    of rows.
    """
    diff = a-b
    diff_squared = diff*diff
    return np.sqrt(diff_squared.T.sum(axis=0))

class knn_classifier(abstract_classifier):
    def __init__(self, training_data, training_labels, k = 3):
        self.training_data = training_data
        self.training_labels = np.array(training_labels, dtype=bool)
        self.k = k


    def classify(self, sample):
        assert len(sample) == self.training_data.shape[1]
        distances = euclidean_distance(self.training_data, sample)
        indices = np.argpartition(distances, self.k-1)[:self.k]
        labels = self.training_labels[indices]
        return labels.sum() > (self.k // 2)


class knn_factory(abstract_classifier_factory):
    def __init__(self, k = 3):
        self.k = k


    def train(self, data, labels):
        return knn_classifier(data, labels, self.k)


class sklearn_classifier(abstract_classifier):
    def __init__(self, clf):
        self.clf = clf

    def classify(self, sample):
        return self.clf.predict(sample.reshape(1,-1))


class sklearn_factory(abstract_classifier_factory):
    def __init__(self, clf):
        self.clf = clf

    def train(self, data, labels):
        self.clf.fit(data, labels)
        return sklearn_classifier(self.clf)
