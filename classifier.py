from utils import euclidean_distance
from hw3_utils import  *

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
        false_count, true_count = 0, 0
        for label in labels:
            if label:
                true_count += 1
            else:
                false_count += 1
        return true_count > false_count


class knn_factory(abstract_classifier_factory):
    def __init__(self, k = 3):
        self.k = k


    def train(self, data, labels):
        return knn_classifier(data, labels, self.k)


