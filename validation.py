import numpy as np
import pickle

def split_in_k(data, num_folds):
    num_samples = len(data)
    samples_per_fold = num_samples // num_folds
    overflow = num_samples % samples_per_fold
    current = 0
    for _ in range(num_folds):
        stop = current + samples_per_fold
        if overflow:
            stop += 1
            overflow -= 1
        yield data[current:stop]
        current = stop

# assume both arguments are np.array
def split_groups(data, labels, num_folds):
    indices = np.random.permutation(len(data))
    data = data[indices]
    labels = labels[indices]

    true_class  = data[labels]
    false_class = data[np.logical_not(labels)]
    for true_part, false_part in zip(split_in_k(true_class, num_folds), split_in_k(false_class, num_folds)):
        merged = np.concatenate((true_part, false_part), axis=0)
        l = np.zeros(len(merged), dtype=bool)
        l[:len(true_part)] = True
        yield merged, l

def ecg_file_name(idx):
    return 'ecg_fold_{}.data'.format(idx)


def split_crosscheck_groups(data, labels, num_folds):
    for idx, data in enumerate(split_groups(data, labels, num_folds)):
        with open(ecg_file_name(idx+1), 'wb') as f:
            pickle.dump(data, f)

def load_k_fold_data(idx=1):
    with open(ecg_file_name(idx), 'rb') as f:
        return pickle.load(f)

def evaluate_internal(factory, data_set, k):
    accuracies, errors = [], []
    for i in range(k):
        current = data_set[:i] + data_set[i+1:]
        samples, classes = zip(*current)
        samples, classes = np.concatenate(samples), np.concatenate(classes)
        test_samples, test_classes = data_set[i]
        n_test_samples = len(test_samples)

        good = 0
        clf = factory.train(samples, classes)
        for i in range(n_test_samples):
            result = clf.classify(test_samples[i])
            if result == test_classes[i]:
                good += 1

        accuracy = good/n_test_samples
        accuracies.append(accuracy)
        errors.append(1-accuracy)

    return np.average(accuracies), np.average(errors)


def evaluate_matrix(factory, X, y, k=2):
    data_set = [pair for pair in split_groups(X, y, k)]
    return evaluate_internal(factory, data_set, k)


def evaluate(factory, k=2):
    data_set = [load_k_fold_data(i) for i in range(1,k+1)]
    return evaluate_internal(factory, data_set, k)
