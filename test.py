import numpy as np
from classifier import *
from validation import *
from hw3_utils import *



# train_features, train_labels ,test_features = load_data()


# sick_people = len(train_features[train_labels])
# print("We have %d sick people" % sick_people)


# split_crosscheck_groups(train_features,np.array(train_labels),2)

# for i in range(1,3):
    # samples, l = load_k_fold_data(i)
    # print('#'*80)
    # print('-'*80)
    # print(samples)
    # print('-'*80)
    # print(l)

for i in [1,3,5,7,13]:
    knn = knn_factory(i)
    print(i,evaluate(knn, 2), sep=',')

