# An ensemble takes multiple individual learning models and combines them to produce an aggregate model that is more powerful than any of its individual learning models alone.
#%%%%%%%% POINTS: The data used to build each tree is selected randomly.
#%%%%%%%% POINTS: The features chosen in each split tests are also randomly selected.
#%%%%%%%% POINTS: Each tree were built from a different random sample of the data called the bootstrap sample.
#%%%%%%%% POINTS: We don't have to perform scaling or other pre-processing as we do with a number of other supervised learning methods. This is one advantage of using random forests.
#%%%%%%%% POINTS: Like decision trees, random forests may not be a good choice for tasks that have very high-dimensional sparse features like text classification, where linear models can provide efficient training and fast accurate prediction.
## Key parameters:
# 1. n_estimators: number of trees to use in ensemble (default=10). Should be larger for larger datasets to reduce overfitting.
# 2. max_features: has a strong effect on performance. Influences the diversity of trees in the forest. (default works well in practice). Default for classification=square root of total number of features.
#            and for regression is the log base two of the total number of features. Smaller values of max features tending to reduce overfitting.
# 3. max_depth: (default=None, splits until all leaves are pure)
# 4. n_jobs: How many cores to use in parallel during training. If you put it -1, it will use all available cores of cpu
# 5. random_state : fixed random setting to make the results reproducible




# Random forest on a binary sample
from sklearn.datasets import make_blobs
X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2,
                       centers = 8, cluster_std = 1.3,
                       random_state = 4)
y_D2 = y_D2 % 2


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from mlplt import plot_class_regions_for_classifier_subplot
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2,
                                                   random_state = 0)
fig, subaxes = plt.subplots(1, 1, figsize=(6, 6))

clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
title = 'Random Forest Classifier, complex binary dataset, default settings'
plot_class_regions_for_classifier_subplot(clf, X_train, y_train, X_test,
                                         y_test, title, subaxes)

plt.show()

# Random forest on fruit dataset:
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from mlplt import plot_class_regions_for_classifier_subplot
import pandas as pd
fruits = pd.read_table('fruit_data_with_colors.txt')
feature_names_fruits = ['height', 'width', 'mass', 'color_score']
X_fruits = fruits[feature_names_fruits]
y_fruits = fruits['fruit_label']
target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon']

X_train, X_test, y_train, y_test = train_test_split(X_fruits.to_numpy(),
                                                   y_fruits.to_numpy(),
                                                   random_state = 0)
fig, subaxes = plt.subplots(6, 1, figsize=(6, 32))

title = 'Random Forest, fruits dataset, default settings'
pair_list = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]

for pair, axis in zip(pair_list, subaxes):
    X = X_train[:, pair]
    y = y_train

    clf = RandomForestClassifier().fit(X, y)
    plot_class_regions_for_classifier_subplot(clf, X, y, None,
                                             None, title, axis,
                                             target_names_fruits)

    axis.set_xlabel(feature_names_fruits[pair[0]])
    axis.set_ylabel(feature_names_fruits[pair[1]])

plt.tight_layout()
plt.show()

clf = RandomForestClassifier(n_estimators = 10,
                            random_state=0).fit(X_train, y_train)

print('Random Forest, Fruit dataset, default settings')
print('Accuracy of RF classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of RF classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))

# Random Forests on a real-world dataset:
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)

clf = RandomForestClassifier(max_features = 8, random_state = 0)
clf.fit(X_train, y_train)

print('Breast cancer dataset')
print('Accuracy of RF classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of RF classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))
