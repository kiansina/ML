####### Unlike random forest method that builds and combines a forest of randomly different trees in parallel, the key idea of gradient boosted decision trees is that they build a series of trees where each tree is trained so that it attempts to correct
#           the mistakes of the previous tree in the series
#!!!!!!!!!! POINT : New parameter in this method is learning rate ===> Learning rate controls how the gradient boost tree algorithm builds a series of corrective trees. By default it is equal to 0.1
#!!!!!!!!!! POINT : High learning rate --> each successive tree put strong emphases on correcting the mistakes of its predecessor --> result in a more complex individual tree ---> more complex model
#!!!!!!!!!! POINT : smaller learning rate --> less emphasis on thoroughly correcting the errors of the previous step. ---> simpler tree
#!!!!!!!!!! POINT : by default : n_estimators = 100 , max depth = 3
#!!!!!!!!!! POINT : like other decision tree based learning methods, you don't need to apply feature scaling for the algorithm to do well
#!!!!!!!!!! POINT : like other decision tree based learning methods,may not be a good choice for tasks that have very high-dimensional sparse features like text classification, where linear models can provide efficient training and fast accurate prediction.
####### : The key parameters controlling model complexity for gradient boosted tree models are, n_estimators and the learning rate. Typically, these two parameters are tuned together. Since making the learning rates smaller,
#         will require more trees to maintain model complexity.
####### : Unlike random forest, increasing an n_estimators can lead to overfitting.



# application of gradient boosted decision trees:

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from mlplt import plot_class_regions_for_classifier_subplot
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2,
                       centers = 8, cluster_std = 1.3,
                       random_state = 4)
y_D2 = y_D2 % 2


X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)
fig, subaxes = plt.subplots(1, 1, figsize=(6, 6))

clf = GradientBoostingClassifier().fit(X_train, y_train)
title = 'GBDT, complex binary dataset, default settings'
plot_class_regions_for_classifier_subplot(clf, X_train, y_train, X_test,
                                         y_test, title, subaxes)

plt.show()


# application of gradient boosted decision trees on breast cancer data set:
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

from sklearn.ensemble import GradientBoostingClassifier

X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)

clf = GradientBoostingClassifier(random_state = 0)
clf.fit(X_train, y_train)

print('Breast cancer dataset (learning_rate=0.1, max_depth=3)')
print('Accuracy of GBDT classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of GBDT classifier on test set: {:.2f}\n'
     .format(clf.score(X_test, y_test)))


######### The model is a bit iverfitting ===> two ways to learn a less complex gradient boosted tree model are, to reduce the learning rate so that each tree does not try as hard to learn a more complex model, that fixes the mistakes of its predecessor,
#              and to reduce the max _depth parameter for the individual trees in the ensemble
clf = GradientBoostingClassifier(learning_rate = 0.01, max_depth = 2, random_state = 0)
clf.fit(X_train, y_train)

print('Breast cancer dataset (learning_rate=0.01, max_depth=2)')
print('Accuracy of GBDT classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of GBDT classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))
