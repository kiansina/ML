# There are three flavors of Naive Bayes Classifier that are available in scikit learn:
# 1. Bernoulli : binary features (eg. word presence/absence)
# 2. Multinomial : discrete features (eg. word counts)
# 3. Gaussian : Continuous/real-valued features ===> Statistics computed for each class: For each feature: mean, standard deviation
#See applied text mining course for more details on the bernoulli and multinomial Naive bayes models.

#%%%%%%%% POINTS: The decision boundary between classes in the two class Gaussian Navie Bayes classifier, in general is a parabolic curve between the classes.
#                       and in the special case where the variance of these features are the same for both classes the decision boundary will be linear.
#%%%%%%%% POINTS: The ellipses gives idea of the shape of the Gaussian distribuition for each class. The centers of the Gaussian's correspond to the mean value of each feature for each class
#%%%%%%%% POINTS: The ellipses show the contour line of the Gaussian distribuition for each class. that corresponds to about two standard deviations from the mean
#%%%%%%%% POINTS: Navie Bayes models are among a few classifiers in scikit learn that support a method called partial fit, which can be used instead of fit to train the classifier incrementally, in case you're working with
#                 a huge data set that does not fit into memory.
#%%%%%%%% POINTS: When the classes are no longer as easily separable as with this second more difficult binary example (mentioned in previous files), like linear models Navie Bayes does not perform as well
#%%%%%%%% POINTS: Typically it is used for high-dimensional data, when each data instance has hundreds thousands or maybe even more features. Likewise the Bernouli and multinomial flavors  are used for text classification where
#                 there are very large number of distinct words as features, and where the feature vectors are sparse
#%%%%%%%% POINTS: It considers each feature independent (Which is not realistic)
#%%%%%%%% POINTS: Their confidence estimates for predictions are not very accurate

## Data set
from sklearn.datasets import make_classification
X_C2, y_C2 = make_classification(n_samples = 100, n_features=2,
                                n_redundant=0, n_informative=2,
                                n_clusters_per_class=1, flip_y = 0.1,
                                class_sep = 0.5, random_state=0)

## Navie Bayes EX1
from sklearn.naive_bayes import GaussianNB
from mlplt import plot_class_regions_for_classifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state=0)

nbclf = GaussianNB().fit(X_train, y_train)
plot_class_regions_for_classifier(nbclf, X_train, y_train, X_test, y_test,
                                 'Gaussian Naive Bayes classifier: Dataset 1')

## Navie Bayes EX2
from sklearn.datasets import make_blobs
X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2,
                       centers = 8, cluster_std = 1.3,
                       random_state = 4)
y_D2 = y_D2 % 2

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2,
                                                   random_state=0)

nbclf = GaussianNB().fit(X_train, y_train)
plot_class_regions_for_classifier(nbclf, X_train, y_train, X_test, y_test,
                                 'Gaussian Naive Bayes classifier: Dataset 2')


## Navie Bayes EX3
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)

nbclf = GaussianNB().fit(X_train, y_train)
print('Breast cancer dataset')
print('Accuracy of GaussianNB classifier on training set: {:.2f}'
     .format(nbclf.score(X_train, y_train)))
print('Accuracy of GaussianNB classifier on test set: {:.2f}'
     .format(nbclf.score(X_test, y_test)))
