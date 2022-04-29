# When the classes are not linearly seperatable we use kernalized support vector machine.
# Kernalized support vector machine is a powerful extension of linear support vector machine.
# SVM can be used both for classification and regression.
# What kernalized svms do? They take the original input data space and transform it to a new higher dimensional feature space,
# where it becomes much easier to classify the transformed data using a linear classifier.
# Examples: 1) Radial basis function kernel (RBF) 2) Polynomial Kernel

# The kernel function in an SVM tells us, given two points in the original input space, what is their similarity in the new feature space.
# For the radial basis function kernel, the similarity between two points and the transformed feature space is an exponentially decaying function of the distance between
# the vectors and the original input space

from sklearn.svm import SVC
from mlplt import plot_class_regions_for_classifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
X_D2,y_D2=make_blobs(n_samples=100,n_features=2,centers=8,cluster_std=1.3,random_state=4) #to create 8 different clusters, but the problem is that they are labeled from 1 to 8 (not binary)
y_D2=y_D2 % 2

X_train,X_test,y_train,y_test=train_test_split(X_D2,y_D2,random_state=0)

## The default SVC kernel is radial basis function (RBF)
plot_class_regions_for_classifier(SVC().fit(X_train, y_train),X_train, y_train, None, None,'Support Vector Classifier: RBF kernel')

# Compare decision boundries with polynomial kernel, degree = 3
plot_class_regions_for_classifier(SVC(kernel = 'poly', degree = 3).fit(X_train, y_train), X_train,y_train, None, None,'Support Vector Classifier: Polynomial kernel, degree = 3')


#%%%%%%%%%%%%%%%%%% POINTS:Gamma controls how far the influence of a single trending example reaches, which in turn affects how tightly the decision boundaries end up surrounding points in the input space.
#%%%%%%%%%%%%%%%%%% POINTS: Small gamma means a larger similarity radius. So that points farther apart are considered similar. Which results in more points being group together and smoother decision boundaries.
##### GAMA PArameter : small gamma more generalized
#%%%%%%%%%%%%%%%%%% POINTS:     Larger C => less regularization => fit the training set with few errors as possible EVEN IF IT MEANS USING A SMALLER MARGIN DECISION BOUNDARY
#%%%%%%%%%%%%%%%%%% POINTS:    smaller C => more regularization => Encourages the classifier to find a large margin decision boundary EVEN IF THAT LEADS TO MORE POINTS BEING MISSCLASSIFIED

from mlplt import plot_class_regions_for_classifier_subplot

fig, subaxes = plt.subplots(3, 1, figsize=(4, 11))

for this_gamma, subplot in zip([0.01, 1.0, 10.0], subaxes):
    clf = SVC(kernel = 'rbf', gamma=this_gamma).fit(X_train, y_train)
    title = 'Support Vector Classifier: \nRBF kernel, gamma = {:.2f}'.format(this_gamma)
    plot_class_regions_for_classifier_subplot(clf, X_train, y_train,
                                             None, None, title, subplot)
    plt.tight_layout()

######## using both C and gamma parameter

fig, subaxes = plt.subplots(3, 4, figsize=(15, 10), dpi=50)

for this_gamma, this_axis in zip([0.01, 1, 5], subaxes):
    for this_C, subplot in zip([0.1, 1, 15, 250], this_axis):
        title = 'gamma = {:.2f}, C = {:.2f}'.format(this_gamma, this_C)
        clf = SVC(kernel = 'rbf', gamma = this_gamma,
                 C = this_C).fit(X_train, y_train)
        plot_class_regions_for_classifier_subplot(clf, X_train, y_train,
                                                 X_test, y_test, title,
                                                 subplot)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

plt.show()
###%%%%%%%%%%%%%%%% If gamma is large then C will have little to no effect.
###%%%%%%%%%%%%%%%% If gamma is small the model is much more constrained and effect of C will be similar to how it would affect a linear classifier.
###%%%%%%%%%%%%%%%% Typically gamma and C are tuned together
###%%%%%%%%%%%%%%%% Kernalized SVMs are pretty sensitive to settings of gamma
###%%%%%%%%%%%%%%%% IT IS IMPORTANT TO NORMALIZE THE INPUT DATA


##### Application of SVM to real dataset: UNNORMALIZED
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer,
                                                   random_state = 0)

clf = SVC(C=10).fit(X_train, y_train)
print('Breast cancer dataset (unnormalized features)')
print('Accuracy of RBF-kernel SVC on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of RBF-kernel SVC on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))

# Train set accuracy = 0.92
# Test set accuracy = 0.94

##### Application of SVM to real dataset: NORMALIZED
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = SVC(C=10).fit(X_train_scaled, y_train)
print('Breast cancer dataset (normalized with MinMax scaling)')
print('RBF-kernel SVC (with MinMax scaling) training set accuracy: {:.2f}'
     .format(clf.score(X_train_scaled, y_train)))
print('RBF-kernel SVC (with MinMax scaling) test set accuracy: {:.2f}'
     .format(clf.score(X_test_scaled, y_test)))

# Train set accuracy = 0.99
# Test set accuracy = 0.97
