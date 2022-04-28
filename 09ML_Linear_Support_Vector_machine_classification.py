# Linear support vector machine: Uses linear models for classification
# This approach uses the same linear functional form as regression, but instead of predicting a continuous target value,
# we take the output of the linear function and apply the sign function to produce a binary output
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from mlplt import plot_class_regions_for_classifier_subplot
from sklearn.datasets import make_classification
X_C2,y_C2=make_classification(n_samples=100,n_features=2,n_redundant=0,n_informative=2,n_clusters_per_class=1,flip_y=0.1,class_sep=0.5,random_state=0)
X_train,X_test,y_train,y_test=train_test_split(X_C2,y_C2,random_state=0)
fig,subaxes=plt.subplots(1,1,figsize=(7,5))
this_C=1
clf=SVC(kernel='linear',C=this_C).fit(X_train,y_train)
title='Linear SVC, C={:.3f}'.format(this_C)
plot_class_regions_for_classifier_subplot(clf,X_train,y_train,None,None,title,subaxes)
plt.show()
##################
from sklearn.svm import LinearSVC
from mlplt import plot_class_regions_for_classifier

X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state = 0)
fig, subaxes = plt.subplots(1, 2, figsize=(8, 4))

#%%%%%%%%%%%%%%%%%% POINTS:     Larger C => less regularization => fit the training set with few errors as possible EVEN IF IT MEANS USING A SMALLER MARGIN DECISION BOUNDARY
#%%%%%%%%%%%%%%%%%% POINTS:    smaller C => more regularization => Encourages the classifier to find a large margin decision boundary EVEN IF THAT LEADS TO MORE POINTS BEING MISSCLASSIFIED

for this_C, subplot in zip([0.00001, 100], subaxes):
    clf = LinearSVC(C=this_C).fit(X_train, y_train)
    title = 'Linear SVC, C = {:.5f}'.format(this_C)
    plot_class_regions_for_classifier_subplot(clf, X_train, y_train,None, None, title, subplot)

plt.tight_layout()

plt.show()
