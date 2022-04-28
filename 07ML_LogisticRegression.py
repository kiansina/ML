# Logistic Regression: applies a function that transfer result on a s-shape diagram between 0 and 1. Values larger or equal to 0.5 are considered as 1, others as 0.
# Although thi seems binary, but with application of the function for more times we can have it as classifier. Apple/Not Apple , Orange/Not Orange etc.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from mlplt import(plot_class_regions_for_classifier_subplot)
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

fruits = pd.read_table('fruit_data_with_colors.txt')
X_fruits_2d = fruits[['height', 'width']]
y_fruits_2d = fruits['fruit_label']
fig,subaxes=plt.subplots(1,1,figsize=(7,5))
y_fruits_apple=y_fruits_2d==1 #make into a binary problem: apples vs everything else
X_train,X_test,y_train,y_test=train_test_split(X_fruits_2d.to_numpy(),y_fruits_apple.to_numpy(),random_state=0)
clf=LogisticRegression(C=100).fit(X_train,y_train)
plot_class_regions_for_classifier_subplot(clf,X_train,y_train,None,None,'Logistic regression for binary classification\nFruit dataset: Apple vs others',subaxes)
h = 6
w = 8
print('A fruit with height {} and width {} is predicted to be: {}'.format(h,w, ['not an apple', 'an apple'][clf.predict([[h,w]])[0]]))

h = 10
w = 7
print('A fruit with height {} and width {} is predicted to be: {}'.format(h,w, ['not an apple', 'an apple'][clf.predict([[h,w]])[0]]))

subaxes.set_xlabel('height')
subaxes.set_ylabel('width')
print('Accuracy of Logistic regression classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))

################### Logistic on synthetic dataset
X_C2,y_C2=make_classification(n_samples=100,n_features=2,n_redundant=0,n_informative=2,n_clusters_per_class=1,flip_y=0.1,class_sep=0.5,random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2,random_state = 0)

fig, subaxes = plt.subplots(1, 1, figsize=(7, 5))
clf = LogisticRegression().fit(X_train, y_train)
title = 'Logistic regression, simple synthetic dataset C = {:.3f}'.format(1.0)
plot_class_regions_for_classifier_subplot(clf, X_train, y_train,None, None, title, subaxes)

print('Accuracy of Logistic regression classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))

plt.show()

###### C parameter
# POINT: for both support vector machines and logistic regression:  HIGHER C => less Regularization
# Large valuec of C => Logistic regression tries to fit the training data as well as possible
#Small values of C => model tries to find model coefficients that are closed to 0 even if that model fits the training data a little bit worse
X_train, X_test, y_train, y_test = (train_test_split(X_fruits_2d.to_numpy(),y_fruits_apple.to_numpy(),random_state=0))
fig, subaxes = plt.subplots(3, 1, figsize=(4, 10))
for this_C, subplot in zip([0.1, 1, 100], subaxes):
    clf = LogisticRegression(C=this_C).fit(X_train, y_train)
    title ='Logistic regression (apple vs rest), C = {:.3f}'.format(this_C)
    plot_class_regions_for_classifier_subplot(clf, X_train, y_train,X_test, y_test, title,subplot)

plt.tight_layout()
plt.show()
