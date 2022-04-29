from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from mlplt import plot_decision_tree
from sklearn.model_selection import train_test_split


iris=load_iris()
X_train,X_test,y_train,y_test=train_test_split(iris.data, iris.target, random_state=3)
clf=DecisionTreeClassifier().fit(X_train,y_train)

print('Accuracy of Decision Tree Classifier on training set: {:.2f}'.format(clf.score(X_train,y_train)))
print('Accuracy of Decision Tree Classifier on test set: {:.2f}'.format(clf.score(X_test,y_test)))

####### setting max decision tree depth to help avoid overfitting
clf2 = DecisionTreeClassifier(max_depth=3).fit(X_train,y_train)

print('Accuracy of Decision Tree Classifier on training set: {:.2f}'.format(clf2.score(X_train,y_train)))
print('Accuracy of Decision Tree Classifier on test set: {:.2f}'.format(clf2.score(X_test,y_test)))

## visualizing decision tree
dot=plot_decision_tree(clf, iris.feature_names, iris.target_names)
dot.render('treee.gv', view=True)

####### Pre-pruned version (max_depth=3)
# it can be pre-pruned also by: min_sample_leaf or max_leaf_nodes
dot=plot_decision_tree(clf2, iris.feature_names, iris.target_names)
dot.render('treee2.gv', view=True)

####### Feature Importance
from mlplt import plot_feature_importances
import matplotlib.pyplot as plt

plt.figure(figsize=(10,4),dpi=80)
plot_feature_importances(clf,iris.feature_names)
plt.show()
print('feature importances: {}'.format(clf.feature_importances_))


########## plot
from sklearn.tree import DecisionTreeClassifier
from mlplt import plot_class_regions_for_classifier_subplot
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state = 0)
fig, subaxes = plt.subplots(6, 1, figsize=(6, 32))

pair_list = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]
tree_max_depth = 4

for pair, axis in zip(pair_list, subaxes):
    X = X_train[:, pair]
    y = y_train
    clf = DecisionTreeClassifier(max_depth=tree_max_depth).fit(X, y)
    title = 'Decision Tree, max_depth = {:d}'.format(tree_max_depth)
    plot_class_regions_for_classifier_subplot(clf, X, y, None,
                                             None, title, axis,
                                             iris.target_names)
    axis.set_xlabel(iris.feature_names[pair[0]])
    axis.set_ylabel(iris.feature_names[pair[1]])

plt.tight_layout()
plt.show()
################

####### Decision trees on a real-world dataset
from sklearn.tree import DecisionTreeClassifier
from ml1plt import plot_decision_tree
from ml1plt import plot_feature_importances
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)

clf = DecisionTreeClassifier(max_depth = 4, min_samples_leaf = 8, #max_leaf_nodes
                            random_state = 0).fit(X_train, y_train)

dot=plot_decision_tree(clf, cancer.feature_names, cancer.target_names)
dot.render('treee3.gv', view=True)

print('Breast cancer dataset: decision tree')
print('Accuracy of DT classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of DT classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))

plt.figure(figsize=(10,6),dpi=80)
plot_feature_importances(clf, cancer.feature_names)
plt.tight_layout()

plt.show()
