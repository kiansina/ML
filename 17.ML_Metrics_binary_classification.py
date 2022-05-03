##### Evaluation metrics for binary classification
# Accuracy = TP + TN / (TP + TN + FP + FN)
# Classification Error = FP + FN / (TP + TN + FP + FN) = 1-Accuracy
# Precision = TP / (TP + FP)
# Recall = TP / (TP + FN)  Also known as sensitivity, or True Positive Rate
# Specificity= FP / (TN + FP) Also known as False positive rate
#### F1-score: Combining precision & recall into a single number. This F1 score is a special case of a more general evaluation metric known as an F score that introduces a parameter beta
# F1 = 2 * Precision * Recall / (Precision + Recall) = (2 * TP) / (2 * TP + FN + FP)
#### By adjusting beta we can control how much emphasis an evaluation is given to precision versus recall:
#### precision oriented users : beta = 0.5 (False positives hurt performance more than false negatives)
#### Recall oriented users : beta = 2 (False negatives hurt performance more than false positives) (In general more than 1)
#### If beta = 1 ====> F1 special case
# Fbeta =(1 + beta^2) * (Precision * Recall) / ((beta^2 * precision) + Recall) = ((1 + beta^2) * TP) / (((1 + beta^2) * TP) + beta * FN + FP)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

dataset = load_digits()
X, y = dataset.data, dataset.target
y_binary_imbalanced = y.copy()
y_binary_imbalanced[y_binary_imbalanced != 1] = 0

X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
tree_predicted = dt.predict(X_test)
confusion = confusion_matrix(y_test, tree_predicted)

print('Decision tree classifier (max_depth = 2)\n', confusion)



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, tree_predicted)))
print('Precision: {:.2f}'.format(precision_score(y_test, tree_predicted)))
print('Recall: {:.2f}'.format(recall_score(y_test, tree_predicted)))
print('F1: {:.2f}'.format(f1_score(y_test, tree_predicted)))

## Report all together:
# Combined report with all above metrics
from sklearn.metrics import classification_report

print(classification_report(y_test, tree_predicted, target_names=['not 1', '1']))
