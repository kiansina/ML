#Sometimes we want to evaluate the effect that an important parameter of a model has on the cross-validation scores. The useful function validation curve makes it easy to run this type of experiment.
#Like cross-value score, validation curve will do threefold cross-validation by default but you can adjust this with the CV parameter as well.
#cross-validation is used to evaluate the model and not learn or tune a new model.
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fruits=pd.read_table('fruit_data_with_colors.txt')
X_fruits_2d=fruits[['height','width']]
y_fruits_2d=fruits['fruit_label']
X=X_fruits_2d.to_numpy()
y=y_fruits_2d.to_numpy()




param_range=np.logspace(-3,3,4)
train_scores , test_scores = validation_curve(SVC(), X, y, param_name='gamma',param_range=param_range, cv=3)  # cv changes the number of folds (clusters)

print(train_scores)
print(test_scores)



##### Plot
# This code based on scikit-learn validation_plot example
#  See:  http://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html
plt.figure()

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title('Validation Curve with SVM')
plt.xlabel('$\gamma$ (gamma)')
plt.ylabel('Score')
plt.ylim(0.0, 1.1)
lw = 2

plt.semilogx(param_range, train_scores_mean, label='Training score',
            color='darkorange', lw=lw)

plt.fill_between(param_range, train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std, alpha=0.2,
                color='darkorange', lw=lw)

plt.semilogx(param_range, test_scores_mean, label='Cross-validation score',
            color='navy', lw=lw)

plt.fill_between(param_range, test_scores_mean - test_scores_std,
                test_scores_mean + test_scores_std, alpha=0.2,
                color='navy', lw=lw)

plt.legend(loc='best')
plt.show()
