from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import seaborn as sns
import numpy as np
import pandas as pd

## 1
########### Least_squares
X_R1,y_R1=make_regression(n_samples=100,n_features=1,n_informative=1,bias=150.0,noise=30,random_state=0)
X_train,X_test,y_train,y_test= train_test_split(X_R1,y_R1,random_state=0)

linreg=LinearRegression().fit(X_train,y_train)

print('linear model intercept (b): {:.3f}'.format(linreg.intercept_))
print('linear model coeff (w): {}'.format(linreg.coef_))
print('R-squared score (training): {:.3f}'.format(linreg.score(X_train,y_train)))
print('R-squared score (test): {:.3f}'.format(linreg.score(X_test,y_test)))

###### plot
plt.figure(figsize=(5,4))
plt.scatter(X_R1,y_R1,marker='o',s=50,alpha=0.8)
plt.plot(X_R1,linreg.coef_*X_R1+linreg.intercept_,'r-')
plt.title('Least-Square linear regression')
plt.xlabel('feature value (x)')
plt.ylabel('Target value (y)')
plt.show()



## 2
########### Ridge
# Ridge Regression: (Regularization) (Regularization is better when the number of training data is relatively small compared to the number of the features)
from sklearn.datasets import make_friedman1
X_F1,y_F1=make_friedman1(n_samples=100,n_features=7,random_state=0)

from sklearn.linear_model import Ridge

X_train,X_test,y_train,y_test=train_test_split(X_F1,y_F1,random_state=0)
linridge=Ridge(alpha=20).fit(X_train,y_train)

print('Crime dataset')
print('ridge regression linear model intercept: {}'.format(linridge.intercept_))
print('ridge regression linear model coeff:\n{}'.format(linridge.coef_))
print('R-squared score (training): {:.3f}'.format(linridge.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(linridge.score(X_test, y_test)))
print('Number of non-zero features: {}'.format(np.sum(linridge.coef_ != 0)))

###### Scaler object: Effect of scaling on ridge:
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
#It must be fit to the train and then transform both test and train

#instead of three lines below we can write more efficiently:
#scaler.fit(X_train)
#X_train_scaled=scaler.transform(X_train)
#X_test_scaled=scaler.transform(X_test)
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
clf=Ridge().fit(X_train_scaled,y_train)
r2_score=clf.score(X_test_scaled,y_test)


## 3
###### Lasso regression
# Lasso regression: (Regularization)  Is most helpful if you think there are only a few variables that have a medium or large effect on the output variable.
#Otherwise if there are lots of variables that contribute small or medium effects ridge regression is typically the better choice.

from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_train,X_test,y_train,y_test=train_test_split(X_F1,y_F1,random_state=0)
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
linlasso=Lasso(alpha=2,max_iter=10000).fit(X_train_scaled,y_train)
print('R1')
print('lasso regression linear model intercept: {}'.format(linlasso.intercept_))
print('lasso regression linear model coeff:\n {}'.format(linlasso.coef_))
print('non zero features: {}'.format(np.sum(linlasso.coef_ != 0)))

print('R_squared score (training): {:.3f}'.format(linlasso.score(X_train_scaled,y_train)))
print('R_squared score (test): {:.3f}\n'.format(linlasso.score(X_test_scaled,y_test)))
print('Features with non-zero weight (sorted by absolute magnitude):')
for e in sorted (list(zip(list(X_F1),linlasso.coef_)),key=lambda e: -abs(e[1])):
    if e[1] != 0:
        print('\t{},{:.3f}'.format(e[0],e[1]))


# alpha effect:
for alpha in [0.5,1,2,3,5,10,20,50]:
    linlasso=Lasso(alpha,max_iter=10000).fit(X_train_scaled,y_train)
    r2_train=linlasso.score(X_train_scaled,y_train)
    r2_test=linlasso.score(X_test_scaled,y_test)
    print('Alpha={:.2f}\n\
    Features kept: {}, r_squared training: {:.2f},\
    r_squared test: {:.2f}\n'.format(alpha,np.sum(linlasso.coef_ !=0),r2_train,r2_test))
