from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_friedman1
X_F1,y_F1=make_friedman1(n_samples=100,n_features=7,random_state=0)
X_train,X_test,y_train,y_test=train_test_split(X_F1,y_F1,random_state=0)
linreg=LinearRegression().fit(X_train,y_train)
print('linear model coeff (w): {}'.format(linreg.coef_))
print('linear model intercept (b): {:.3f}'.format(linreg.intercept_))
print('R-squared score (training): {:.3f}'.format(linreg.score(X_train,y_train)))
print('R_squared score (test): {:.3f}'.format(linreg.score(X_test,y_test)))

######## Poly
poly=PolynomialFeatures(degree=2)
X_F1_poly=poly.fit_transform(X_F1)
X_train,X_test,y_train,y_test=train_test_split(X_F1_poly,y_F1,random_state=0)
linreg=LinearRegression().fit(X_train,y_train)
print('(Poly deg 2) linear model coeff (w): {}'.format(linreg.coef_))
print('(Poly deg 2) linear model intercept (b): {:.3f}'.format(linreg.intercept_))
print('(Poly deg 2) R-squared score (training): {:.3f}'.format(linreg.score(X_train,y_train)))
print('(Poly deg 2) R_squared score (test): {:.3f}'.format(linreg.score(X_test,y_test)))

#%%%%%%%%%%%%%%%%%% POINT : Addition of many polynomial features often leads to overfitting, so we often use polynomial features in combination with regression that has a regularization penalty, like ridge regression


#######   poly+ridge
X_train,X_test,y_train,y_test=train_test_split(X_F1_poly,y_F1,random_state=0)
linreg=Ridge().fit(X_train,y_train)
print('(Poly deg 2 + ridge) linear model coeff (w): {}'.format(linreg.coef_))
print('(Poly deg 2 + ridge) linear model intercept (b): {:.3f}'.format(linreg.intercept_))
print('(Poly deg 2 + ridge) R-squared score (training): {:.3f}'.format(linreg.score(X_train,y_train)))
print('(Poly deg 2 + ridge) R_squared score (test): {:.3f}'.format(linreg.score(X_test,y_test)))
