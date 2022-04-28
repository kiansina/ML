######################## Regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np
X_R1,y_R1=make_regression(n_samples=100,n_features=1,n_informative=1,bias=150.0,noise=30,random_state=0)
X_train,X_test,y_train,y_test=train_test_split(X_R1,y_R1,random_state=0)
knnreg=KNeighborsRegressor(n_neighbors=5).fit(X_train,y_train)
print(knnreg.predict(X_test))
print('R-squared test score: {:.3f}'.format(knnreg.score(X_test,y_test)))
###########################
# plot k-NN regression on sample dataset for different values of K
fig,subaxes=plt.subplots(5,1,figsize=(5,20))
X_predict_input=np.linspace(-3,3,500).reshape(-1,1)


for thisaxis,K in zip(subaxes,[1,3,7,15,55]):
    knnreg=KNeighborsRegressor(n_neighbors=K).fit(X_train,y_train)
    y_predict_output=knnreg.predict(X_predict_input)
    train_score=knnreg.score(X_train,y_train)
    test_score=knnreg.score(X_test,y_test)
    thisaxis.plot(X_predict_input,y_predict_output)
    thisaxis.plot(X_train,y_train,'o',alpha=0.9,label='train')
    thisaxis.plot(X_test,y_test,'^',alpha=0.9,label='Test')
    thisaxis.set_xlabel('Input feature')
    thisaxis.set_ylabel('Targrt value')
    thisaxis.set_title('KNN Regression (K={})\n train $R^2={:.3f}$, test $R^2={:.3f}$'.format(K,train_score,test_score))
    thisaxis.legend()
    plt.tight_layout(pad=0.4,w_pad=0.5,h_pad=1)

plt.show()
