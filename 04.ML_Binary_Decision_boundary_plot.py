from mlplt import plot_two_class_knn
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

X_C2,y_C2=make_classification(n_samples=100,n_features=2,n_redundant=0,n_informative=2,n_clusters_per_class=1,flip_y=0.1,class_sep=0.5,random_state=0)
X_train,X_test,y_train,y_test=train_test_split(X_C2,y_C2, random_state=0)

plot_two_class_knn(X_train,y_train,1,'uniform',X_test,y_test)
plot_two_class_knn(X_train,y_train,3,'uniform',X_test,y_test)
plot_two_class_knn(X_train,y_train,11,'uniform',X_test,y_test)
