# In cross validation there is no need for train and test set, the data would be splited in different folds and for each fold one model would be created and tested to other folds.
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

fruits=pd.read_table('fruit_data_with_colors.txt')
X_fruits_2d=fruits[['height','width']]
y_fruits_2d=fruits['fruit_label']

clf=KNeighborsClassifier(n_neighbors=5)
X=X_fruits_2d.to_numpy()
y=y_fruits_2d.to_numpy()
cv_score=cross_val_score(clf,X,y) #By default there is 3 folds, if you want to change, set CV=5

print('cross validation score (3-fold):',cv_score)
print('mean cross validation score (3-fold): {:.3f}'.format(np.mean(cv_score)))

### NORMALIZING :
# The proper way to do cross-validation when you need to scale the data is not to scale the entire dataset with a single transform, since this will indirectly leak information into the training data about the whole dataset, including the test data.
# Instead, scaling/normalizing must be computed and applied for each cross-validation fold separately. To do this, the easiest way in scikit-learn is to use pipelines.
# http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

########### THIS IS AN EXAMPLE OF PIPELINES
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
X, y = make_classification(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=0)
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
# The pipeline can be used as any other estimator
# and avoids leaking the test set into the train set
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)






############ I CREATE THIS CODE COMBINING TWO ABOVE CODES I AM NOT SURE IF IT IS TRUE
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

fruits=pd.read_table('fruit_data_with_colors.txt')
X_fruits_2d=fruits[['height','width']]
y_fruits_2d=fruits['fruit_label']

clf=KNeighborsClassifier(n_neighbors=5)
X=X_fruits_2d.to_numpy()
y=y_fruits_2d.to_numpy()


pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])



cv_score=cross_val_score(pipe,X,y)

print('cross validation score (3-fold):',cv_score)
print('mean cross validation score (3-fold): {:.3f}'.format(np.mean(cv_score)))
