# Contents:
#            1. Principal Components Analysis (PCA)
#            2. Multidimensional scaling (MDS)


import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# Breast cancer dataset
cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

# Our sample fruits dataset
fruits = pd.read_table('fruit_data_with_colors.txt')
X_fruits = fruits[['mass','width','height', 'color_score']]
y_fruits = fruits[['fruit_label']] - 1

#
# 1 Principal Components Analysis (PCA) (Dimensionality Reduction learning)
#

## %% 1.1 Using PCA to find the first two principal components of the breast cancer dataset

#!!!!!!!!!!!!!!!! POINT: Since we are not doing supervised learning in evaluating a model against a test set we don't have to split our dataset into training and test sets.
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

# Before applying PCA, each feature should be centered (zero mean) and with unit variance
X_normalized = StandardScaler().fit(X_cancer).transform(X_cancer)

pca = PCA(n_components = 2).fit(X_normalized)

X_pca = pca.transform(X_normalized)
print(X_cancer.shape, X_pca.shape)


## %% 1.2 Plotting the PCA-transformed version of the breast cancer dataset
from mlplt import plot_labelled_scatter
plot_labelled_scatter(X_pca, y_cancer, ['malignant', 'benign'])

plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.title('Breast Cancer Dataset PCA (n_components = 2)');



## %% 1.3 Plotting the magnitude of each feature value for the first two principal components
fig = plt.figure(figsize=(8, 4))
plt.imshow(pca.components_, interpolation = 'none', cmap = 'plasma')
feature_names = list(cancer.feature_names)

plt.gca().set_xticks(np.arange(0, len(feature_names)));
plt.gca().set_yticks(np.arange(0.5, 2));
plt.gca().set_xticklabels(feature_names, rotation=90, ha='left', fontsize=12);
plt.gca().set_yticklabels(['First PC', 'Second PC'], va='bottom', fontsize=12);

plt.colorbar(orientation='horizontal', ticks=[pca.components_.min(), 0,
                                              pca.components_.max()], pad=0.65);



## %% 1.4 PCA on the fruit dataset (for comparison)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# each feature should be centered (zero mean) and with unit variance
X_normalized = StandardScaler().fit(X_fruits).transform(X_fruits)

pca = PCA(n_components = 2).fit(X_normalized)
X_pca = pca.transform(X_normalized)

from mlplt import plot_labelled_scatter
y_fruits=y_fruits.to_numpy().reshape(59,)
plot_labelled_scatter(X_pca, y_fruits, ['apple','mandarin','orange','lemon'])

plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.title('Fruits Dataset PCA (n_components = 2)');


#!!!!!!!!!!!!!!!! POINT: PCA gives a good initial tool for exploring a dataset, but may not be able to find more subtle groupings that produce better visualizations for more complex datasets.,
#                        There is a family of unsupervised algorithms called Manifold Learning Algorithms that are very good at finding low dimensional structure in high dimensional data and are very useful for visualizations

#
# 2 Multidimensional scaling (MDS) (Manifold learning methods)
#

## %% 2.1 Multidimensional scaling (MDS) on the fruit dataset
from mlplt import plot_labelled_scatter
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
#!!!!!!!!!!!!!!!! POINT: MDS is one widely used manifold learning method. The goal is to visualize a high dimensional dataset and project it onto a lower dimensional space.
#                        IN A WAY THAT PRESERVES INFORMATION ABOUT HOW THE POINTS IN THE ORIGINAL DATA SPACE ARE CLOSE TO EACH OTHER.
# each feature should be centered (zero mean) and with unit variance
X_fruits_normalized = StandardScaler().fit(X_fruits).transform(X_fruits)

mds = MDS(n_components = 2)

X_fruits_mds = mds.fit_transform(X_fruits_normalized)

plot_labelled_scatter(X_fruits_mds, y_fruits, ['apple', 'mandarin', 'orange', 'lemon'])
plt.xlabel('First MDS feature')
plt.ylabel('Second MDS feature')
plt.title('Fruit sample dataset MDS');


#
# 3 t-SNE (Manifold learning methods)
#

#!!!!!!!!!!!!!!!! POINT: t-SNE is a powerful manifold learning method that finds a 2D projection preserving information about neighbors. The distance between points in the 2D scatterplot match as closely as possible the distances
#                        between the same points in the original high dimensional dataset. In particular t-SNE gives much more weight to preserving information about distances between points that are neighbors.
## %% 3.1 t-SNE on the fruit dataset

from sklearn.manifold import TSNE

tsne = TSNE(random_state = 0)

X_tsne = tsne.fit_transform(X_fruits_normalized)

plot_labelled_scatter(X_tsne, y_fruits,
    ['apple', 'mandarin', 'orange', 'lemon'])
plt.xlabel('First t-SNE feature')
plt.ylabel('Second t-SNE feature')
plt.title('Fruits dataset t-SNE');


## %% 3.2 t-SNE on the breast cancer dataset
tsne = TSNE(random_state = 0)
X_normalized = StandardScaler().fit(X_cancer).transform(X_cancer)
X_tsne = tsne.fit_transform(X_normalized)

plot_labelled_scatter(X_tsne, y_cancer,
    ['malignant', 'benign'])
plt.xlabel('First t-SNE feature')
plt.ylabel('Second t-SNE feature')
plt.title('Breast cancer dataset t-SNE');

#!!!!!!!!!!!!!!!! POINT: t-SNE does a poor job of finding structure in this rather small and simple fruit dataset, which reminds us that we should try at least a few different approaches when visualizing data using manifold learning to see which works
#                        best for a particular dataset
#!!!!!!!!!!!!!!!! POINT: t-SNE tends to work better on datasets that have more well-defined local structure; in other words, more clearly defined patterns of neighbors.
