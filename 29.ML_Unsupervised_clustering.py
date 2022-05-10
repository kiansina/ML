
# 1 K-means clustering

#!!!!!!!!!!!!!!!! POINT: You need to specify the value of k ahead of time, which is one of the draw backs of k-means.
#!!!!!!!!!!!!!!!! POINT: K-means operates by first randomly picking locations for the k-cluster centers.
#!!!!!!!!!!!!!!!! POINT: different random starting points for the cluster centers often result in very different clustering solutions. So typically, the k-means algorithm is run in scikit-learn with ten different random initializations. And the solution
#                        occurring the most number of times is chosen.
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from adspy_shared_utilities import plot_labelled_scatter

X, y = make_blobs(random_state = 10)

kmeans = KMeans(n_clusters = 3)
kmeans.fit(X)

plot_labelled_scatter(X, kmeans.labels_, ['Cluster 1', 'Cluster 2', 'Cluster 3'])

#!!!!!!!!!!!!!!!! POINT: One distinction should be made here between clustering algorithms that can predict which center new data points should be assigned to, and those that cannot make such predictions. K-means supports the predict method, and so we
#                         can call the fit and predict methods separately.
#!!!!!!!!!!!!!!!! POINT: agglomerative clustering does not and must perform the fit and predict in a single step
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from mlplt import plot_labelled_scatter
from sklearn.preprocessing import MinMaxScaler

fruits = pd.read_table('fruit_data_with_colors.txt')
X_fruits = fruits[['mass','width','height', 'color_score']].as_matrix()
y_fruits = fruits[['fruit_label']] - 1

X_fruits_normalized = MinMaxScaler().fit(X_fruits).transform(X_fruits)

kmeans = KMeans(n_clusters = 4, random_state = 0)
kmeans.fit(X_fruits_normalized)

plot_labelled_scatter(X_fruits_normalized, kmeans.labels_,
                      ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'])


# 2 Agglomerative Clustering
#!!!!!!!!!!!!!!!! POINT: Note that kmeans is very sensitive to the range of future values. So if your data has features with very different ranges, it's important to normalize using min-max scaling, as we did for some supervised learning methods.
#!!!!!!!!!!!!!!!! POINT: Also the version k-means we saw here assumed that the data features were continuous values. However, in some cases we may have categorical features, where taking the mean doesn't make sense. In that case, there are variants
#                         of k-means that can use a more general definition of distance. Such as the k-medoids algorithm that can work with categorical features.
#!!!!!!!!!!!!!!!! POINT: Such as the k-medoids algorithm that can work with categorical features. First, each data point is put into its own cluster of one item. Then, a sequence of clusterings are done where the most similar two clusters at each stage
#                        are merged into a new cluster. Then, this process is repeated until some stopping condition is met. In scikit-learn, the stopping condition is the number of clusters.
#!!!!!!!!!!!!!!!! POINT: You can choose how the agglomerative clustering algorithm determines the most similar cluster by specifying one of several possible linkage criteria. In scikit-learn, the following three linkage criteria are available, ward,
#                        average, and complete.
#                        1. Ward's method chooses to merge the two clusters that give the smallest increase in total variance within all clusters.
#                        2. Average linkage merges the two clusters that have the smallest average distance between points.
#                        3. Complete linkage, which is also known as maximum linkage, merges the two clusters that have the smallest maximum distance between their points.
#!!!!!!!!!!!!!!!! POINT: In general, Ward's method works well on most data sets, and that's our usual method of choice.
#!!!!!!!!!!!!!!!! POINT: In some cases, if you expect the sizes of the clusters to be very different, for example, that one cluster is much larger than the rest. It's worth trying average and complete linkage criteria as well.
#!!!!!!!!!!!!!!!! POINT: One of the nice things about agglomerative clustering is that it automatically arranges the data into a hierarchy as an effect of the algorithm, reflecting the order and cluster distance at which each data point is assigned
#                        to successive clusters. This hierarchy can be useful to visualize using what's called a dendrogram, which can be used even with higher dimensional data. Note that you can tell how far apart the merged clusters are by the
#                        length of each branch of the tree.
#!!!!!!!!!!!!!!!! POINT: Scikit-learn doesn't provide the ability to plot dendrograms, but SciPy does. SciPy handles clustering a little differently than scikit-learn.
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from mlplt import plot_labelled_scatter

X, y = make_blobs(random_state = 10)

cls = AgglomerativeClustering(n_clusters = 3)
cls_assignment = cls.fit_predict(X)

plot_labelled_scatter(X, cls_assignment,
        ['Cluster 1', 'Cluster 2', 'Cluster 3'])

#%%%%%%% 2.1 Creating a dendrogram (using scipy)
X, y = make_blobs(random_state = 10, n_samples = 10)
plot_labelled_scatter(X, y,
        ['Cluster 1', 'Cluster 2', 'Cluster 3'])
print(X)

from scipy.cluster.hierarchy import ward, dendrogram
plt.figure()
dendrogram(ward(X))
plt.show()


# 3 DBSCAN clustering
#!!!!!!!!!!!!!!!! POINT: But there are other data sets where both k-means clustering and agglomerative clustering don't perform well. So we're now going to give an overview of a third clustering method called DBSCAN.
#!!!!!!!!!!!!!!!! POINT: DBSCAN is an acronym that stands for density-based spatial clustering of applications with noise.
#!!!!!!!!!!!!!!!! POINT: One advantage of DBSCAN is that you don't need to specify the number of clusters in advance.
#!!!!!!!!!!!!!!!! POINT: Another advantage is that it works well with datasets that have more complex cluster shapes.
#!!!!!!!!!!!!!!!! POINT: It can also find points that are outliers that shouldn't reasonably be assigned to any cluster.
#!!!!!!!!!!!!!!!! POINT: DBSCAN is relatively efficient and can be used for large datasets. The main idea behind DBSCAN is that clusters represent areas in the dataspace that are more dense with data points, while being separated by regions that are empty
#                        or at least much less densely populated.
#!!!!!!!!!!!!!!!! POINT: All points that lie in a more dense region are called core samples.
#!!!!!!!!!!!!!!!! POINT: For a given data point, if there are min sample of other data points that lie within a distance of eps, that given data points is labeled as a core sample. Then, all core samples that are with a distance of eps units apart are put
#                        into the same cluster. points that don't end up belonging to any cluster are considered as noise. While points that are within a distance of eps units from core points, but not core points themselves, are termed boundary points.
#!!!!!!!!!!!!!!!! POINT: One consequence of not having the right settings of eps and min samples for your particular dataset might be that the cluster memberships returned by DBSCAN may all be assigned the label -1, which indicates noise.
#!!!!!!!!!!!!!!!! POINT: With DBSCAN, if you've scaled your data using a standard scalar or min-max scalar to make sure the feature values have comparable ranges, finding an appropriate value for eps is a bit easer to do.
#!!!!!!!!!!!!!!!! POINT: make sure that when you use the cluster assignments from DBSCAN, you check for and handle the -1 noise value appropriately. Since this negative value might cause problems, for example, if the cluster assignment is used as an index into
#                        another array later on.
#!!!!!!!!!!!!!!!! POINT: in the case of clustering, for example, there's ambiguity, in a sense that there are typically multiple clusterings that could be plausibly assigned to a given data set. And none of them is obviously better than another unless we
#                        have some additional criteria. Such as, performance on the specific application task that does have an objective evaluation to use as a basis for comparison.

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

X, y = make_blobs(random_state = 9, n_samples = 25)

dbscan = DBSCAN(eps = 2, min_samples = 2)

cls = dbscan.fit_predict(X)
print("Cluster membership values:\n{}".format(cls))

plot_labelled_scatter(X, cls + 1,
        ['Noise', 'Cluster 0', 'Cluster 1', 'Cluster 2'])
