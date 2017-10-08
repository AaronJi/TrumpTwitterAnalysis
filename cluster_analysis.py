import numpy as np

def cluster_analysis(X, nTopics, methodType, featureNames=None):

    if methodType == 'Kmeans':
        # KMeans
        from sklearn.cluster import KMeans

        cluster = KMeans(n_clusters=nTopics, random_state=0)
        cluster.fit(X)

        if featureNames is not None:
            print_top_words(cluster.cluster_centers_, featureNames, 20)

        return cluster.labels_, cluster.cluster_centers_
    elif methodType == 'AffinityPropagation':
        # AffinityPropagation
        from sklearn.cluster import AffinityPropagation

        cluster = AffinityPropagation()
        cluster.fit(X)

        if featureNames is not None:
            n = cluster.cluster_centers_.shape[0]
            m = cluster.cluster_centers_.shape[1]
            clusterFeature = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    clusterFeature[i, j] = cluster.cluster_centers_[i, j]

            print_top_words(clusterFeature, featureNames, 20)
        return cluster.labels_, cluster.cluster_centers_, cluster.cluster_centers_indices_

    elif methodType == 2:
        # DBSCAN
        from sklearn.cluster import DBSCAN
        min_samples = np.ceil(X.shape[0] / nTopics)
        cluster = DBSCAN(min_samples=min_samples)
        cluster.fit(X)

        if featureNames is not None:
            n = cluster.cluster_centers_.shape[0]
            m = cluster.cluster_centers_.shape[1]
            clusterFeature = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    clusterFeature[i, j] = cluster.cluster_centers_[i, j]
                    print_top_words(clusterFeature, featureNames, 20)
        return cluster.labels_, cluster.components_, cluster.core_sample_indices_

    else:
        # kNN
        from sklearn.neighbors import NearestNeighbors
        n_neighbors = np.ceil(X.shape[0] / nTopics)
        cluster = NearestNeighbors(n_neighbors=n_neighbors)
        cluster.fit(X)

        X_cluster = np.zeros_like(X, dtype=float)
        for i, x in enumerate(X):
            indices = cluster.kneighbors(x, n_neighbors, False)
            X_cluster[i] = np.mean(X[indices], axis=0)
        return None, None

    return


def print_top_words(clusterFeature, featureNames, n_top_words):
    for topic_idx, topic in enumerate(clusterFeature):
        print "Topic #%d" % topic_idx
        print " ".join([str(topic[i]) + '*' + featureNames[i] for i in topic.argsort()[:-n_top_words-1:-1]])
    return