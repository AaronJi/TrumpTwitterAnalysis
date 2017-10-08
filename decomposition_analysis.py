import numpy as np

def decomposition_analysis(X, nTopics, methodType, featureNames=None):

    if methodType == 'LDA':
        # LDA
        from sklearn.decomposition import LatentDirichletAllocation

        learning_offset = 20
        learning_method = 'batch'
        analyzer = LatentDirichletAllocation(n_components=nTopics, max_iter=50, learning_method=learning_method, learning_offset=learning_offset, random_state=0)
        X_trans = analyzer.fit_transform(X)

        #from sklearn.metrics.pairwise import cosine_similarity
        #cosSim = cosine_similarity(X, analyzer.components_, False)
        #from sklearn.preprocessing import normalize
        #cosSim_norm = normalize(cosSim, norm='l1')

    elif methodType == 'NMF':
        # NMF
        from sklearn.decomposition import NMF

        beta_loss = 'kullback-leibler'
        alpha = .1
        l1_ratio = .5
        analyzer = NMF(n_components=nTopics, random_state=0, beta_loss=beta_loss, solver='mu', max_iter=500, alpha=alpha, l1_ratio=l1_ratio)
        X_trans = analyzer.fit_transform(X)
    elif methodType == 'PCA':
        # PCA
        from sklearn.decomposition import PCA

        analyzer = PCA(n_components=nTopics, random_state=0)
        X_trans = analyzer.fit_transform(X.todense())

    else:
        return None, None

    # show topics
    if featureNames is not None:
        print_top_words(analyzer, featureNames, 20)

    return X_trans, analyzer

def print_top_words(model, featureNames, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic%d" % topic_idx
        print " ".join([str(topic[i]) + '*' + featureNames[i] for i in topic.argsort()[:-n_top_words-1:-1]])
    return

def get_decompositionResult(model, featureNames):
    nTopics = model.components_.shape[0]
    results0_featurename = list()
    results0_possiblity = list()
    topics_freq = np.array(nTopics)
    for iTopic, topic in enumerate(model.components_):
        topics_freq[iTopic] = np.sum(topic)
        topic_featurename = list()
        topic_possiblity = list()
        for i in topic.argsort()[:-len(featureNames)-1:-1]:
            topic_featurename.append(featureNames[i])
            topic_possiblity.append(topic[i])

        results0_featurename.append(topic_featurename)
        results0_possiblity.append(topic_possiblity)

    results_featurename = list()
    results_possiblity = list()
    for iTopic in topics_freq.argsort()[:-nTopics-1:-1]:
        results_featurename.append(results0_featurename[iTopic])
        results_possiblity.append(results0_possiblity[iTopic])

    return results_featurename, results_possiblity
