

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import pyLDAvis, pyLDAvis.sklearn

def LDA_analysis(texts, nTopics, onlyCount=True, showPic=True):
    min_ngram = 2
    max_ngram = 4
    max_df = 1.0
    min_df = 0
    max_features = 500
    learning_offset = 20

    lda = LatentDirichletAllocation(n_topics=nTopics, max_iter=50, learning_method='batch', learning_offset=learning_offset, random_state=0)

    if onlyCount:
        vectorizer = CountVectorizer(analyzer='word', ngram_range=(min_ngram, max_ngram), max_features=max_features, encoding='utf-8', strip_accents='unicode', stop_words='english', max_df=max_df, min_df=min_df)
        X = vectorizer.fit_transform(texts)
    else:
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(min_ngram, max_ngram), max_features=max_features, encoding='utf-8', strip_accents='unicode', stop_words='english', max_df=max_df, min_df=min_df)
        X = vectorizer.fit_transform(texts)
    X_new = lda.fit_transform(X)
    feature_names = vectorizer.get_feature_names()

    print_top_words(lda, feature_names, 10)

    print lda.components_.shape
    print X_new[9]

    cosSim = cosine_similarity(X, lda.components_, False)
    print texts[9]
    print cosSim[9]

    if showPic:
        #pyLDAvis.enable_notebook()
        data_pyLDAvis = pyLDAvis.sklearn.prepare(lda, X, vectorizer)
        pyLDAvis.show(data_pyLDAvis)

    return cosSim

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic #%d" % topic_idx
        print " ".join([str(topic[i]) + '*' + feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]])
    return

if __name__ == '__main__':
    doc1 = "Shipment of gold damaged in a fire"
    doc2 = "Delivery of silver arrived in a silver truck"
    doc3 = "Shipment of gold arrived in a truck"
    documents = doc1 + " " + doc2 + " " + doc3

    LDA_analysis(documents.split(), 2)

