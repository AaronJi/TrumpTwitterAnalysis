
from gensim import corpora, models, similarities
import numpy as np

class LSI_analyzer(object):
    # methodType: 0: CBOW; 1: skip-gram
    def __init__(self, nTopics):
        self.nTopics = nTopics

    def fit_transform(self, texts):
        from basic_analysis import filter_set

        new_texts = list()
        for text in texts:
            tokens = text.lower().split()

            new_tokens = [token for token in tokens if token not in filter_set]
            new_texts.append(new_tokens)

        # extract the BOW model, map token to id
        self.dictionary = corpora.Dictionary(new_texts)

        # convert the text to a BOW vector represented by id
        corpus = [self.dictionary.doc2bow(text) for text in new_texts]

        # train a TF-IDF model based on corpus
        self.tfidf = models.TfidfModel(corpus)

        # generate the IF-IDF vector from the corpus
        corpus_tfidf = self.tfidf[corpus]

        # train a LSI model
        self.lsi = models.LsiModel(corpus_tfidf, id2word=self.dictionary, num_topics=self.nTopics)

        # map the text to the 2-dim topic space
        self.corpus_lsi = self.lsi[corpus_tfidf]

        # build the index to calculate the similarity
        self.index = similarities.MatrixSimilarity(self.lsi[corpus])
        self.index_tfidf = similarities.MatrixSimilarity(self.lsi[corpus_tfidf])

        return self.corpus2mat(self.corpus_lsi)

    def transform(self, texts):
        texts = [text.lower().split() for text in texts]
        corpus = [self.dictionary.doc2bow(text) for text in texts]
        text_tfidf = self.tfidf[corpus]

        # map the query to the n-dim topic space
        text_tfidf_lsi = self.lsi[text_tfidf]
        return self.corpus2mat(text_tfidf_lsi)

    def query_simularity(self, query, model):
        query_bow = self.dictionary.doc2bow(query.lower().split())
        if model == 'bow':
            # map the query to the n-dim topic space
            query_lsi = self.lsi[query_bow]

            # calculate the cos similarity between query and doc:
            sim = self.index[query_lsi]
        else:
            query_tfidf = self.tfidf[query_bow]

            # map the query to the n-dim topic space
            query_tfidf_lsi = self.lsi[query_tfidf]

            # calculate the cos similarity between query and doc:
            sim = self.index_tfidf[query_tfidf_lsi]

        return sim

    def corpus2mat(self, corpus):
        X = np.zeros((len(corpus), self.nTopics))
        for i, c in enumerate(self.corpus_lsi):
            for tc in c:
                j = tc[0]
                X[i, j] = tc[1]
        return X

    def get_topics(self, nWords):
        l_topics = self.lsi.show_topics(self.nTopics, nWords, False, False)
        topics = [l_topic[1] for l_topic in l_topics]

        topics_words = list()
        topics_probs = list()
        for topic in topics:
            topic_words = [kt[0] for kt in topic]
            topic_probs = [kt[1] for kt in topic]
            topics_words.append(topic_words)
            topics_probs.append(topic_probs)
        return topics_words, topics_probs

    def print_topics(self, nTopics_print, nWords):
        print self.lsi.print_topics(min((nTopics_print, self.nTopics)), nWords)

        return
