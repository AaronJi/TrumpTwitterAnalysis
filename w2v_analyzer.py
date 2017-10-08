from gensim.models import Phrases, Word2Vec
import multiprocessing
import numpy as np

class w2v_analyzer(object):
    # methodType: 0: CBOW; 1: skip-gram
    def __init__(self, source, methodType, nFeature, niter, ignoredWordSet):
        self.nFeature = nFeature
        self.niter = niter
        self.ignoredWordSet = ignoredWordSet
        self.vocab = set()
        self.status = 0  # 0: not initiated; 1: initiated, can be trained; -1: initiated, can not be traned

        min_count = 1

        if source is None:
            # initiaze an empty model
            self.model = Word2Vec(iter=1, sg=methodType, size=self.nFeature, min_count=min_count, window=5, workers=multiprocessing.cpu_count())
        elif source == 'twitter-glove':
            # currently not work!
            from gensim.models.keyedvectors import KeyedVectors
            import os
            current_folder = os.path.dirname(os.path.realpath(__file__))
            self.model = KeyedVectors.load_word2vec_format(current_folder+'/Data/w2vModel/glove/glove.twitter.27B.25d.txt', binary=False)
            #self.model = Word2Vec.load_word2vec_format('/Data/w2vModel/glove/glove.twitter.27B.200d.txt', binary=False)

            self.status = -1
        elif source == 'GoogleNews':
            from gensim.models.keyedvectors import KeyedVectors
            import os

            current_folder = os.path.dirname(os.path.realpath(__file__))
            self.nFeature = 300
            self.model = Word2Vec(iter=1, sg=methodType, size=self.nFeature, min_count=min_count, window=5, workers=multiprocessing.cpu_count())
            self.model.wv = KeyedVectors.load_word2vec_format(current_folder+'/Data/w2vModel/GoogleNews-vectors-negative300.bin', binary=True)  # C binary format
            # self.model = KeyedVectors.load_word2vec_format(current_folder+'/Data/w2vModel/glove/glove.twitter.27B.25d.txt', binary=False)  # C text format

            self.status = -1
        elif source == 'path of a pre-saved model':
            self.model = Word2Vec.load(source)
            # self.model.save(source)
            self.status = 1
        # pretrained model: https://github.com/3Top/word2vec-api
        else:
            # initilaze with given texts
            bigram = False
            if bigram:
                bigram_transformer = Phrases(source)
                self.model = Word2Vec(bigram_transformer[source], sg=methodType, iter=self.niter, size=self.nFeature, min_count=min_count, window=5, workers=multiprocessing.cpu_count())
            else:
                self.model = Word2Vec(source, sg=methodType, iter=self.niter, size=self.nFeature, min_count=min_count, window=5, workers=multiprocessing.cpu_count())
        return

    def fit(self, texts):
        if self.status == 0:
            self.model.build_vocab(texts)
        if self.status >= 0:
            print "Train docs " + str(len(texts))
            self.model.train(texts, total_examples=self.model.corpus_count, epochs=self.niter)
        return

    def transform(self, texts):
        X = np.zeros((len(texts), self.nFeature))
        for i, text in enumerate(texts):
            sv = self.getSenVec(text)
            X[i, :] = sv
        return X

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)

    def getWordSimilarity(self, word1, word2):

        return self.wv.similarity(word1, word2)

    def getSimilarWords(self, word, n=5):

        similarResults = self.model.wv.most_similar(positive=word, topn=n)

        simWords = list()
        simNums = list()
        for sR in similarResults:
            simNums.append(sR[1])

        return simWords, simNums

    def getSimilarWordsFromList(self, posWordList, negWordList, n):

        similarResults = self.model.wv.most_similar(positive=posWordList, negative=negWordList, topn=n)

        simWords = list()
        simNums = list()
        for sR in similarResults:
            simWords.append(sR[0])
            simNums.append(sR[1])

        return simWords, simNums

    def getUnmathedWord(self, wordList):
        return self.model.wv.doesnt_match(wordList)

    def wvs2sv(self, wvs, d):
        #sv = np.mean(wvs, axis=0)
        sv = np.zeros(self.nFeature)
        for i in range(d.shape[0]):
            sv += d[i]*wvs[i]
        return sv

    # given a list of words, return a matrix with each row as a word's vector
    def getSenMat(self, words):
        from basic_analysis import subsList

        wvs = None
        words_queue = list()
        d = list()
        for word in words:
            if word not in self.ignoredWordSet:
                if word in subsList:
                    word = subsList[word]
                if word in words_queue:
                    i = words_queue.index(word)
                    d[i] += 1
                else:
                    try:
                        wv = self.model.wv[word]
                        if wvs is None:
                            wvs = wv.copy().reshape(1, -1)
                        else:
                            wvs = np.vstack((wvs, wv))

                        words_queue.append(word)
                        d.append(1)
                    except KeyError:
                        # not included in vocabulary?
                        #print word
                        continue

        d = np.array(d, dtype=float)
        d /= float(np.sum(d))
        if wvs is not None:
            assert len(words_queue) == wvs.shape[0] == d.shape[0]

        return wvs, d

    def getSenVec(self, words):
        wvs, d = self.getSenMat(words)
        return self.wvs2sv(wvs, d)

    def getSenDiff(self, words1, words2, senDiffType):

        wvs1, d1 = self.getSenMat(words1)
        wvs2, d2 = self.getSenMat(words2)

        if wvs1 is None or wvs2 is None:
            return -1

        if senDiffType >= 2:
            # WMD
            from lp_optimizer import WMD
            senDiff, Topt, LPstatus = WMD(wvs1, wvs2, d1, d2)
        else:
            sv1 = self.wvs2sv(wvs1, d1)
            sv2 = self.wvs2sv(wvs2, d2)

            senDiff0 = np.linalg.norm(sv1-sv2, 2)

            if senDiffType == 0:
                # WCD
                senDiff = senDiff0
            elif senDiffType == 1:
                # RWMD
                senDiff = 0
                for i in range(wvs1.shape[0]):
                    cij_min = np.inf
                    for j in range(wvs2.shape[0]):
                        cij = np.linalg.norm(wvs1[i]-wvs2[j], 2)
                        if cij < cij_min:
                            cij_min = cij
                    senDiff += d1[i]*cij_min

                #senDiff = max(senDiff, senDiff0)

        return senDiff