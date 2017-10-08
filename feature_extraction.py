
import numpy as np

def feature_extraction(texts, min_ngram, max_ngram, modelType):
    max_df = 1.0
    min_df = 0
    max_features = 500

    from basic_analysis import filter_set

    if modelType == 'tf':
        # tf
        from sklearn.feature_extraction.text import CountVectorizer

        extractor = CountVectorizer(analyzer='word', ngram_range=(min_ngram, max_ngram), max_features=max_features, encoding='utf-8', strip_accents='unicode', stop_words=filter_set, max_df=max_df, min_df=min_df)
        X = extractor.fit_transform(texts)
        featureNames = extractor.get_feature_names()

    elif modelType == 'tf-idf':
        # tf-idf
        from sklearn.feature_extraction.text import TfidfVectorizer

        extractor = TfidfVectorizer(analyzer='word', ngram_range=(min_ngram, max_ngram), max_features=max_features, encoding='utf-8', strip_accents='unicode', stop_words=filter_set, max_df=max_df, min_df=min_df)
        X = extractor.fit_transform(texts)
        featureNames = extractor.get_feature_names()

    elif modelType == 'word2vec':
        # word2vec
        from w2v_analyzer import w2v_analyzer

        texts_tokens = [text.split() for text in texts]
        epoch = 40

        #from sklearn.feature_extraction import text
        #my_stop_words = text.ENGLISH_STOP_WORDS

        #w2vSource = None
        #w2vSource = 'glove.twitter'
        w2vSource = 'GoogleNews'

        m2vmethod = 1  # methodType: 0: CBOW; 1: skip-gram
        extractor = w2v_analyzer(w2vSource, m2vmethod, max_features, epoch, filter_set)

        # a dummy feature names
        featureNames = list()
        for i in range(max_features):
            feature = 'feature' + str(i)
            featureNames.append(feature)

        X = extractor.fit_transform(texts_tokens)

        #extractor = w2v_analyzer(texts_tokens, 1, 0, set())
        #X = extractor.transform(texts_tokens)

        #print X.shape
        #print extractor.getSimilarWords('hillary', 10)

        if False:
            testText = 'hillary is bad hillary is crooked'.split()
            testV = extractor.getSenVec(testText)
            wv0 = extractor.w2v('hillary')
            wv1 = extractor.w2v('is')
            wv2 = extractor.w2v('bad')
            wv3 = extractor.w2v('crooked')
            #testv = 2.0/6.0*wv0 + 2.0/6.0*wv1 + 1.0/6.0*wv2 + 1.0/6.0*wv3
            testv = 2 * wv0 + 2 * wv1 + 1 * wv2 + 1 * wv3
            testv = testv/len(testText)
            print testV - testv
            print np.max(np.abs(testV-testv))

            wvs = np.zeros((len(testText), 500))
            wvs[0, :] = wv0
            wvs[1, :] = wv1
            wvs[2, :] = wv2
            wvs[3, :] = wv0
            wvs[4, :] = wv1
            wvs[5, :] = wv3
            testv1 = np.mean(wvs, axis=0)

            #print testV - testv1
            print np.max(np.abs(testv - testv1))

    else:
        return None, None, None

    return extractor, X, featureNames
