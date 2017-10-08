import os, sys
import pandas as pd
from tweet_obj import tweet_obj
import logging
import pickle

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sys.stdout.flush()

'''
Create a path if necessary
'''
def mkdir_p(path):
    import errno

    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

''' 
Open "path" for writing, creating any parent directories as needed.
'''
def safe_open_w(path, _):
    mkdir_p(os.path.dirname(path))
    return open(path, 'w')

'''
read data
'''
input_filename = "realDonaldTrump_tweets.csv"
input_path = os.path.join(os.path.expanduser("~"), "TrumpTwitterAnalysis", "Data", input_filename)
tweets_pd = pd.read_csv(input_path)

nTweets_full = len(tweets_pd)

# there is some earlier downloaded data with some columns empty
input_filename_old = "realDonaldTrump_tweets_old.csv"
input_path_old = os.path.join(os.path.expanduser("~"), "TrumpTwitterAnalysis", "Data", input_filename_old)
tweets_pd_old = pd.read_csv(input_path_old)

from getData import combine_from_old_tweets
tweets_pd = combine_from_old_tweets(tweets_pd, tweets_pd_old)

nTweets = len(tweets_pd)


'''
analysis
'''
# if we neglect the retweets
ignore_RT = True
ignore_Quote = False
use_hashtag = False

def getTweetNormalTokens(tweetObj):

    normalTokens = list()
    for tokenObj in tweetObj.tokenList:
        if tokenObj.type == 0:
            if not ignore_Quote or not tokenObj.inQuote:
                normalTokens.append(tokenObj.content)
        elif use_hashtag and tokenObj.type == 2:
            normalTokens.append(tokenObj.content)

    return normalTokens

def getTweetCaptialTokens(tweetObj):
    capitalTokens = list()
    for tokenObj in tweetObj.tokenList:
        if tokenObj.isCapital and tokenObj.type == 0 and not tokenObj.inQuote:
            capitalTokens.append(tokenObj.content)
    return capitalTokens

def list2string(tokenList):
    s = ''
    for i, token in enumerate(tokenList):
        s += token
        if i < len(tokenList)-1:
            s += ' '
    return s

# analyze all tweets, get their tokens with their types
normalWordbag = list()
capitalWordbag = list()
clearTextList = list()
normalTokensList = list()
texts = list()
for i in range(nTweets):
    tweetObj = tweet_obj(tweets_pd.iloc[i])
    tweetObj.text_analyzer()

    if i == 296:
        print tweetObj.text
        print tweetObj.convert2analysis()

    if ignore_RT and len(tweetObj.retweet_source) > 0:
        normalTokensList.append('')
        texts.append('')
    else:
        normalTokens = getTweetNormalTokens(tweetObj)
        normalWordbag.extend(normalTokens)

        capitalTokens = getTweetCaptialTokens(tweetObj)
        capitalWordbag.extend(capitalTokens)

        clearTextList.append(tweetObj.getClearText())
        normalTokensList.append(normalTokens)
        texts.append(list2string(normalTokens))


print '%d tweets, %d words totally.' % (nTweets, len(normalWordbag))

tweets_pd['normalText'] = texts
tweets_pd['hasText'] = [len(text) > 0 for text in texts]

# basic analysis
from basic_analysis import *
basic_analysis(normalWordbag, capitalWordbag, False)

# topics analysis
nTopics = 20

# LSI
analyze_LSI = False
if analyze_LSI:
    print 'LSI:'
    from LSI_analyzer import *
    lsi_analyzer = LSI_analyzer(nTopics)
    X_lsi_trans = lsi_analyzer.fit_transform(texts)
    #lsi_analyzer.print_topics(nTopics, 10)

    topics_words, topics_probs = lsi_analyzer.get_topics(10)
    i = 0
    for topic_words, topic_probs in zip(topics_words, topics_probs):
        i += 1
        print "Topic" + str(i)
        print topic_words
        #print topic_probs

    LSIpickle_filename = 'LSIdata_' + str(nTopics) + 'topics.pkl'
    LSIpickle_path = os.path.join(os.path.expanduser("~"), "TrumpTwitterAnalysis", "Pickles", LSIpickle_filename)
    LSIpickle_file = open(LSIpickle_path, 'wb')
    pickle.dump((X_lsi_trans, lsi_analyzer, topics_words, topics_probs), LSIpickle_file)
    LSIpickle_file.close()


# LDA
min_ngram = 1
max_ngram = 1

use_cluster = 2  # 0: not use cluster; 1: use word2vec to cluster; 2: use LSI result to cluster
calculate_cluster_LDA = False
if use_cluster > 0:
    # cluster analysis
    print 'Cluster analysis:'
    from cluster_analysis import *

    nCluster = 1000
    if calculate_cluster_LDA:
        methodType = 'Kmeans'  # method type: Kmeans, AffinityPropagation, DBSCAN
        valid_irow = [i for i in range(nTweets) if tweets_pd.iloc[i]['hasText']]
        from feature_extraction import *
        if use_cluster == 1:
            # feature extraction
            print 'Feature extraction:'
            modelType = 'word2vec'  # model type: tf, tf-idf, word2vec
            extractor_w2v, X_w2v, featureNames_w2v = feature_extraction(texts, min_ngram, max_ngram, modelType)

            X_extract = X_w2v[valid_irow]
            clusterId, clusterFeature = cluster_analysis(X_extract, nCluster, methodType, None)
        else:
            X_extract = X_lsi_trans[valid_irow]
            clusterId, clusterFeature = cluster_analysis(X_extract, nCluster, methodType, None)

        from lp_optimizer import *
        minC = np.floor(len(valid_irow)/float(nCluster))
        #clusterId, Xopt, status = eqAssign(X_lsi_trans[valid_irow], clusterFeature, minC, False)  # too slow! even not use MIP.
        #print "LP status: " + status

        maxC = np.ceil(len(valid_irow)/float(nCluster))
        clusterId = fastAssign(X_extract, clusterFeature, maxC, minC)

        # label the cluster id for each tweet
        count = 0
        clusterids = list()
        for irow, row in tweets_pd.iterrows():
            if row['hasText']:
                clusterids.append(clusterId[count])
                count += 1
            else:
                clusterids.append(-1)

        tweets_pd['clusterId'] = clusterids
        from collections import Counter
        print Counter(tweets_pd['clusterId'].values)


        groupedText = list()
        for iCluster in range(nCluster):
            clusterText = [row['normalText'] for irow, row in tweets_pd.iterrows() if row['clusterId'] == iCluster]
            gt = " ".join(clusterText)
            groupedText.append(gt)


        # save the new results
        import csv
        output_filename = "groupedText.csv"
        output_file = os.path.join(os.path.expanduser("~"), "TrumpTwitterAnalysis", "Data", output_filename)
        with open(output_file, 'wb') as resultFile:
            wr = csv.writer(resultFile, dialect='excel')
            for gT in groupedText:
                wr.writerow([gT])
            resultFile.close()


        tweets_pd.to_csv(output_file, encoding='utf-8', index=False)

        # feature extraction
        print 'Feature extraction:'
        modelType = 'tf-idf'  # model type: tf, tf-idf, word2vec
        extractor_LDAex, X_LDAex, featureNames_LDAex = feature_extraction(groupedText, min_ngram, max_ngram, modelType)

        # decomposition
        print 'Decomposition analysis:'
        from decomposition_analysis import *
        methodType = 'LDA'  # method type: LDA, NMF, PCA
        X_trans, analyzer = decomposition_analysis(X_LDAex, nTopics, methodType, featureNames_LDAex)
    else:
        methodType = 'LDA'  # method type: LDA, NMF, PCA
        LDApickle_filename = 'LDAdata_' + str(nTopics) + 'topics_' + str(nCluster) + 'clusters_' + str(max_ngram) + 'ngram.pkl'
        LDApickle_path = os.path.join(os.path.expanduser("~"), "TrumpTwitterAnalysis", "Pickles", LDApickle_filename)
        (extractor_LDAex, X_LDAex, featureNames_LDAex, X_trans, analyzer) = pd.read_pickle(LDApickle_path)
else:
    # feature extraction
    print 'Feature extraction:'
    from feature_extraction import *
    modelType = 'tf-idf'  # model type: tf, tf-idf, word2vec
    extractor_LDAex, X_LDAex, featureNames_LDAex = feature_extraction(texts, min_ngram, max_ngram, modelType)

    # decomposition
    print 'Decomposition analysis:'
    from decomposition_analysis import *
    methodType = 'LDA'  # method type: LDA, NMF, PCA
    X_trans, analyzer = decomposition_analysis(X_LDAex, nTopics, methodType, featureNames_LDAex)

if use_cluster == 0 or calculate_cluster_LDA:
    LDApickle_filename = 'LDAdata_' + str(nTopics) + 'topics_' + str(nCluster) + 'clusters_' + str(max_ngram) + 'ngram.pkl'
    LDApickle_path = os.path.join(os.path.expanduser("~"), "TrumpTwitterAnalysis", "Pickles", LDApickle_filename)
    LDApickle_file = open(LDApickle_path, 'wb')
    pickle.dump((extractor_LDAex, X_LDAex, featureNames_LDAex, X_trans, analyzer), LDApickle_file)
    LDApickle_file.close()

# show LDA graphically
showLDAPic = False
if methodType == 'LDA' and showLDAPic:
    import pyLDAvis, pyLDAvis.sklearn
    # pyLDAvis.enable_notebook()
    data_pyLDAvis = pyLDAvis.sklearn.prepare(analyzer, X_LDAex, extractor_LDAex)
    pyLDAvis.show(data_pyLDAvis)


# calculate similarities between tweets and topic keywords
from basic_analysis import myTopics, myTopicsNames, filter_set
nMyTopics = len(myTopicsNames)

calculate_simi = False
if calculate_simi:
    from w2v_analyzer import w2v_analyzer
    w2vSource = 'GoogleNews'
    m2vmethod = 1  # methodType: 0: CBOW; 1: skip-gram
    w2vAnalyzer = w2v_analyzer(w2vSource, m2vmethod, 300, 40, filter_set)

    for i, theTopic in enumerate(myTopics):
        for keyword in theTopic:
            try:
                wv = w2vAnalyzer.model.wv[keyword]
            except KeyError:
                print 'In Topic' + str(i+1) + ', keyword: ' + keyword + ' not found in w2v vocabulary!'

    def transFun(x, M, k):
        if x < 0:
            return 0.0
        if x < 0.01:
            return 1.0
        assert x <= M
        y = 1.0 - 2.0/(1 + np.exp(k*(M/x-1)))
        return y

    w2vTokens_count = list()
    for text in texts:
        wvs, d = w2vAnalyzer.getSenMat(text.split())
        w2vTokens_count.append(d.shape[0])
    tweets_pd['w2vTokens_count'] = w2vTokens_count

    for theTopic, topicName in zip(myTopics, myTopicsNames):
        theTopicDiff = list()
        for text in texts:
            diff = w2vAnalyzer.getSenDiff(text.split(), theTopic, 1)
            theTopicDiff.append(diff)
        tweets_pd[topicName] = theTopicDiff

    simu = tweets_pd[myTopicsNames].values
    M = np.max(simu)*1.0001

    trans_simu = np.zeros((nTweets, len(myTopics)))
    k = 3.0
    for i in range(nTweets):
        for j in range(len(myTopics)):
            trans_simu[i, j] = transFun(simu[i, j], M, k)

    for j, theTopic in enumerate(myTopics):
        tweets_pd[myTopicsNames[j]+'_trans'] = trans_simu[:, j]

    # save the new results
    output_filename = "realDonaldTrump_tweets_analyzed.csv"
    output_path = os.path.join(os.path.expanduser("~"), "TrumpTwitterAnalysis", "Data", output_filename)
    tweets_pd.to_csv(output_path, encoding='utf-8', index=False)
else:
    input_filename = "realDonaldTrump_tweets_analyzed.csv"
    input_path = os.path.join(os.path.expanduser("~"), "TrumpTwitterAnalysis", "Data", input_filename)
    tweets_pd = pd.read_csv(input_path)

simi = tweets_pd[myTopicsNames].values
simi_trans = tweets_pd[[topicName+'_trans' for topicName in myTopicsNames]].values

from datetime import date, timedelta
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

#prop_cycle = plt.rcParams['axes.prop_cycle']
#colors = prop_cycle.by_key()['color']
from colors import cdict
colorNames = ['blue', 'black', 'red', 'brown', 'cyan', 'orange', 'skyblue', 'purple', 'tomato', 'yellow',
              'tan', 'magenta', 'green', 'darkblue', 'yellowgreen', 'gray']
colors = [cdict[cn] for cn in colorNames]


import time

dates = []
dates_day = []
simi_dates = []
nTweets_date = []
lastDate = date(2000, 1, 1)
nDays = 0
simi_trans_valid = None
for irow, row in tweets_pd.iterrows():
    if row['w2vTokens_count'] == 0:
        continue
    theDate = date(row['created_y'], row['created_m'], 1)  # row['created_d']
    if simi_trans_valid is None:
        simi_trans_valid = simi_trans[irow].reshape(1, -1)
    else:
        simi_trans_valid = np.vstack((simi_trans_valid, simi_trans[irow].reshape(1, -1)))

    if theDate != lastDate:
        nDays += 1
        dates.append(theDate)
        nTweets_date.append(1)
    else:
        nTweets_date[-1] += 1

    lastDate = theDate

plot_Topics = False
if plot_Topics:
    # show histogram of topic simularities
    plt.figure(0, figsize=(15, 12))
    #fig, axs = plt.subplots(4, 4, sharey=True)
    for j in range(nMyTopics):
        ax = plt.subplot(4, 4, j+1)
        ax.hist(simi_trans_valid[:, j], bins=40, color=cdict['dimgray'], histtype='bar')
        plt.xlim((-0.05, 1.05))
        plt.ylim((0, 500))
        plt.xticks([0, 0.25, 0.5, 0.75, 1])
        plt.yticks([0, 100, 200, 300, 400, 500])
        plt.text(1.0, 450, myTopicsNames[j], fontsize=15, horizontalalignment='right', verticalalignment='center', fontweight='bold', style='italic', color=colors[j])  #
    plt.savefig("topics_hist.png")

    # form days during the simulation period
    dates_day.append(dates[0])
    for i, theDate in enumerate(dates):
        for d in range(30, 1, -1):
            if theDate.month == 2 and d >= 29:
                continue
            theDate_day = date(theDate.year, theDate.month, d)
            if theDate_day < dates[0] and theDate_day > dates[-1]:
                dates_day.append(theDate_day)
    dates_day.append(dates[-1])

    ts_month = [time.mktime(theDate.timetuple()) for theDate in dates]
    ts_day = [time.mktime(theDate_day.timetuple()) for theDate_day in dates_day]

    irow = 0
    for nTweets_day in nTweets_date:
        dateSimu = np.mean(simi_trans_valid[irow:irow + nTweets_day], axis=0)
        irow += nTweets_day
        simi_dates.append(dateSimu)
    simu_dates = np.array(simi_dates)
    assert irow == simi_trans_valid.shape[0]
    assert len(nTweets_date) == simu_dates.shape[0]

    from sklearn.preprocessing import scale
    simu_dates = scale(simu_dates, axis=0, with_mean=True, with_std=True, copy=False)

    from scipy.interpolate import interp1d

    #plt.figure(1, figsize=(20, 10))
    plt.figure(1, figsize=(24, 8))
    #ax = plt.subplot(1, 1, 1)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

    # group the topics to show clearly
    #kind_set = [[0, 2, 5, 8], [1, 3, 10, 13], [4, 7, 9, 11], [6, 12, 14, 15]]
    #kind_set = [[0, 1, 9, 6], [2, 10, 4, 12], [5, 3, 7, 14], [8, 13, 11, 15]]
    kind_set = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]

    myTopicsNames_rearrange = list()
    for ks in kind_set:
        for k in ks:
            myTopicsNames_rearrange.append(myTopicsNames[k])
    ls = list()
    for k, ks in enumerate(kind_set):
        ax = plt.subplot(len(kind_set), 1, k+1)
        for j in range(nMyTopics):
            if j in ks:
                # plt.plot(dates, simu_dates[:, j], color=colors[j], label=myTopicsNames[j])
                #ax.scatter(dates, simu_dates[:, j], color=colors[j])

                f = interp1d(ts_month, simu_dates[:, j], kind='zero')  # kind='zero' 'nearest' 'slinear' 'quadratic'
                l, = ax.plot(dates_day, f(ts_day), color=colors[j], linewidth=3)  #
                ls.append(l)
                plt.xlim((dates_day[-1]+timedelta(days=-2), dates_day[0]+timedelta(days=2)))

                # labels of significant events
                if j == 0:
                    plt.text(date(2016, 6, 6), 1.71, '(1)', fontsize=10, horizontalalignment='center',
                             verticalalignment='bottom', fontweight='bold', style='italic', color=colors[j])
                    plt.text(date(2016, 7, 5), 2.27, '(2)', fontsize=10, horizontalalignment='center',
                             verticalalignment='bottom', fontweight='bold', style='italic', color=colors[j])
                    plt.text(date(2016, 7, 22), 2.27, '(4)', fontsize=10, horizontalalignment='center',
                             verticalalignment='bottom', fontweight='bold', style='italic', color=colors[j])
                    plt.text(date(2016, 9, 27), -0.12, '(6)', fontsize=10, horizontalalignment='center',
                             verticalalignment='bottom', fontweight='bold', style='italic', color=colors[j])
                    plt.text(date(2016, 10, 7), 1.3, '(7)', fontsize=10, horizontalalignment='center',
                             verticalalignment='bottom', fontweight='bold', style='italic', color=colors[j])
                    plt.text(date(2016, 10, 19), 1.3, '(9)', fontsize=10, horizontalalignment='center',
                             verticalalignment='bottom', fontweight='bold', style='italic', color=colors[j])
                    plt.text(date(2017, 3, 4), -0.30, '(18)', fontsize=10, horizontalalignment='center',
                             verticalalignment='bottom', fontweight='bold', style='italic', color=colors[j])
                elif j == 1:
                    plt.text(date(2017, 1, 10), 1.85, '(13)', fontsize=10, horizontalalignment='center',
                             verticalalignment='bottom', fontweight='bold', style='italic', color=colors[j])
                    plt.text(date(2017, 5, 28), 0.65, '(24)', fontsize=10, horizontalalignment='center',
                             verticalalignment='bottom', fontweight='bold', style='italic', color=colors[j])
                elif j == 2:
                    pass
                elif j == 3:
                    plt.text(date(2017, 1, 20), 0.9, '(14)', fontsize=10, horizontalalignment='center',
                             verticalalignment='top', fontweight='bold', style='italic', color=colors[j])
                elif j == 4:
                    plt.text(date(2017, 1, 22), 0.7, '(16)', fontsize=10, horizontalalignment='center',
                             verticalalignment='top', fontweight='bold', style='italic', color=colors[j])
                    plt.text(date(2017, 3, 24), 0.9, '(20)', fontsize=10, horizontalalignment='center',
                             verticalalignment='bottom', fontweight='bold', style='italic', color=colors[j])
                    plt.text(date(2017, 7, 26), 1.32, '(30)', fontsize=10, horizontalalignment='center',
                             verticalalignment='bottom', fontweight='bold', style='italic', color=colors[j])
                elif j == 5:
                    plt.text(date(2016, 9, 6), -0.51, '(5)', fontsize=10, horizontalalignment='center',
                             verticalalignment='bottom', fontweight='bold', style='italic', color=colors[j])
                    plt.text(date(2016, 11, 6), 2.25, '(12)', fontsize=10, horizontalalignment='center',
                             verticalalignment='bottom', fontweight='bold', style='italic', color=colors[j])
                elif j == 6:
                    plt.text(date(2017, 1, 22), 0.20, '(16)', fontsize=10, horizontalalignment='center',
                             verticalalignment='top', fontweight='bold', style='italic', color=colors[j])
                    plt.text(date(2017, 3, 6), -0.20, '(19)', fontsize=10, horizontalalignment='center',
                             verticalalignment='bottom', fontweight='bold', style='italic', color=colors[j])
                    plt.text(date(2017, 4, 12), 1.15, '(22)', fontsize=10, horizontalalignment='center',
                             verticalalignment='bottom', fontweight='bold', style='italic', color=colors[j])
                    plt.text(date(2017, 7, 4), 0.80, '(28)', fontsize=10, horizontalalignment='center',
                             verticalalignment='bottom', fontweight='bold', style='italic', color=colors[j])
                    plt.text(date(2017, 9, 3), 1.6, '(34)', fontsize=10, horizontalalignment='right',
                             verticalalignment='bottom', fontweight='bold', style='italic', color=colors[j])
                elif j == 7:
                    plt.text(date(2016, 7, 14), -0.70, '(3)', fontsize=10, horizontalalignment='center',
                             verticalalignment='bottom', fontweight='bold', style='italic', color=colors[j])
                    plt.text(date(2017, 1, 21), 1.3, '(15)', fontsize=10, horizontalalignment='center',
                             verticalalignment='bottom', fontweight='bold', style='italic', color=colors[j])
                    plt.text(date(2017, 8, 12), 1.2, '(31)', fontsize=10, horizontalalignment='center',
                             verticalalignment='bottom', fontweight='bold', style='italic', color=colors[j])
                elif j == 8:
                    plt.text(date(2016, 9, 6), 0.645, '(5)', fontsize=10, horizontalalignment='center',
                             verticalalignment='bottom', fontweight='bold', style='italic', color=colors[j])
                    plt.text(date(2016, 11, 8), 2.7, '(12)', fontsize=10, horizontalalignment='center',
                             verticalalignment='bottom', fontweight='bold', style='italic', color=colors[j])
                elif j == 9:
                    plt.text(date(2017, 7, 17), 0.58, '(29)', fontsize=10, horizontalalignment='center',
                             verticalalignment='bottom', fontweight='bold', style='italic', color=colors[j])
                elif j == 10:
                    plt.text(date(2017, 1, 20), 1.9, '(14)', fontsize=10, horizontalalignment='center',
                             verticalalignment='bottom', fontweight='bold', style='italic', color=colors[j])
                elif j == 11:
                    plt.text(date(2017, 6, 23), 0.92, '(26)', fontsize=10, horizontalalignment='center',
                             verticalalignment='bottom', fontweight='bold', style='italic', color=colors[j])
                elif j == 12:
                    plt.text(date(2017, 4, 3), 0.47, '(21)', fontsize=10, horizontalalignment='center',
                             verticalalignment='top', fontweight='bold', style='italic', color=colors[j])
                    plt.text(date(2017, 6, 3), 0.80, '(25)', fontsize=10, horizontalalignment='center',
                             verticalalignment='bottom', fontweight='bold', style='italic', color=colors[j])
                    plt.text(date(2017, 8, 17), 1.15, '(32)', fontsize=10, horizontalalignment='center',
                             verticalalignment='bottom', fontweight='bold', style='italic', color=colors[j])
                elif j == 13:
                    plt.text(date(2016, 7, 5), -0.24, '(2)', fontsize=10, horizontalalignment='center',
                             verticalalignment='bottom', fontweight='bold', style='italic', color=colors[j])
                    plt.text(date(2016, 10, 7), -0.1, '(8)', fontsize=10, horizontalalignment='center',
                             verticalalignment='bottom', fontweight='bold', style='italic', color=colors[j])
                    plt.text(date(2016, 10, 25), -0.1, '(10)', fontsize=10, horizontalalignment='center',
                             verticalalignment='bottom', fontweight='bold', style='italic', color=colors[j])
                    plt.text(date(2016, 11, 6), 0.55, '(11)', fontsize=10, horizontalalignment='center',
                             verticalalignment='top', fontweight='bold', style='italic', color=colors[j])
                    plt.text(date(2017, 1, 10), 0.9, '(13)', fontsize=10, horizontalalignment='center',
                             verticalalignment='top', fontweight='bold', style='italic', color=colors[j])
                    plt.text(date(2017, 5, 9), 1.3, '(23)', fontsize=10, horizontalalignment='center',
                             verticalalignment='bottom', fontweight='bold', style='italic', color=colors[j])
                elif j == 14:
                    plt.text(date(2017, 1, 22), 2.1, '(16)', fontsize=10, horizontalalignment='center',
                             verticalalignment='bottom', fontweight='bold', style='italic', color=colors[j])
                    plt.text(date(2017, 8, 22), 0.8, '(33)', fontsize=10, horizontalalignment='center',
                             verticalalignment='top', fontweight='bold', style='italic', color=colors[j])
                else:
                    plt.text(date(2017, 1, 27), 1.5, '(17)', fontsize=10, horizontalalignment='center',
                             verticalalignment='bottom', fontweight='bold', style='italic', color=colors[j])
                    plt.text(date(2017, 6, 26), 0.22, '(27)', fontsize=10, horizontalalignment='center',
                             verticalalignment='top', fontweight='bold', style='italic', color=colors[j])

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
        ax.titlesize = 12

        dates_forward = dates
        dates_forward.reverse()
        ax.set_xticks(dates_forward)
        ax.set_xticklabels(dates_forward, rotation=90)
        plt.ylim((-2.5, 3.0))
        plt.ylabel('\'Hotness\'', color='black')  # , fontweight='bold'
    plt.legend(ls, myTopicsNames_rearrange, bbox_to_anchor=(1.13, 4.65), ncol=1, fontsize=10, labelspacing=1.5, frameon=False)
    #plt.legend(ls, myTopicsNames_rearrange, bbox_to_anchor=(1.13, 4.65), ncol=1, fontsize=12, labelspacing=1.0, frameon=False)
    plt.gcf().autofmt_xdate()
    plt.savefig("topics_date.png")

# extract data
delta_hour = 1
hours_grid = np.linspace(0, 24-delta_hour, 24/delta_hour)
hours_grid = [int(h) for h in hours_grid]
nhgrid = len(hours_grid)

count_times = np.zeros(nhgrid)
topic_times = np.zeros((nhgrid, nMyTopics))
X = None
Y = None
for irow, row in tweets_pd.iterrows():
    if row['w2vTokens_count'] == 0 or irow >= nTweets_full:
        continue

    rX = row[[topicName + '_trans' for topicName in myTopicsNames]].values.reshape(1, -1)
    rY = row[['favorite_count', 'retweet_count']].values.reshape(1, -1)

    if X is None:
        X = rX
        Y = rY
    else:
        X = np.vstack((X, rX))
        Y = np.vstack((Y, rY))

    h = row['created_h']
    ih = nhgrid - 1
    for i in range(nhgrid - 1):
        if h >= hours_grid[i] and h < hours_grid[i + 1]:
            ih = i
            break
    count_times[ih] += 1
    for j, topicName in enumerate(myTopicsNames):
        topic_times[ih, j] += row[topicName + '_trans']
# percentage of tweets as function of time
count_times /= sum(count_times)
# calculate the relative strength
for j in range(nMyTopics):
    topic_times[:, j] /= count_times
# normalize
from sklearn.preprocessing import normalize
topic_times = normalize(topic_times, norm='l1', axis=0)

nD = X.shape[0]
y = Y[:, 0] # favorite or retweet?
from transformers import LabelConverter
label = LabelConverter(0.5).fit_transform(y)

# plot time analysis
if plot_Topics:
    from datetime import time

    x_times = list()
    for i in range(nhgrid):
        x_times.append(time(hours_grid[i], 30, 0))

    delta_ticks = 2
    hours_ticks = np.linspace(0, 24 - delta_ticks, 24 / delta_ticks)
    xtick_times = [time(int(h), 0, 0) for h in hours_ticks]

    # plot percentage of tweets
    plt.figure(2)
    ax1 = plt.subplot(1, 1, 1)
    plt.plot(x_times, count_times, '-k', linewidth=3)
    ax1.set_xticks(xtick_times)
    plt.xlim((time(0, 0, 0), time(23, 59, 59)))
    plt.xlabel('Time')
    plt.ylabel('Percentage of tweets')
    plt.gcf().autofmt_xdate()
    plt.savefig("tweets_time.png")

    # plot relative strength of topics
    plt.figure(3, figsize=(24, 8))
    ls = list()
    for k, ks in enumerate(kind_set):
        #ax = plt.subplot(2, 2, k + 1)
        ax = plt.subplot(len(kind_set), 1, k + 1)
        for j in range(nMyTopics):
            if j in ks:
                l, = ax.plot(x_times, topic_times[:, j], color=colors[j], linewidth=3)
                ls.append(l)
        ax.set_xticks(xtick_times)
        ax.set_yticks([0.04, 0.05])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.titlesize = 12
        plt.xlim((time(0, 0, 0), time(23, 59, 59)))
        #plt.ylim((0.035, 0.055))
        plt.xlabel('Time')
        plt.ylabel('Intensity')
        plt.gcf().autofmt_xdate()

    plt.legend(ls, myTopicsNames_rearrange, bbox_to_anchor=(1.13, 4.65), ncol=1, fontsize=10, labelspacing=1.5, frameon=False)
    plt.savefig("topics_time.png")


# regression
analyze_regression = False
if analyze_regression:
    from regression_analysis import filterSimu
    from regression_analysis import regression_analysis

    # 0: linear regression; 1: decision tree; 2: adaboost; 3: random forest; 4: GBR
    methodType = 3

    fS = filterSimu(0.97)
    fS.fit(X)
    X = fS.transform(X)

    regressor, label_pred, score = regression_analysis(X, label, methodType)
    print regressor.coef_
    print regressor.intercept_
    print regressor.feature_importances_

    # Plot the results
    if plot_Topics:
        D = np.linspace(0, nD - 1, nD)
        plt.figure()
        plt.scatter(D, label, c="k", label="training samples")
        plt.plot(D, label_pred, c="r", label="predict", linewidth=1)
        plt.xlabel("data")
        plt.ylabel("target")
        plt.title("Regression")
        plt.legend()

# classfication
analyze_classification = True
if analyze_classification:
    from classifier_analysis import classifier_analysis
    methodType = 6
    clf, label_pred, score = classifier_analysis(X, label, methodType)
    print score



if plot_Topics:
    plt.show()


