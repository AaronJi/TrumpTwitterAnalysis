# -- coding: utf-8 --

from gensim import corpora, models, similarities
import logging
import sys
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sys.stdout.flush()

'''
Latent Semantic Indexing/Analysis
'''
def LSI_analysis(texts, nTopics):
    # 抽取一个“词袋（bag-of-words)"，将文档的token映射为id
    dictionary = corpora.Dictionary(texts)

    # 将用字符串表示的文档转换为用id表示的文档向量
    corpus = [dictionary.doc2bow(text) for text in texts]
    #print 'Corpus: '
    #print corpus

    # 基于这些“训练文档”计算一个TF-IDF“模型
    tfidf = models.TfidfModel(corpus)
    #print 'tfidf model:'
    #print tfidf.dfs
    #print tfidf.idfs

    # 将上述用词频表示文档向量表示为一个用tf-idf值表示的文档向量
    corpus_tfidf = tfidf[corpus]
    print 'Text vector formed by tf-idf'
    #for doc in corpus_tfidf:
        #print doc

    # 训练一个LSI模型
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=nTopics)
    print 'Top 2 topics of LSI model:'
    print lsi.print_topics(2)

    # 将文档映射到一个n维的topic空间中
    print 'The text projection in the n-dim topic space:'
    corpus_lsi = lsi[corpus_tfidf]  # n of doc * n of topic
    for doc in corpus_lsi:
        print doc

    # LDA模型; lda模型中的每个主题单词都有概率意义，其加和为1，值越大权重越大，物理意义比较明确
    #lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=nTopics)
    #print 'Top 2 topics of LDA model:'
    #print lda.print_topics(2)

    # 将文档映射到一个二维的topic空间中
    #print 'The text projection in the n-dim topic space:'
    #corpus_lda = lda[corpus_tfidf]
    #for doc in corpus_lda:
        ##print doc

    # 计算文档之间的相似度，或者给定一个查询Query，如何找到最相关的文档: 首先建索引
    index = similarities.MatrixSimilarity(lsi[corpus])

    # 将query向量化
    query = "shipment of silver arrived"
    query_bow = dictionary.doc2bow(query.lower().split())
    print "query: " + query + "; the bow vector: "
    #print query_bow

    # 用之前训练好的LSI模型将其映射到n维的topic空间
    query_lsi = lsi[query_bow]

    print 'The projection of query in the n-dim topic space:'
    print query_lsi

    # 计算其和index中doc的余弦相似度
    sims = index[query_lsi]

    print 'The cos simularity between query and doc:'
    print sims
    print list(enumerate(sims))

    # 也可以按相似度进行排序
    sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
    print sort_sims

    query_tfidf = tfidf[query_bow]
    query_tfidf_lsi = lsi[query_tfidf]
    index_tfidf = similarities.MatrixSimilarity(lsi[corpus_tfidf])
    sims_tfid = index_tfidf[query_tfidf_lsi]
    print 'The cos simularity between query and doc:'
    print list(enumerate(sims_tfid))
    return

if __name__ == '__main__':
    doc1 = "Shipment of gold damaged in a fire"
    doc2 = "Delivery of silver arrived in a silver truck"
    doc3 = "Shipment of gold arrived in a truck"
    documents = [doc1.split(), doc2.split(), doc3.split()]

    LSI_analysis(documents, 2)
