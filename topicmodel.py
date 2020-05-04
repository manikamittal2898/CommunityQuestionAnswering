import pandas as pd
from numpy import array
import nltk

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import pyLDAvis
import pyLDAvis.gensim 
import matplotlib.pyplot as plt

data = pd.read_csv('/home/manika/Desktop/padhai/IR/hojaabhai.csv', error_bad_lines=False)

data['question'] = data['subject'].str.cat(data['content'], sep ="") 

data['tokens'] = data.apply(lambda row: nltk.word_tokenize(row['question']), axis=1)

dictionary = gensim.corpora.Dictionary(data['tokens'])
# count = 0
# for k, v in dictionary.iteritems():
#     print(k, v)
#     count += 1
#     if count > 10:
#         break
dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=100000)
bow_corpus = [dictionary.doc2bow(doc) for doc in data['tokens']]
# bow_corpus[10]
# print(bow_corpus)
from gensim import corpora, models
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
from pprint import pprint
# print([[(dictionary[id], freq) for id, freq in cp] for cp in corpus_tfidf[:2819]])
# for doc in corpus_tfidf:
#     pprint(doc)
    # break
print(len(dictionary))
# lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
# for idx, topic in lda_model.print_topics(-1):
#     print('Topic: {} \nWords: {}'.format(idx, topic))





# def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
#     """
#     Compute c_v coherence for various number of topics

#     Parameters:
#     ----------
#     dictionary : Gensim dictionary
#     corpus : Gensim corpus
#     texts : List of input texts
#     limit : Max num of topics

#     Returns:
#     -------
#     model_list : List of LDA topic models
#     coherence_values : Coherence values corresponding to the LDA model with respective number of topics
#     """
#     coherence_values = []
#     model_list = []
#     for num_topics in range(start, limit, step):
#         model = gensim.models.LdaMulticore(corpus_tfidf, num_topics=num_topics, id2word=dictionary, passes=10, workers=4)
#         # model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
#         model_list.append(model)
#         coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
#         coherence_values.append(coherencemodel.get_coherence())

#     return model_list, coherence_values

# model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus_tfidf, texts=data['tokens'], start=2, limit=40, step=2)

# # Show graph
# limit=40
# start=2
# step=2
# x = range(start, limit, step)
# plt.plot(x, coherence_values)
# plt.xlabel("Num Topics")
# plt.ylabel("Coherence score")
# plt.legend(("coherence_values"), loc='best')
# plt.show()

lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=8, id2word=dictionary, passes=10, workers=4)
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))
for index, score in sorted(lda_model_tfidf[bow_corpus[0]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))

coherence_model_lda = CoherenceModel(model=lda_model_tfidf, texts=data['tokens'],dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)



vis = pyLDAvis.gensim.prepare(lda_model_tfidf, corpus_tfidf, dictionary)
# vis
pyLDAvis.save_html(vis, 'LDA_Visualization8.html')
docs=[]
c=0
for c in range(0,len(bow_corpus)):
    vector=[]
    for index, score in lda_model_tfidf[bow_corpus[c]]:
        print("\nScore: {}\t \nTopic: {}".format(score, index))
        vector.append(score)
    c=c+1
    docs.append(vector)
print(len(bow_corpus)-1)
print(len(data['question']))
data['questopics']=docs
print(data['questopics'])
# df2 = df.assign(address = ['Delhi', 'Bangalore', 'Chennai', 'Patna'])