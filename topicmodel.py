import pandas as pd

import nltk

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
# matplotlib inline
# from imp import reload

data = pd.read_csv('/home/manika/Desktop/padhai/IR/hojaabhai.csv', error_bad_lines=False)
# data.head()
# data_text = data[['subject']]
# data_text['index'] = data_text.index
# documents = data_text
# print(len(documents))
# print(data[:5])
# print(data.iloc[:]['subject'])
# print(data.keys())
data['question'] = data['subject'].str.cat(data['content'], sep ="") 
# print(data.keys())
# print(data['question'])
data['tokens'] = data.apply(lambda row: nltk.word_tokenize(row['question']), axis=1)
# print(data['tokens'])
# print(data)
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
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=8, id2word=dictionary, passes=10, workers=4)
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))
for index, score in sorted(lda_model_tfidf[bow_corpus[0]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 8)))


coherence_model_lda = CoherenceModel(model=lda_model_tfidf, texts=data['tokens'],dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)



vis = pyLDAvis.gensim.prepare(lda_model_tfidf, corpus_tfidf, dictionary)
# vis
pyLDAvis.save_html(vis, 'LDA_Visualization8.html')