import pandas as pd
import numpy as np
from numpy import array
from numpy import dot
from numpy.linalg import norm
import nltk

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import pyLDAvis
import pyLDAvis.gensim 
import matplotlib.pyplot as plt

data = pd.read_csv('/home/manika/Desktop/padhai/IR/finaldataver2.csv', error_bad_lines=False)
print(data.keys())
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
# print(len(dictionary))
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
np.random.seed(1)
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=8, id2word=dictionary, passes=10, workers=4)
# for idx, topic in lda_model_tfidf.print_topics(-1):
# #     # print('Topic: {} Word: {}'.format(idx, topic))
# for index, score in sorted(lda_model_tfidf[bow_corpus[0]], key=lambda tup: -1*tup[1]):
#      print("\nScore: {}\t \nTopic: {}".format(score, index))

coherence_model_lda = CoherenceModel(model=lda_model_tfidf, texts=data['tokens'],dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
# print('\nCoherence Score: ', coherence_lda)



# vis = pyLDAvis.gensim.prepare(lda_model_tfidf, corpus_tfidf, dictionary)
# # vis
# pyLDAvis.save_html(vis, 'LDA_Visualization8.html')
#creating vectors for questions
docs=[]

c=0
for c in range(0,len(bow_corpus)):
    counter=0
    vector=[]
    for index, score in lda_model_tfidf[bow_corpus[c]]:
        # print("\nScore: {}\t \nTopic: {}".format(score, index))
        while index!=counter:
            vector.append(0.0)
            counter+=1
        vector.append(score)
        counter+=1
    #number of topics is 8

        
      
    # while(len(vector)!=8):
    #     vector.append(0)
    while counter!=8:
        vector.append(0.0)
        counter+=1   
    c=c+1
    docs.append(vector)
# print(len(bow_corpus)-1)
# print(len(data['question']))
data['questopics']=docs
# print(data['questopics'])

#creating answer vectors
# bow_corpus = [dictionary.doc2bow(doc) for doc in data['token']
# tokens0=[]
# tokens0=data.apply(lambda row: nltk.word_tokenize(row['ans0']), axis=1)
# print(tokens0)
list0 = [dictionary.doc2bow(doc) for doc in data.apply(lambda row: nltk.word_tokenize(row['ans0']), axis=1)]
list1 = [dictionary.doc2bow(doc) for doc in data.apply(lambda row: nltk.word_tokenize(row['ans1']), axis=1)]
list2 = [dictionary.doc2bow(doc) for doc in data.apply(lambda row: nltk.word_tokenize(row['ans2']), axis=1)]
list3 = [dictionary.doc2bow(doc) for doc in data.apply(lambda row: nltk.word_tokenize(row['ans3']), axis=1)]
list4 = [dictionary.doc2bow(doc) for doc in data.apply(lambda row: nltk.word_tokenize(row['ans4']), axis=1)]
list5 = [dictionary.doc2bow(doc) for doc in data.apply(lambda row: nltk.word_tokenize(row['ans5']), axis=1)]
list6 = [dictionary.doc2bow(doc) for doc in data.apply(lambda row: nltk.word_tokenize(row['ans6']), axis=1)]
list7 = [dictionary.doc2bow(doc) for doc in data.apply(lambda row: nltk.word_tokenize(row['ans7']), axis=1)]
list8 = [dictionary.doc2bow(doc) for doc in data.apply(lambda row: nltk.word_tokenize(row['ans8']), axis=1)]
list9 = [dictionary.doc2bow(doc) for doc in data.apply(lambda row: nltk.word_tokenize(row['ans9']), axis=1)]
anslist=[]
anslist.append(list0)
anslist.append(list1)
anslist.append(list2)
anslist.append(list3)
anslist.append(list4)
anslist.append(list5)
anslist.append(list6)
anslist.append(list7)
anslist.append(list8)
anslist.append(list9)
# print(anslist[0][0])
# print(bow_corpus[0])

for i in range(0,len(anslist)):
    docs=[]
    c=0
    for c in range(0,len(anslist[i])):
        counter=0
        vector=[]
        for index, score in lda_model_tfidf[anslist[i][c]]:
            # print("\nScore: {}\t \nTopic: {}".format(score, index))
            while index!=counter:
                vector.append(0.0)
                counter+=1
            vector.append(score)
            counter+=1
    #number of topics is 8

        
      
    # while(len(vector)!=8):
    #     vector.append(0)
        while counter!=8:
            vector.append(0.0)
            counter+=1   
            # vector.append(score)
        c=c+1
        docs.append(vector)

    data['ans'+str(i)+'topics']=docs
# for i in range(0,len(anslist)):
#     print(data['ans'+str(i)+'topics'])
# print(data.keys())

# data.to_csv('topic_modelled.csv')

# dfObj= pd.DataFrame(columns=['ans0sim','ans1sim','ans2sim','ans3sim','ans4sim','ans5sim','ans6sim','ans7sim','ans8sim','ans9sim'])

# def cosine_val(a,b):
#       cos_sim = np.dot(a, b)/(norm(a)*norm(b))
#   return cos_sim
#print(cos_sim)


# flag=-1
# rows=0
# columns=15
# for rows in data:
#     x=0
#     columns=15
#     for columns in rows:
#       a= data[rows][columns]
#       while columns<26:
#           columns+=1
#           b= data[rows][columns]
#           if b*b!=2.165:
#               similarity=cosine_val(a,b)
#               dfObj[rows][x]=similarity
#               x+=1
#           else:
#               dfObj[flag][flag3]='Null'
#               flag3+=1


# for index in range(data.shape[1]):
#        print('Column Number : ', index)
#    # Select column by index position using iloc[]
#    columnSeriesObj = empDfObj.iloc[: , index]
#    print('Column Contents : ', columnSeriesObj.values)

# print(np.dot([1,-1],[0,1]))
answerscol=data.iloc[:,16:] 
# b=data.iloc[:,15]
columns=list(answerscol)
r=0
sim=[]
# a=[]
# b=[]
counter=0
for r in range(0,len(data)):
    row_sim=[]
    for i in columns:
    # if r<len(data):
        a=np.array(data['questopics'][r])
        b= np.array(data[i][r])
        # counter+=1
        # if cmp(a.shape,b.shape)==1:
        c=data[i][r]
        result = False
        result=all(elem==c[0] for elem in c)
        if result:
            row_sim.append(0)
        else:
            row_sim.append(np.dot(a,b)/(norm(a)*norm(b)))
        #####print(str(a*b)+"  count is "+ str(counter))
        # else:
        #     row_sim.append("NULL")

    sim.append(row_sim)
data['cosinescores']=sim
print(data['cosinescores'])


data.to_csv('topic_modelled.csv')


