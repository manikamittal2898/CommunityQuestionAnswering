# this model creates the reference model
#based on WordCount approach, and computes cosine similarity between the 
#question and answer vectors



import numpy as np
import pandas as pd
import ast 
import csv
import math
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

data = pd.read_csv('/home/manika/Desktop/padhai/IR/preprocessed.csv', error_bad_lines=False)
print(data.keys())
data=data.astype(str)
np.random.seed(2018)
data['question'] = data['subject'].str.cat(data['content'], sep =" ") 
print(data['question'])
data['tokens'] = data.apply(lambda row: nltk.word_tokenize(row['question']), axis=1)

dictionary = gensim.corpora.Dictionary(data['tokens'])

bow_corpus = [dictionary.doc2bow(doc) for doc in data['tokens']]
# bow_corpus[10]
# print(bow_corpus)
from gensim import corpora, models
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
from pprint import pprint
# print("Length of vocabulary is "+ str(len(dictionary)))
# for doc in corpus_tfidf:
#     pprint(doc)
#     print(doc[0][1])
#     break
    
# model = Word2Vec(data['question'], min_count=1,size= 4000,workers=3, window =3, sg = 1)

docs=[]
for c in range(0,len(bow_corpus)):
    
    vector=[]
    # print(len(bow_corpus))
    # print(len(corpus_tfidf[c]))
    # break
    i=0
    for word in range(0,len(corpus_tfidf[c])):
        while i != corpus_tfidf[c][word][0]:
            vector.append(0.0)
            i=i+1
        vector.append(corpus_tfidf[c][word][1])
    while i!=(len(dictionary)-1):
        vector.append(0.0)
        i+=1   
    docs.append(vector)
data['questionvector']=docs

# print(data['questionvector'])
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
        index=0
        vector=[]
        for word in range(0,len(corpus_tfidf[c])):
            while index != corpus_tfidf[c][word][0]:
                vector.append(0.0)
                index=i+1
            vector.append(corpus_tfidf[c][word][1])
        while index!=(len(dictionary)-1):
            vector.append(0.0)
            index+=1   
    docs.append(vector)
    print(docs)
data['ans'+str(i)+'vector']=docs

answerscol=data.iloc[:,16:] 
# b=data.iloc[:,15]
columns=list(answerscol)
r=0
sim=[]

counter=0
for r in range(0,len(data)):
    row_sim=[]
    for i in columns:
    # if r<len(data):
        a=np.array(data['quesvector'][r])
        b= np.array(data[i][r])
        # counter+=1
        # if cmp(a.shape,b.shape)==1:
        # c=data[i][r]
        # result = False
        # result=all(elem==c[0] for elem in c)
        # if result:
        #     row_sim.append(0)
        # else:
        row_sim.append(np.dot(a,b)/(norm(a)*norm(b)))
        
    sim.append(row_sim)
data['cosinescores']=sim
# print(data['cosinescores'])

data.to_csv('ref_data.csv')
