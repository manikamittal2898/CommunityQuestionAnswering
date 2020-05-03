import gensim
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
np.random.seed(2018)
import nltk
nltk.download('wordnet')

import csv

with open("/home/manika/Desktop/padhai/IR/Dataset.csv") as csvDataFile:
    data = [row for row in csv.reader(csvDataFile)]
    data=data[1:5]
lemmatizer= WordNetLemmatizer()
def lemmatize_stemming(text):
    return lemmatizer.lemmatize(text) 
def preprocess(text):
    result = ""
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result=result+" "+lemmatize_stemming(token)
    return result

flag=-1

listofrows=[]
for rows in data:
    flag+=1
    flag2=-1
    for columns in rows:
        col=[]
        flag2+=1
        doc_sample = data[flag][flag2]
        # print(doc_sample)
        print('original document: ')
        # words = []
        # for word in doc_sample.split(' '):
        #     words.append(word)
        #     # if words.count==0:
        #     #     continue
        #     # else: 
        print(doc_sample)
        print('\n\n tokenized and lemmatized document: ')
        print(preprocess(doc_sample))
        col.append(preprocess(doc_sample))
    listofrows.append(col)
 	
# Creating a dataframe object from listoftuples
# dfObj = pd.DataFrame(students,) 
final_doc=pd.DataFrame(listofrows)
final_doc.to_csv('file1.csv')