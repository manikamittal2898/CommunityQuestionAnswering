import gensim
from nltk.stem import WordNetLemmatizer
import numpy as np

np.random.seed(2018)
import nltk
nltk.download('wordnet')

import csv
import openpyxl 

wb = openpyxl.Workbook() 

sheet = wb.active 
with open("/home/manika/Desktop/padhai/IR/Dataset1.csv") as csvDataFile:
    data = [row for row in csv.reader(csvDataFile)]

lemmatizer= WordNetLemmatizer()
def lemmatize_stemming(text):
    return lemmatizer.lemmatize(text) 
def preprocess(text):

    string = ' '
    for token in gensim.utils.simple_preprocess(text):

        if token not in gensim.parsing.preprocessing.STOPWORDS:
            string+= " " + (lemmatize_stemming(token))
    return string

flag=-1

for rows in data:
    flag+=1
    flag2=-1
    for columns in rows:
        flag2+=1
        doc_sample = data[flag][flag2]
        print('original document: ')
      
        print(doc_sample)
        print('tokenized and lemmatized document: ')
        print(preprocess(doc_sample))
        c1 = sheet.cell(row = flag+1, column = flag2+1) 
  
        # writing values to cells 
        c1.value = (preprocess(doc_sample))
wb.save(r"/home/manika/Desktop/padhai/IR/file.xlsx")

