#checking for heuristics

#importing libraries
import numpy as np
import pandas as pd
import ast 
import csv

#importing data into dataframe
dataset= pd.read_csv('preprocessed_withlinks.csv')
print(dataset.info())
with open('preprocessed_withlinks.csv') as csvDataFile:
    data = [row for row in csv.reader(csvDataFile)]

#checking for presence of links in answers and adding to list
col_links=[]
for i in range(1,len(dataset)+1):
  link_list=[]
  for j in range(3,13):
    ans= data[i][j]
    if 'http' in ans:
      link_list.append(1)
    elif len(ans)==0:
      link_list.append(-1)
    else:
      link_list.append(0)
  col_links.append(link_list)

print(len(col_links))

dataset['links']=col_links
print(dataset)

#downloading dataframe
from google.colab import files
dataset.to_csv('links_preprocessed.csv') 
files.download('links_preprocessed.csv')
#links lists achieved for 1175 entries 