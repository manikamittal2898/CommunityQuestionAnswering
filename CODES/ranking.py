#Final score sorted and alloted rank
import numpy as np
import pandas as pd
import ast 
import csv

dataset= pd.read_csv('topic_modelled_final.csv')
print(dataset.info())
with open('topic_modelled_final.csv') as csvDataFile:
    data = [row for row in csv.reader(csvDataFile)]

print(dataset['modified_cos'][0])
print(data[0][39])
flag=0
ranking=[]

#ranking performed
for rows in data:
  while flag<1175:
    flag+=1
    for columns in rows:
      a=[]

      cosinescores= data[flag][39]
      res = ast.literal_eval(cosinescores)
      flag2=10
      while flag2!=0:
        flag2-=1
        max_val= max(res)
        index_of_max= res.index(max_val) #max_val checked and its index is added to list
        if max_val!=0:
          
          a.append(index_of_max)
          res[index_of_max]=-1
        else:
          a.append(-1)
          res[index_of_max]=-1
    ranking.append(a)
dataset['ranking_modified']=ranking

#accuracy calculated
acc=0
for i in range(0,len(dataset)):
  a= dataset['ranking_modified'][i]
  b= a[0]
  if b== dataset['bestansnumber'][i]:
    acc+=1
print(acc)

