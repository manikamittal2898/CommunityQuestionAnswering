#linear regression with cosine and links on rank
import numpy as np
import pandas as pd
import ast 

dataset= pd.read_csv('rankings (1).csv')
rank_list= dataset['ranking'][1]
ranks= ast.literal_eval(rank_list)
print(type(ranks))

#taking X1 as presence of links and X2 as cosine score 
#y signifies rank of answer
col_x=[]
col_y=[]
for i in range(0,len(dataset)):
  rank_list= dataset['ranking'][i]
  ranks= ast.literal_eval(rank_list)
  
  links_list = dataset['links'][i]
  links=ast.literal_eval(links_list)

  cos_list= dataset['cosinescores'][i]
  cosvals= ast.literal_eval(cos_list)

  for j in range(0,10):
    tuples=[]
    y=ranks[j]
    if y!=-1:
      x1=links[j]    
      x2=cosvals[j]
      tuples.append(x1)
      tuples.append(x2)
      col_x.append(tuples)
      col_y.append(y)
    else:
      continue

data = {'Y':col_y, 'X':col_x} 
print(col_x)

# Create DataFrame 
df = pd.DataFrame(data)

#perform linear regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
model=regressor.fit(col_x,col_y)  
score=model.score(col_x,col_y)
print(score)     
print(model.intercept_) 
print(model.coef_)

