import pandas as pd 
import xml.etree.ElementTree as et 
import csv

xtree = et.parse("/home/manika/Desktop/padhai/IR/small_sample.xml")
root = xtree.getroot()
thread=[]
# print(root.tag)
for item in root.iter('vespaadd'):
    root1=et.Element('root')
    root1=item
# print(root1.tag)
    ques={}
    for child in root1.findall('document'):
        ques['subject']=child.find('subject').text.encode('utf-8')
        ques['content']=child.find('content').text.encode('utf-8')
        ques['bestanswer']=child.find('bestanswer').text.encode('utf-8')
        root2=et.Element('root1')
        root2=child.find('nbestanswers')
 # print(root2.tag)
        c=0
        for child in root2.findall('answer_item'):
            ques['ans'+str(c)]=child.text.encode('utf-8')
            c+=1
        while(c<10):
            ques['ans'+str(c)]='None'
            c+=1
    thread.append(ques)
for x in range(len(thread)): 
    print thread[x]
    print ' \n'

fields = ["subject", "content", "bestanswer","ans0","ans1","ans2","ans3","ans4","ans5","ans6","ans7","ans8","ans9"]

# writing to csv file 
with open("dataset1.csv", 'w') as csvfile: 

# creating a csv dict writer object 
    writer = csv.DictWriter(csvfile, fieldnames = fields) 

# writing headers (field names) 
    writer.writeheader() 

# writing data rows 
    writer.writerows(thread) 