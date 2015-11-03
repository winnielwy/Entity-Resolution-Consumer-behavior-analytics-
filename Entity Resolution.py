# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 11:46:07 2015

@author: wenyingliu
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from pandas import DataFrame, Series

import fileinput
import string
import nltk


#data cleaning
def remove_punctuation(s):
    s = ''.join([i for i in s if i not in frozenset(string.punctuation)])
    return s

##remove generic company tail
com_tail=pd.read_csv("Company Identifier Abbreviations.csv")    
           
com_tail['co_tail'] = com_tail.co_tail.str.lower()
com_tail['co_tail'] = com_tail['co_tail'].apply(remove_punctuation)

comtail=tuple(com_tail['co_tail'] .tolist())

def rm_suffix(s):
    for suf in comtail:
       if s.endswith(suf):
          return s[:-len(suf)]
    return s
    
def cleancom(col):
    col1=col.astype('string')
    col2=col1.fillna("")
    col3=col2.str.lower()
    col4=col3.str.strip()
    col5=col4.fillna("")
    col6=col5.apply(remove_punctuation)
    col7=col6.apply(rm_suffix)
    return col7
    
def cleanemail(col):
    col1=col.astype('string')
    col2=col1.fillna("")
    col3=col2.str.lower()
    col4=col3.str.strip()
    col5=col4.fillna("")
    return col5
    
def emailtail(col):
    col1=col.astype('string')
    col2=col1.fillna("")
    col3=col2.str.lower()
    col4=col3.str.strip()
    col5=col4.fillna("")
    col6=col5.apply(lambda x:x.split('@', 1)[-1])
    return col6

##Test if it is generic tail
gen_tail=pd.read_csv("generic email tails set.csv")    
           
gen_tail['unietail'] = gen_tail.unietail.str.lower()
gen_tail['unietail'] =gen_tail.unietail.apply(lambda x:x.replace('@',''))
gentail=tuple(gen_tail['unietail'].tolist())

def alert(c):
  if c in gentail:
    return 'generic'
  else:
    return 'unique'
    
def unitest(col):
    col1=col.astype('string')
    col2=col1.fillna("")
    col3=col2.str.lower()
    col4=col3.str.strip()
    col5=col4.fillna("")
    col6=col5.apply(lambda x:x.split('@', 1)[-1])
    col7=col6.apply(alert)
    return col7
    
#entity resolution
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models, similarities      

def make_unicode(input):
    if type(input) != unicode:
        input =  input.decode('latin-1')
        return input
    else:
        return input

def entity(target):
        cell = []     
        for t in target:
            if t not in cell:
                 cell.append(t)
        
        counts=target.value_counts()  
        
        unicode_pp=[]
        for i in cell:
            unicode_pp.append(make_unicode(i))
            

        tokenizer = RegexpTokenizer(r'\w+')
        tokenized = [tokenizer.tokenize(p) for p in unicode_pp]
        porter_stemmer = PorterStemmer()
        cleaned = [[porter_stemmer.stem(word) for word in p] for p in tokenized]

        dictionary = corpora.Dictionary(cleaned)
        corpus = [dictionary.doc2bow(text) for text in cleaned]

        tfidf = models.TfidfModel(corpus)
        c_tfidf = tfidf[corpus]
        
        lsi = models.LsiModel(c_tfidf, id2word=dictionary, num_topics=len(cell))
        corpus_lsi = lsi[c_tfidf]
        index = similarities.MatrixSimilarity(tfidf[corpus])
        
        #test
        sort=[]
        for test_data in unicode_pp:
            input=make_unicode(test_data.decode('latin-1'))
            query = dictionary.doc2bow([porter_stemmer.stem(w) for w in [t for t in tokenizer.tokenize(input.lower())]])
            query_lsi = tfidf[query]
            sims = index[query_lsi]
            sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
            sort.append(sort_sims)
        #accuracy and similarity for each cell    
        accu=[]
        sim_num=[]
        for cell in sort:
            n0=[]
            n1=[]
            
            for i in cell:
                t0=i[0]
                t1=i[1]
                if t1>threshold:
                    n0.append(t0)
                    n1.append(t1)
            sim_num.append(n0)
            accu.append(n1)
         #similarity number for each cell   
        com_num=[]
        for i in range(0,len(accu)):
             num=len(accu[i])
             com_num.append(num)
        
        sim_name=[]
        for i in sim_num:
            company=[]        
            for j in i:
                clean_name=unicode_pp[j].encode('ascii', 'ignore')
                company.append(clean_name)
            sim_name.append(company)

        clean_company=[]
        for name in unicode_pp:
            clean_company.append(name.encode('ascii', 'ignore'))

        mydict={'company':clean_company,'sim_company':sim_name,'accuracy':accu,'cell':sim_num,'number':com_num}
        myDF=pd.DataFrame(mydict)
        
        com=myDF['company'].tolist()
        sim_com=myDF['sim_company'].tolist()
    
        t=[]
        newrow=[]
        lst=[]
        col=0
        for i in range(len(com)):
            found_flag=False
            for row,lst in enumerate(t):
                for col,sim_company in enumerate(lst):
                    if com[i]==sim_company:
                        found_flag=True
                        newrow =list(set(t[row]+ sim_com[i]))
                        t[row] = newrow
            if found_flag==False:
                t.append(sim_com[i])
        total=filter(None, t) 
     
        size=[]
        for i in total:
             size.append(len(i))
             
        mail_count=counts.to_dict()
        mail_count={k: v for k, v in mail_count.items() if k}

        tail_num=[]
        for i in total:
            tt=[]
            for j in i:
                t=mail_count.get(j)
                tt.append(t)
            tail_num.append(tt)
        my={'maillist':total,'size':tail_num,'num':size}
        mymail=pd.DataFrame(my) 
        
        pop_com=[]
        for cell in range(len(tail_num)):
            m=tail_num[cell]
            r=max(m)
            t=m.index(r)
            n=[total[cell][t]]
            x=size[cell]
            y=n*x
            for comcell in y:
                pop_com.append(comcell)
        comcom=[]
        for cell_i in total:
             for cell_j in cell_i:
                 comcom.append(cell_j)

        md={'original_com':comcom,'clean_com':pop_com}
        myDF=pd.DataFrame(md) 
        return myDF

#data import
df=pd.read_csv("Test.csv")
df.dtypes
df.info
mydf=df.filter(items=['Bear Unique ID','Company_Name (Organization Name)','Email'])
mydf=mydf.rename(columns={'Bear Unique ID':'id','Company_Name (Organization Name)':'Company','Email':'email'})

#data cleaning 
mydf['clean_com']=cleancom(mydf['Company'])
mydf['clean_email']=cleanemail(mydf['email'])
mydf['email_tail']=emailtail(mydf['email'])
mydf['test']=unitest(mydf['email'])

#data subset
tail_unique=mydf[mydf.test == 'unique']
tail_generic=mydf[mydf.test == 'generic']
email=tail_unique['email_tail']

company=mydf['clean_com']

#entity resolution output
threshold=0.8
myDF3=entity(company)

threshold=0.7
myDF4=entity(email)
myDF4=myDF4.rename(columns = {'clean_com':'sim_email','original_com':'email_tail'})

#merge file
#same email-same company
ddaa0=tail_unique.filter(items=['clean_com','email_tail'])
ddaa0=ddaa0[ddaa0['email_tail'].map(len) >0]
ddaa0=ddaa0[ddaa0['clean_com'].map(len) >0]
ddaa0=ddaa0.dropna()

bb = ddaa0.groupby(['email_tail','clean_com']).count()
b=bb.add_suffix('_Count').reset_index()
c=DataFrame({'count' : ddaa0.groupby(['email_tail','clean_com']).size()}).reset_index()

idx = c.groupby('email_tail')['count'].idxmax()
yy=c.loc[idx, ['email_tail', 'clean_com']]
yy=yy.rename(columns = {'clean_com':'sim_com'})
result1=pd.merge(mydf,yy,how='left',on='email_tail')
result1.sim_com.fillna(result1.clean_com, inplace=True)

#similar email
df4=myDF4.drop_duplicates(['email_tail'])   #4689
dup4=myDF4[myDF4.duplicated()]
result2=pd.merge(result1,df4,how='left',on='email_tail')
ddaa2=result2.filter(items=['sim_email','sim_com'])
ddaa2=ddaa2.drop_duplicates(['sim_email'])
ddaa2=ddaa2.rename(columns = {'sim_com':'sim_com2'})
ddaa2=ddaa2.dropna()
ddaa2=ddaa2[ddaa2['sim_com2'].map(len) >0]
result3=pd.merge(result2,ddaa2,how='left',on='sim_email')
result3.sim_com2.fillna(result3.sim_com, inplace=True)

#similar company
df3=myDF3.drop_duplicates(['original_com'])   
dup=myDF3[myDF3.duplicated()]
df3=df3.rename(columns = {'original_com':'clean_com','clean_com':'sim_com3'})
result4=pd.merge(result3,df3,how='left',on='clean_com')
result4.sim_com3.fillna(result4.sim_com2, inplace=True)

result4['Final_Com']=result4['sim_com3'].apply(lambda x:x.title())

result4.to_csv('Test_out.csv')

ddaa4=result4.filter(items=['Final_Com'])
ddaa4=ddaa4.drop_duplicates(['Final_Com'])
len(ddaa4)
#final unique com:5789
#unique com:7360



        
         
         
