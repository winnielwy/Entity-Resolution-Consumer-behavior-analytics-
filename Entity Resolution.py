# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:53:10 2015

@author: wenyingliu
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from pandas import DataFrame, Series
#import profile
import line_profiler 
import memory_profiler
import gc

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
    col8=col7.apply(lambda x:x.replace('no data',''))
    col9=col8.str.strip()
    return col9
    
def cleanemail(col):
    col1=col.astype('string')
    col2=col1.fillna("")
    col3=col2.str.lower()
    col4=col3.str.strip()
    col5=col4.fillna("")
    col6=col5.apply(lambda x:x.replace('no data',''))
    col7=col6.str.strip()
    return col7
    
def emailtail(col):
    col1=col.astype('string')
    col2=col1.fillna("")
    col3=col2.str.lower()
    col4=col3.str.strip()
    col5=col4.fillna("")
    col6=col5.apply(lambda x:x.split('@', 1)[-1])
    col7=col6.apply(lambda x:x.replace('no data',''))
    col8=col7.str.strip()
    return col8
    
def emailfront(col):
    col1=col.astype('string')
    col2=col1.fillna("")
    col3=col2.str.lower()
    col4=col3.str.strip()
    col5=col4.fillna("")
    col6=col5.apply(lambda x:x.split('@', 1)[0])
    col7=col6.apply(lambda x:x.replace('no data',''))
    col8=col7.str.strip()
    return col8

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
    
##fuzzy matching
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

def fuzzymatching(target,threshold):
        cell = []     
        for t in target:
            if t not in cell:
                cell.append(t)
                
        com=sorted(cell)
        
        comcom=[]
        similarity=[]    
        for i in range(len(com)):
            second=[]
            third=[]
            if i+10<=len(com):
                for j in range(i,i+10):
                    t=fuzz.ratio(com[i],com[j])
                    if t>threshold:
                        second.append(com[j])
                        third.append(t)             
            else:
                for j in range(i,len(com)):
                    t=fuzz.ratio(com[i],com[j])
                    if t>threshold:
                        second.append(com[j])
                        third.append(t)
            comcom.append(second)
            similarity.append(third)
        
    
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
                        newrow =list(set(t[row]+ comcom[i]))
                        t[row] = newrow
            if found_flag==False:
                t.append(comcom[i])
        total=filter(None, t) 
     
        size=[]
        for i in total:
             size.append(len(i))
        
        counts=target.value_counts()     
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
####################
#data import
df=pd.read_csv("Sample Data.csv")
#df.dtypes
#df.info
mydf=df.filter(items=['01_Bear_ID','04_First_Name','05_Last_Name','06_Email','07_Company_Name'])
mydf=mydf.rename(columns={'01_Bear_ID':'id','04_First_Name':'first_name','05_Last_Name':'last_name','06_Email':'email','07_Company_Name':'Company'})

#data cleaning
mydf['First']=cleanemail(mydf['first_name'])
mydf['Last']=cleanemail(mydf['last_name']) 
#mydf['full_name']=mydf.First.map(str) + "_" + mydf.Last
mydf['clean_com']=cleancom(mydf['Company'])
mydf['clean_email']=cleanemail(mydf['email'])
mydf['email_front']=emailfront(mydf['email'])
mydf['email_tail']=emailtail(mydf['email'])
mydf['test']=unitest(mydf['email'])


#data subset
####delete NA
def exists(it):
    return (it is not None)
   
tail_unique=mydf[mydf.test == 'unique']
tail_generic=mydf[mydf.test == 'generic']
emailtail=tail_unique['email_tail']
emailtail=emailtail[emailtail.str.len() != 0]
emailtail=emailtail.dropna()

emailfront=mydf['email_front']
emailfront=emailfront[emailfront.str.len() != 0]
emailfront=emailfront.dropna()

company=mydf['clean_com']
company=company[company.str.len() != 0]
company=company.dropna()

firstname=mydf['First']
firstname=firstname[firstname.str.len() != 0]
firstname=firstname.dropna()

lastname=mydf['Last']
lastname=lastname[lastname.str.len() != 0]
lastname=lastname.dropna()


#######output
myDF_com=fuzzymatching(company,86)
myDF_com=myDF_com.dropna()
myDF_com=myDF_com[myDF_com['original_com'].map(len) >0]
##counts test
counts_com=company.value_counts()
len(counts_com)
counts_coms=myDF_com.original_com.value_counts()
len(counts_coms)


myDF_tail=fuzzymatching(emailtail,86)
myDF_tail=myDF_tail.rename(columns = {'clean_com':'sim_tail','original_com':'email_tail'})
myDF_tail=myDF_tail.dropna()
myDF_tail=myDF_tail[myDF_tail['email_tail'].map(len) >0]
##counts test
counts_tail=emailtail.value_counts()
len(counts_tail)

####merge file
#company
#similar company
df_com=myDF_com.drop_duplicates(['original_com'])   
dup_com=myDF_com[myDF_com.duplicated()]
df_com=df_com.rename(columns = {'original_com':'clean_com','clean_com':'sim_com'})
result1=pd.merge(mydf,df_com,how='left',on='clean_com')
result1.sim_com.fillna(result1.clean_com, inplace=True)

#email- company
df_mail=myDF_tail.drop_duplicates(['email_tail'])   #4689
dup_mail=myDF_tail[myDF_tail.duplicated()]
result2=pd.merge(result1,df_mail,how='left',on='email_tail')

ddaa0=tail_unique.filter(items=['clean_com','email_tail'])
ddaa0=ddaa0[ddaa0['email_tail'].map(len) >0]
ddaa0=ddaa0[ddaa0['clean_com'].map(len) >0]
ddaa0=ddaa0.dropna()
c=DataFrame({'count' : ddaa0.groupby(['email_tail','clean_com']).size()}).reset_index()
idx = c.groupby('email_tail')['count'].idxmax()
yy=c.loc[idx, ['email_tail', 'clean_com']]
yy=yy.rename(columns = {'email_tail':'sim_tail','clean_com':'sim_com2'})
result3=pd.merge(result2,yy,how='left',on='sim_tail')
result3.sim_com2.fillna(result3.sim_com, inplace=True)

#same email-same company
ddaa1=result3.filter(items=['sim_com2','email_tail','test'])
ddaa1=ddaa1[ddaa1.test == 'unique']

ddaa1=ddaa1[ddaa1['email_tail'].map(len) >0]
ddaa1=ddaa1[ddaa1['sim_com2'].map(len) >0]
ddaa1=ddaa1.dropna()

c1=DataFrame({'count' : ddaa1.groupby(['email_tail','sim_com2']).size()}).reset_index()

idx1 = c1.groupby('email_tail')['count'].idxmax()
yy1=c1.loc[idx, ['email_tail', 'sim_com2']]
yy1=yy1.rename(columns = {'sim_com2':'sim_com3'})
result4=pd.merge(result3,yy1,how='left',on='email_tail')
result4.sim_com3.fillna(result3.sim_com2, inplace=True)

###clean_com-sim_com3
ddaa2=result4.filter(items=['clean_com','sim_com3'])
ddaa2=ddaa2[ddaa2['clean_com'].map(len) >0]
ddaa2=ddaa2[ddaa2['sim_com3'].map(len) >0]
ddaa2=ddaa2.dropna()

df = ddaa2.groupby(['clean_com','sim_com3'])['sim_com3'].agg({'no':'count'})
mask = df.groupby(level=0).agg('idxmax')
df_count = df.loc[mask['no']]
df_count = df_count.reset_index()
df_count=df_count.rename(columns = {'sim_com3':'sim_com4'})

result5=pd.merge(result4,df_count,how='left',on='clean_com')
result5.sim_com4.fillna(result4.sim_com3, inplace=True)
del result5['no']
result5['Final_Com']=result5['sim_com4'].apply(lambda x:x.title())
#result5.to_csv('Test3.csv')

ddaa4=result5.filter(items=['Final_Com'])
ddaa4=ddaa4.drop_duplicates(['Final_Com'])
len(ddaa4)


#######individual
def similarity(com):
        '''
        com = []     
        for t in target:
            if t not in com:
                com.append(t) 
                
        #com=sorted(cell)
        '''
        simrate=[]   
        for i in range(len(com)):
            if i+1<len(com):
                t=fuzz.ratio(com[i],com[i+1])
                simrate.append(t) 
            else:
                t=0
                simrate.append(t)        
        my={'sim_com4':com,'simrate':simrate}
        sim_rate=pd.DataFrame(my) 
        return sim_rate
        
result5=result5.sort_index(by=['sim_com4', 'Last'], ascending=[True, True])
result5 = result5.reset_index(drop=True)
#simcom
simcom=result5['sim_com4']
simcom=simcom.tolist()
myDF_simcom=similarity(simcom)
myDF_simcom=myDF_simcom.rename(columns = {'simrate':'simrate_com'})

#simail
result5['sim_mail']=result5.email_front.map(str) + " " + result5.email_tail
simail=result5['sim_mail']
simail=simail.tolist()

myDF_simail=similarity(simail)
myDF_simail=myDF_simail.rename(columns = {'sim_com4':'sim_mail','simrate':'simrate_mail'})
#myDF_simail=myDF_simail.dropna()
#myDF_simail=myDF_simail[myDF_simail['sim_mail'].map(len) >0]

#simemail_front
emailfront=result5['email_front']
emailfront=emailfront.tolist()
myDF_front=similarity(emailfront)
myDF_front=myDF_front.rename(columns = {'sim_com4':'email_front','simrate':'simrate_front'})
myDF_front=myDF_front.dropna()
myDF_front=myDF_front[myDF_front['email_front'].map(len) >0]

#simfirst
firstname=result5['First']
firstname=firstname.tolist()
myDF_first=similarity(firstname)
myDF_first=myDF_first.rename(columns = {'sim_com4':'First','simrate':'simrate_first'})
myDF_first=myDF_first.dropna()
myDF_first=myDF_first[myDF_first['First'].map(len) >0]

#simlast
lastname=result5['Last']
lastname=lastname.tolist()
myDF_last=similarity(lastname)
myDF_last=myDF_last.rename(columns = {'sim_com4':'Last','simrate':'simrate_Last'})
myDF_last=myDF_last.dropna()
myDF_last=myDF_last[myDF_last['Last'].map(len) >0]

##mergefile
result=pd.concat([myDF_simcom,myDF_simail,myDF_front,myDF_first,myDF_last],axis=1,ignore_index=False)
result=result.filter(items=['simrate_com','simrate_mail','simrate_front','simrate_first','simrate_Last'])
final=pd.concat([result5,result],axis=1,ignore_index=False)
final.columns=list(result5.columns.values)+list(result.columns.values)

final['id']=final['id'].astype('str')
final['sim_individual']=final.id.shift(-1)
final['sim_individual']=final['sim_individual'].astype('str')
final['iidd']=final.id+' '+final.sim_individual

#final['sim_individual']='BA-' + final['sim_individual']

#final.to_csv('winnie.csv')


####
#individual fuzzy matching
foo = final.ix[(final['simrate_com']>=95) 
& (final['simrate_mail'] >= 94) 
& (final['simrate_front'] >= 70) 
& (final['simrate_first']>= 85) 
& (final['simrate_Last'] >= 90)]
        
#foo.to_csv('innie.csv')
#foo['s_id']=foo.index

id=foo.id.tolist()
iidd=foo.iidd.tolist()

ID=[]
for t in iidd:
    ID.append(t.split())
    
 
t=[]
newrow=[]
lst=[]
col=0
for i in range(len(id)):
    found_flag=False
    for row,lst in enumerate(t):
        for col,sim_company in enumerate(lst):
            if id[i]==sim_company:
                found_flag=True
                newrow =list(set(t[row]+ ID[i]))
                t[row] = newrow
    if found_flag==False:
        t.append(ID[i])
total=filter(None, t)

size=[]
for i in total:
     size.append(len(i))

my={'sim-id':total,'num':size}
my=pd.DataFrame(my) 

pop_com=[]
for cell in range(len(total)):
    n=[total[cell][0]]
    x=size[cell]
    y=n*x
    for comcell in y:
        pop_com.append(comcell)

comcom=[]
for cell_i in total:
     for cell_j in cell_i:
         comcom.append(cell_j)
 
md={'original_id':comcom,'sim_id':pop_com}
myDF=pd.DataFrame(md) 
     
df=myDF.drop_duplicates(['original_id'])  
dup=myDF[myDF.duplicated()]
df=df.rename(columns = {'original_id':'id'})
final_ID=pd.merge(final,df,how='left',on='id')
dup_id=final_ID[final_ID.duplicated('id')] 
final_ID.sim_id.fillna(final_ID.id, inplace=True)
final_ID['sim_id']='BA-' + final_ID['sim_id']      
   
final_ID.to_csv('winnie.csv')    

        

        
        
        
