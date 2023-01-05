# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 09:56:41 2022

@author: Dell
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import re
pip install -U textblob
from textblob import TextBlob
import nltk
from wordcloud import WordCloud
nltk.download('punkt')
nltk.download('stopwords') 
from nltk.corpus import stopwords

df=pd.read_csv("Product_details.csv")
df.isnull().value_counts()

X=df["Product_Description"]
X.shape

Y=df["Sentiment"]


df["Product_Description"]=df.Product_Description.map(lambda x : x.lower())
df["Product_Description"]
PD= ' '.join(df["Product_Description"])

my_stop_words = stopwords.words('english')

no_stop_tokens = [word for word in PD if not word in my_stop_words]

no_stop_tokens = [word for word in PD if not word in my_stop_words]

#=== Stemming ======================================================
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
for i in df["Product_Description"].index:
    df["Product_Description"].iloc[i] = stemmer.stem(df["Product_Description"].iloc[i])

df["Product_Description"]

#=== Lemmatizer ========================================================
from nltk.stem import WordNetLemmatizer
Lemm = WordNetLemmatizer()
for x in df["Product_Description"].index:
    df["Product_Description"].iloc[x] = Lemm.lemmatize(df["Product_Description"].iloc[x])

df["Product_Description"]


def TweetCleaning(tweets):
 cleantweet=re.sub(r"@[a-zA-Z0-9]+"," ",tweets)
 cleantweet=re.sub(r"#[a-zA-Z0-9]+"," ",cleantweet)
 cleantweet=''.join(word for word in cleantweet.split() if word not in no_stop_tokens)
 return cleantweet
df['Product_Description']=df['Product_Description'].apply(TweetCleaning)

#=== TOKENIZATION ======================================================
from sklearn.feature_extraction.text import CountVectorizer
Vectorizer = CountVectorizer()
Vt = Vectorizer.fit_transform(df["Product_Description"])
Vt.toarray()

#TFIDF===========================================================
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer().fit(Vt)
X_vect = transformer.transform(Vt)  
X_vect.toarray()

#====================================================================
from sklearn.model_selection import train_test_split
X_train, X_test,Y_train,Y_test = train_test_split(X_vect,Y,test_size=0.2)

# naive baye
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB(alpha=3)
nb.fit(X_train,Y_train)
Y_predtrain = nb.predict(X_train)
Y_predtest=nb.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_train,Y_predtrain)
cm

from sklearn.metrics import accuracy_score
AC1=accuracy_score(Y_train,Y_predtrain)
AC1

AC2=accuracy_score(Y_test,Y_predtest)
AC2


##################

#RANDOM FOREST..

from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier(n_estimators=400,max_depth=10)
RF.fit(X_train,Y_train)
Y_predtrain=RF.predict(X_train)
Y_predtest=RF.predict(X_test)

from sklearn.metrics import accuracy_score
AC1=accuracy_score(Y_train,Y_predtrain)
AC1

AC2=accuracy_score(Y_test,Y_predtest)
AC2


######Decision tree.........
from sklearn.tree import DecisionTreeClassifier
DT=DecisionTreeClassifier(max_depth=7)
DT.fit(X_train,Y_train)
Y_predtrain=RF.predict(X_train)
Y_predtest=RF.predict(X_test)

from sklearn.metrics import accuracy_score
AC1=accuracy_score(Y_train,Y_predtrain)
AC1

AC2=accuracy_score(Y_test,Y_predtest)
AC2



#boosting
from sklearn.ensemble import AdaBoostClassifier
AD=AdaBoostClassifier(base_estimator=DT,n_estimators=400,learning_rate=0.1)
AD.fit(X_train,Y_train)
Y_predtrain=RF.predict(X_train)
Y_predtest=RF.predict(X_test)

from sklearn.metrics import accuracy_score
AC1=accuracy_score(Y_train,Y_predtrain)
AC1

AC2=accuracy_score(Y_test,Y_predtest)
AC2

###bagging
from sklearn.ensemble import BaggingClassifier
BG=BaggingClassifier(n_estimators=100)
BG.fit(X_train,Y_train)
Y_predtrain=RF.predict(X_train)
Y_predtest=RF.predict(X_test)

from sklearn.metrics import accuracy_score
AC1=accuracy_score(Y_train,Y_predtrain)
AC1

AC2=accuracy_score(Y_test,Y_predtest)
AC2
