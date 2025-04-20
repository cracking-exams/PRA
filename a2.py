# Develop a text classification model that can effectively identify, extract features, and classify documents from the 20 Newsgroups dataset into one of the 20 predefined categories using  pattern recognition techniques.


import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
import re

news = fetch_20newsgroups(subset='all',shuffle=True)

x=news.data
y=news.target

def clean_text(text):
    re.sub('<.*?>','',text)
    text.lower
    return text

x_train,x_test,y_train,y_test=train_test_split(news.data,news.target,test_size=0.3)

model = Pipeline([
    ('tfidf',TfidfVectorizer(preprocessor=clean_text,stop_words='english')),
    ('clf',MultinomialNB())  
])

model.fit(x_train,y_train)

y_pred = model.predict(x_test)
acc=metrics.accuracy_score(y_test,y_pred)
print(acc)

print(metrics.classification_report(y_test, y_pred, target_names=news.target_names))

import seaborn as sns
plt.figure(figsize=(10,14))
sns.heatmap(metrics.confusion_matrix(y_test,y_pred),annot=True,fmt='d',cmap='Blues',xticklabels=news.target_names,yticklabels=news.target_names)
plt.show()