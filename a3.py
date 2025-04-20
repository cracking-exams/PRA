# Design a statistical model to analyze wine quality using Gaussian distribution methods. Utilize synthetic data generated with NumPy or the Wine Quality Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import fetch_openml

wine = fetch_openml('wine-quality-red',version=1,as_frame=True)
df = wine.frame
df.head(4)

df['class']=df['class'].astype('int64')
df.describe()

df.hist(bins=20,figsize=(10,8))
plt.title('feature Distribution')
plt.show()


plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()


X = df.drop(columns="class")
y = (df["class"] >= 6).astype(int) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)


gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))