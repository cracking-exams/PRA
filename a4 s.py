# Develop a classification system for handwritten digit recognition using the MNIST dataset, leveraging Bayes' Decision Theory to optimize decision-making and minimize classification error.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import seaborn as sns

mnist = fetch_openml('mnist_784',version=1)
x = mnist.data.astype('float32')
y = mnist.target.astype('int')

x=x/255.0

pca = PCA(n_components=50,random_state=42)
x_pca = pca.fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(
    x_pca, y, test_size=0.2, random_state=42
)
model = GaussianNB()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
acc=metrics.accuracy_score(y_test,y_pred)
print(acc)


plt.figure(figsize=(10,13))
sns.heatmap(metrics.confusion_matrix(y_test,y_pred),annot=True,cmap='Blues',xticklabels=range(10),yticklabels=range(10))
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show()


