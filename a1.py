# Design and implement pattern recognition system to identify and extract unique species patterns from the Iris dataset

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.decomposition import PCA

iris = sns.load_dataset('iris')
iris.head()

sns.pairplot(iris,hue='species')
plt.show


X=iris.drop(columns='species')
y=iris['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_y_pred = knn.predict(X_test)
knn_accuracy = metrics.accuracy_score(y_test, knn_y_pred)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_y_pred = dt.predict(X_test)
dt_accuracy = metrics.accuracy_score(y_test, dt_y_pred)

clf = RandomForestClassifier(n_estimators=100 ,random_state=42)
clf.fit(X_train,y_train)
clf_y_pred = clf.predict(X_test)
clf_accuracy = metrics.accuracy_score(y_test,clf_y_pred)

print("Accuracy Comparison:")
print("RandomForest Classifier Accuracy:", clf_accuracy)
print("KNN Classifier Accuracy:", knn_accuracy)
print("DT Classifier Accuracy:", dt_accuracy)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=iris, x="petal_length", y="petal_width", hue="species")
plt.title("Sepals visualiztion")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()

pca = PCA(n_components=2)
x_pca = pca.fit_transform(X)

plt.figure(figsize=(8,6))
sns.scatterplot(x=x_pca[:,0],y=x_pca[:,1],hue=y)
plt.title('PCA Visualization')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()

