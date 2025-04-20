# Build a Python application to classify iris flowers using the Nearest Neighbor Rule. Use a given dataset with features such as petal length and width. Experiment with different values of K and evaluate the model's accuracy

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

iris = load_iris()

x=iris.data
y=iris.target #setosa, versicolor , virginica   

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

k_values=range(1,10)
accuracies = []
for k in range(1,10):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)
    acc = metrics.accuracy_score(y_test,y_pred)
    accuracies.append(acc)
    print(f'Accuracy for k={k} is {acc*100}')


plt.figure(figsize=(10,12))
plt.plot(k_values,accuracies,marker='o',linestyle='-')
plt.xticks(k_values)
plt.xlabel("k")
plt.ylabel('Accuracy')
plt.show()