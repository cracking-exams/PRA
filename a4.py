# Develop a classification system for handwritten digit recognition using the MNIST dataset, leveraging Bayes' Decision Theory to optimize decision-making and minimize classification error.


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA

# Dataset Loading
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(np.int8)

# Normalize pixel values
X = X / 255.0  

# Reduce dimensionality using PCA
pca = PCA(n_components=50)  # You can try 30–100 and tune this
X_pca = pca.fit_transform(X)

# Model Development (Using GaussianNB)
model = GaussianNB()

# Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print("\nEvaluation Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Visualization - Correct vs Misclassified
correct = np.where(y_pred == y_test)[0]
incorrect = np.where(y_pred != y_test)[0]

plt.figure(figsize=(12, 5))
for i, idx in enumerate(correct[:5]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"True={y_test[idx]}\nPred={y_pred[idx]}")
    plt.axis('off')

for i, idx in enumerate(incorrect[:5]):
    plt.subplot(2, 5, i + 6)
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"True={y_test[idx]}\nPred={y_pred[idx]}")
    plt.axis('off')

plt.suptitle("Correctly vs Misclassified Digits")
plt.tight_layout()
plt.show()
