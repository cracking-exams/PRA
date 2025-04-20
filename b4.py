# Create a program that fits a mixture of Gaussians to a dataset of handwritten digit features and clusters them into distinct groups. Use the Expectation-Maximization method to estimate the parameters of the Gaussian mixture model.


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn import metrics

mnist = fetch_openml('mnist_784',version=1)
x = mnist.data.astype('float32')
y= mnist.target.astype('int')

x= x/255.0

pca = PCA(n_components=50,random_state=42)
x_pca = pca.fit_transform(x)

model = GaussianMixture(n_components=10,covariance_type='full',random_state=42)
model.fit(x,y)

clusters = model.predict(x)

# Step 7: Define function to plot digits from a cluster
def plot_cluster_images(cluster_number, num_samples=10):
    indices = np.where(clusters == cluster_number)[0][:num_samples]
    fig, axes = plt.subplots(1, num_samples, figsize=(10, 2))
    for i, idx in enumerate(indices):
        axes[i].imshow(x[idx].reshape(28, 28), cmap="gray")
        axes[i].axis("off")
    plt.suptitle(f"Cluster {cluster_number}")
    plt.show()

# Step 8: Show samples from first 5 clusters
for i in range(5):
    plot_cluster_images(i)






print("Reducing to 2D using PCA for visualization...")
pca_vis = PCA(n_components=2, random_state=42)
X_vis = pca_vis.fit_transform(x_pca)

# Step 6: Plot the clusters in 2D
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=clusters, cmap='tab10', s=1)
plt.title("MNIST clustered using GMM (visualized with 2D PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(scatter, label='Cluster Label')
plt.show()

