import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import cv2

# Load and preprocess the image
def load_image(file_path):
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# Reshape the image for k-means
def reshape_image(img):
    return img.reshape((-1, 3))

# Perform k-means clustering
def perform_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    return kmeans

# Create 3D scatter plot
def plot_3d_scatter(data, labels, title, cluster_centers=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize the data to 0-1 range
    data_normalized = data / 255.0
    
    if cluster_centers is not None:
        # If cluster centers are provided, use them for coloring
        colors = cluster_centers[labels] / 255.0
    else:
        colors = data_normalized
    
    scatter = ax.scatter(data_normalized[:, 0], data_normalized[:, 1], data_normalized[:, 2], 
                         c=colors, cmap='viridis')
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_title(title)
    plt.colorbar(scatter)
    plt.show()

def plot_3d_scatter_res(data, labels, title, cluster_centers=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define a unique color for each cluster
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('viridis', len(unique_labels))(labels / len(unique_labels))
    
    # Plot the data points with cluster-based colors
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], 
                         c=colors, cmap='viridis', marker='o', s=30)
    
    # Optionally, plot the cluster centers if provided
    if cluster_centers is not None:
        ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2],
                   c='red', marker='X', s=100, label='Cluster Centers')
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.set_title(title)
    plt.legend()
    plt.show()


# Reconstruct the segmented image
def reconstruct_image(labels, centers, shape):
    return centers[labels].reshape(shape)