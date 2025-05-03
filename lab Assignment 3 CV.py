import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sklearn.cluster import KMeans  
import random
import time

class KMeansClustering:
    
    def __init__(self, k=5, max_iterations=100, tol=1e-4):
        self.k = k
        self.max_iterations = max_iterations
        self.tol = tol
        self.centroids = None
        self.labels = None
        
    def initialize_centroids(self, X):
        n_samples = X.shape[0]

        indices = np.random.choice(n_samples, self.k, replace=False)
        return X[indices]
    
    def assign_clusters(self, X, centroids):
        distances = np.zeros((X.shape[0], self.k))
        for i in range(self.k):
            # Compute squared Euclidean distance
            distances[:, i] = np.sum((X - centroids[i])**2, axis=1)
        return np.argmin(distances, axis=1)
    
    def update_centroids(self, X, labels):
        new_centroids = np.zeros((self.k, X.shape[1]))
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                new_centroids[i] = self.centroids[i]
        return new_centroids
    
    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        for _ in range(self.max_iterations):
            self.labels = self.assign_clusters(X, self.centroids)
            new_centroids = self.update_centroids(X, self.labels)
            centroid_diff = np.linalg.norm(new_centroids - self.centroids)
            if centroid_diff < self.tol:
                break
                
            self.centroids = new_centroids
        
        return self
    
    def predict(self, X):
        return self.assign_clusters(X, self.centroids)

def load_and_preprocess_image(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)
    

    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    

    height, width, channels = img_array.shape
    pixels = img_array.reshape(-1, channels)
    
    return img, img_array, pixels

def extract_number_from_clusters(img_array, labels, k):
    height, width, _ = img_array.shape
    
    cluster_masks = []
    for i in range(k):
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[labels.reshape(height, width) == i] = 255
        cluster_masks.append(mask)
    
    return cluster_masks

def display_results(original_img, cluster_masks, k):
    original_array = np.array(original_img)
    
    n_rows = 2 
    n_cols = max(k // 2 + 1, 2)  # +1 for the original image
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(original_array)
    plt.title('Original Image')
    plt.axis('off')
    
    for i, mask in enumerate(cluster_masks):
        plt.subplot(n_rows, n_cols, i + 2)
        plt.imshow(mask, cmap='gray')
        plt.title(f'Cluster {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()



def apply_kmeans_to_ishihara(image_path, k=5):
    original_img, img_array, pixels = load_and_preprocess_image(image_path)
 
    kmeans = KMeansClustering(k=k)
    kmeans.fit(pixels)
    labels = kmeans.labels
    cluster_masks = extract_number_from_clusters(img_array, labels, k)
    
    return original_img, cluster_masks

def main():

    image_path = "12.jpg"  # image file path
    k = 5  # Number of clusters
    
    try:
        original_img, cluster_masks = apply_kmeans_to_ishihara(image_path, k)
        display_results(original_img, cluster_masks, k)
        

        _, img_array, pixels = load_and_preprocess_image(image_path)
        
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()