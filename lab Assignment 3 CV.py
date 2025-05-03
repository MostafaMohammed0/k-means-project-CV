import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sklearn.cluster import KMeans  # For comparison only
import random
import time

class KMeansClustering:
    """
    Implementation of K-means clustering algorithm from scratch.
    """
    def __init__(self, k=5, max_iterations=100, tol=1e-4):
        """
        Initialize the K-means clustering algorithm.
        
        Args:
            k (int): Number of clusters.
            max_iterations (int): Maximum number of iterations.
            tol (float): Tolerance for convergence.
        """
        self.k = k
        self.max_iterations = max_iterations
        self.tol = tol
        self.centroids = None
        self.labels = None
        
    def initialize_centroids(self, X):
        """
        Initialize centroids randomly from the data points.
        
        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features).
            
        Returns:
            numpy.ndarray: Initial centroids.
        """
        n_samples = X.shape[0]
        # Generate random indices without replacement
        indices = np.random.choice(n_samples, self.k, replace=False)
        return X[indices]
    
    def assign_clusters(self, X, centroids):
        """
        Assign data points to the closest centroid.
        
        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features).
            centroids (numpy.ndarray): Current centroids.
            
        Returns:
            numpy.ndarray: Cluster assignments for each data point.
        """
        # Calculate Euclidean distance between each data point and each centroid
        distances = np.zeros((X.shape[0], self.k))
        for i in range(self.k):
            # Compute squared Euclidean distance
            distances[:, i] = np.sum((X - centroids[i])**2, axis=1)
        
        # Assign each data point to the closest centroid
        return np.argmin(distances, axis=1)
    
    def update_centroids(self, X, labels):
        """
        Update centroids based on cluster assignments.
        
        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features).
            labels (numpy.ndarray): Cluster assignments for each data point.
            
        Returns:
            numpy.ndarray: Updated centroids.
        """
        new_centroids = np.zeros((self.k, X.shape[1]))
        for i in range(self.k):
            # Get all points assigned to cluster i
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                # Calculate the mean to get the new centroid
                new_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                # If no points are assigned to this cluster, keep the old centroid
                new_centroids[i] = self.centroids[i]
        return new_centroids
    
    def fit(self, X):
        """
        Fit the K-means model to the data.
        
        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features).
        """
        # Initialize centroids
        self.centroids = self.initialize_centroids(X)
        
        # Perform K-means clustering
        for _ in range(self.max_iterations):
            # Assign data points to clusters
            self.labels = self.assign_clusters(X, self.centroids)
            
            # Update the centroids
            new_centroids = self.update_centroids(X, self.labels)
            
            # Check for convergence
            centroid_diff = np.linalg.norm(new_centroids - self.centroids)
            if centroid_diff < self.tol:
                # If centroids haven't moved significantly, stop
                break
                
            self.centroids = new_centroids
        
        return self
    
    def predict(self, X):
        """
        Predict the closest cluster for each sample in X.
        
        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features).
            
        Returns:
            numpy.ndarray: Predicted cluster indices.
        """
        return self.assign_clusters(X, self.centroids)

def load_and_preprocess_image(image_path):
    """
    Load and preprocess an image.
    
    Args:
        image_path (str): Path to the image.
        
    Returns:
        tuple: Original image and preprocessed data.
    """
    # Load the image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Check if the image is grayscale
    if len(img_array.shape) == 2:
        # Convert grayscale to RGB
        img_array = np.stack((img_array,) * 3, axis=-1)
    
    # Reshape the image to a 2D array of pixels
    height, width, channels = img_array.shape
    pixels = img_array.reshape(-1, channels)
    
    return img, img_array, pixels

def extract_number_from_clusters(img_array, labels, k):
    """
    Extract the number from the clustered Ishihara test image.
    
    Args:
        img_array (numpy.ndarray): Original image as a numpy array.
        labels (numpy.ndarray): Cluster assignments for each pixel.
        k (int): Number of clusters.
        
    Returns:
        list: List of images, one for each cluster.
    """
    height, width, _ = img_array.shape
    
    # Create a mask for each cluster
    cluster_masks = []
    for i in range(k):
        mask = np.zeros((height, width), dtype=np.uint8)
        # Set pixels belonging to cluster i to 255 (white)
        mask[labels.reshape(height, width) == i] = 255
        cluster_masks.append(mask)
    
    return cluster_masks

def display_results(original_img, cluster_masks, k):
    """
    Display the original image and the segmented clusters.
    
    Args:
        original_img (PIL.Image.Image): Original image.
        cluster_masks (list): List of binary masks, one for each cluster.
        k (int): Number of clusters.
    """
    # Convert PIL image to numpy array
    original_array = np.array(original_img)
    
    # Calculate number of rows and columns for subplots
    n_rows = 2  # Original + some clusters in row 1, rest in row 2
    n_cols = max(k // 2 + 1, 2)  # +1 for the original image
    
    plt.figure(figsize=(15, 10))
    
    # Display original image
    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(original_array)
    plt.title('Original Image')
    plt.axis('off')
    
    # Display each cluster
    for i, mask in enumerate(cluster_masks):
        plt.subplot(n_rows, n_cols, i + 2)
        plt.imshow(mask, cmap='gray')
        plt.title(f'Cluster {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def compare_with_sklearn(pixels, k, random_state=42):
    """
    Compare our K-means implementation with scikit-learn's implementation.
    
    Args:
        pixels (numpy.ndarray): Pixel data.
        k (int): Number of clusters.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        tuple: Execution times and labels for both implementations.
    """
    # Our implementation
    start_time = time.time()
    kmeans_custom = KMeansClustering(k=k)
    kmeans_custom.fit(pixels)
    custom_time = time.time() - start_time
    custom_labels = kmeans_custom.labels
    
    # scikit-learn implementation
    start_time = time.time()
    kmeans_sklearn = KMeans(n_clusters=k, random_state=random_state)
    sklearn_labels = kmeans_sklearn.fit_predict(pixels)
    sklearn_time = time.time() - start_time
    
    print(f"Custom implementation time: {custom_time:.4f} seconds")
    print(f"scikit-learn implementation time: {sklearn_time:.4f} seconds")
    
    return custom_time, sklearn_time, custom_labels, sklearn_labels

def apply_kmeans_to_ishihara(image_path, k=5):
    """
    Apply K-means clustering to an Ishihara test image and extract the number.
    
    Args:
        image_path (str): Path to the Ishihara test image.
        k (int): Number of clusters.
        
    Returns:
        tuple: Original image, cluster masks.
    """
    # Load and preprocess the image
    original_img, img_array, pixels = load_and_preprocess_image(image_path)
    
    # Apply K-means clustering
    kmeans = KMeansClustering(k=k)
    kmeans.fit(pixels)
    labels = kmeans.labels
    
    # Extract the number from clusters
    cluster_masks = extract_number_from_clusters(img_array, labels, k)
    
    return original_img, cluster_masks

def main():
    """
    Main function to demonstrate K-means clustering on Ishihara test images.
    """
    # Example usage
    image_path = "42.jpg"  # Replace with actual path
    k = 12  # Number of clusters
    
    try:
        original_img, cluster_masks = apply_kmeans_to_ishihara(image_path, k)
        display_results(original_img, cluster_masks, k)
        
        # Optional: Compare with scikit-learn implementation
        _, img_array, pixels = load_and_preprocess_image(image_path)
        compare_with_sklearn(pixels, k)
        
        print("\nInstructions for identifying the number:")
        print("1. Look at each cluster image.")
        print("2. The number should be visible in one of the clusters.")
        print("3. You may need to adjust the number of clusters (k) for optimal results.")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please provide a valid path to an Ishihara test image.")

if __name__ == "__main__":
    main()