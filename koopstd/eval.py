import numpy as np
from sklearn.metrics import silhouette_score

class MetricEvaluator:
    def __init__(self, distance_matrix, cluster_sizes):
        """
        Initialize the ClusteringEvaluator with the distance matrix and cluster sizes.

        Args:
            distance_matrix (np.ndarray): A square distance matrix of shape (n_samples, n_samples).
            cluster_sizes (list): A list where each element indicates the number of samples in a cluster.
        """
        self.distance_matrix = distance_matrix
        self.cluster_sizes = cluster_sizes
        self.labels = self._generate_labels(cluster_sizes)

    def _generate_labels(self, cluster_sizes):
        """
        Generate cluster labels based on the cluster sizes.

        Args:
            cluster_sizes (list): List of sample counts per cluster.

        Returns:
            np.ndarray: Array of cluster labels for all samples.
        """
        labels = []
        for cluster_id, size in enumerate(cluster_sizes):
            labels.extend([cluster_id] * size)
        return np.array(labels)

    def compute_silhouette_score(self):
        """
        Compute the Silhouette Score for the clustering.

        Returns:
            float: Silhouette Score.
        """
        return silhouette_score(self.distance_matrix, self.labels, metric="precomputed")

    def compute_dunn_index(self):
        """
        Compute the Dunn Index for the clustering.

        Returns:
            float: Dunn Index.
        """
        unique_labels = np.unique(self.labels)
        if len(unique_labels) < 2:
            raise ValueError("At least two clusters are required for Dunn Index.")

        # Calculate intra-cluster distances
        intra_distances = []
        for label in unique_labels:
            cluster_indices = np.where(self.labels == label)[0]
            cluster_distances = self.distance_matrix[np.ix_(cluster_indices, cluster_indices)]
            intra_distances.append(np.max(cluster_distances))

        # Calculate inter-cluster distances
        inter_distances = []
        for i, label1 in enumerate(unique_labels):
            for label2 in unique_labels[i+1:]:
                indices1 = np.where(self.labels == label1)[0]
                indices2 = np.where(self.labels == label2)[0]
                inter_cluster_distances = self.distance_matrix[np.ix_(indices1, indices2)]
                inter_distances.append(np.min(inter_cluster_distances))

        # Compute Dunn Index
        # dunn = np.min(inter_distances) / np.max(intra_distances)
        dunn = np.mean(inter_distances) / np.mean(intra_distances)
        return dunn

    def evaluate(self):
        """
        Evaluate the clustering using both Silhouette Score and Dunn Index.

        Returns:
            dict: A dictionary containing the Silhouette Score and Dunn Index.
        """
        silhouette = self.compute_silhouette_score()
        return silhouette
        # score = self.compute_dunn_index()
        # return score

# Example Usage
if __name__ == "__main__":
    # Example distance matrix
    distance_matrix = np.array([
        [0.0, 1.0, 1.2, 5.0, 6.0, 6.5, 10.0, 11.0, 11.5],
        [1.0, 0.0, 0.8, 5.5, 6.2, 6.8, 10.5, 11.5, 12.0],
        [1.2, 0.8, 0.0, 5.8, 6.5, 7.0, 10.8, 11.8, 12.5],
        [5.0, 5.5, 5.8, 0.0, 1.0, 1.2, 6.5, 7.0, 7.5],
        [6.0, 6.2, 6.5, 1.0, 0.0, 0.5, 6.8, 7.5, 8.0],
        [6.5, 6.8, 7.0, 1.2, 0.5, 0.0, 7.0, 7.8, 8.5],
        [10.0, 10.5, 10.8, 6.5, 6.8, 7.0, 0.0, 1.0, 1.5],
        [11.0, 11.5, 11.8, 7.0, 7.5, 7.8, 1.0, 0.0, 0.5],
        [11.5, 12.0, 12.5, 7.5, 8.0, 8.5, 1.5, 0.5, 0.0]
    ])

    # Cluster sizes: 3 samples in each of 3 clusters
    cluster_sizes = [3, 3, 3]

    # Initialize and evaluate
    evaluator = MetricEvaluator(distance_matrix, cluster_sizes)
    results = evaluator.evaluate()

    print("Evaluation Results:")
    print(results)
