import sys
from koopstd.kmd import *
from koopstd.geometric import *
from koopstd.datasets import *

import torch
import numpy as np
from tqdm import tqdm

disable_tqdm = not sys.stdout.isatty()


class BaseSimilarityMetric:
    def __init__(self, X, Y=None):
        """
        Base class for similarity metrics between dynamical systems.

        Parameters:
        -----------
        X : list or numpy.ndarray or torch.Tensor
            First dataset or list of datasets for comparison
        Y : list or numpy.ndarray or torch.Tensor, optional
            Second dataset or list of datasets for comparison
        """
        self.mode = self._determine_comparison_mode(X, Y)
        self.data = self._prepare_data(X, Y)
        self.device = 'cpu'

    def _determine_comparison_mode(self, X, Y):
        """Determine the comparison mode based on input data types."""
        if Y is None:
            if isinstance(X, list):
                return 'self-pairwise'
            else:
                raise ValueError("For single sample input, Y must be provided")
        else:
            if isinstance(X, list) and isinstance(Y, list):
                return 'bipartite-pairwise'
            else:
                return 'one-to-one'

    def _prepare_data(self, X, Y):
        """Prepare data based on input format."""
        data = []

        # Convert numpy arrays to torch tensors if needed
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if Y is not None and isinstance(Y, np.ndarray):
            Y = torch.from_numpy(Y).float()

        # Handle different input formats
        if self.mode == 'self-pairwise':
            data.append(X)
        elif self.mode == 'one-to-one':
            data.append([X])
            data.append([Y])
        elif self.mode == 'bipartite-pairwise':
            data.append(X)
            data.append(Y)

        return data


    def calculate_dist_matrix(self, representations1, representations2=None):
        """
        Calculate distance matrix between all representations.

        Parameters:
        -----------
        representations1 : list
            First list of representations
        representations2 : list, optional
            Second list of representations (if None, uses representations1)

        Returns:
        --------
        numpy.ndarray
            Distance matrix
        """
        if self.mode == 'one-to-one':
            return self.compute_distance(representations1[0], representations2[0])

        # For other modes, calculate full distance matrix
        if self.mode == 'self-pairwise' or representations2 is None:
            n, m = len(representations1), len(representations1)
            representations2 = representations1
        else:
            n, m = len(representations1), len(representations2)

        dist_matrix = np.zeros((n, m))

        for i, d1 in tqdm(enumerate(representations1), total=n, desc="Computing Distance Matrix", dynamic_ncols=False, disable=disable_tqdm):
            for j, d2 in enumerate(representations2):
                if self.mode == 'self-pairwise' and j >= i:
                    # Only compute lower triangle for self-pairwise
                    if i == j:
                        continue
                    dist_matrix[i, j] = dist_matrix[j, i]
                else:
                    dist_matrix[i, j] = self.compute_distance(d1, d2)

                    # Mirror for self-pairwise
                    if self.mode == 'self-pairwise':
                        dist_matrix[j, i] = dist_matrix[i, j]

        return dist_matrix

    def compute_distance(self, representation1, representation2):

        raise NotImplementedError("Subclasses must implement compute_distance")


class KoopOpMetric(BaseSimilarityMetric):
    def __init__(self, X, Y=None, kmd_method='koopstd', kmd_params=None, dist='wasserstein', dist_params=None, device='cuda'):
        """
        Initialize a Koopman Operator-based metric for comparing dynamical systems.

        Parameters:
        -----------
        X : list or numpy.ndarray or torch.Tensor
            First dataset or list of datasets for comparison
        Y : list or numpy.ndarray or torch.Tensor, optional
            Second dataset or list of datasets for comparison
        kmd_method : str, default='koopstd'
            Type of Koopman Mode Decomposition to use
        kmd_params : dict, optional
            Parameters for the KMD algorithm
        dist : str, default='wasserstein'
            Type of distance metric to use
        dist_params : dict, optional
            Parameters for the distance metric
        device : str, default='cuda'
            Device to use for computations
        """
        super().__init__(X, Y)
        self.kmd_params = kmd_params or {}
        self.dist_params = dist_params or {}
        self.kmd_params['device'], self.dist_params['device'] = device, device

        # Initialize KMD models
        self.kmds = []
        for dataset in self.data:
            if kmd_method == 'koopstd':
                self.kmds.append([KoopSTD(d, **self.kmd_params) for d in dataset])
            elif kmd_method == 'havok':
                raise NotImplementedError("Please refer to https://github.com/mitchellostrow/DSA for official HAVOK-based DSA implementation")
            else:
                raise ValueError(f"Unknown KMD: {kmd_method}")

        if dist == 'wasserstein':
            self.dist = WassersteinDistance(**self.dist_params)  # {'p': 1, 'method': 'emd'}
        elif dist == 'procrustes':
            self.dist = ProcrustesDistance(**self.dist_params)  # {'score_method': 'euclidean', 'group': 'O(n)', 'iters': 1000, 'lr': 0.01}
        else:
            raise ValueError(f"Unknown distance metric: {dist}")

    def compute_distance(self, kmd1, kmd2):
        """
        Compute distance between two KMD models.

        Parameters:
        -----------
        kmd1 : KoopSTD
            First KMD model
        kmd2 : KoopSTD
            Second KMD model

        Returns:
        --------
        float
            Distance between representations
        """
        return self.dist.compute(kmd1.A_v, kmd2.A_v)

    def fit_score(self):
        """
        Fit KMD models and calculate distance matrix.

        Returns:
        --------
        numpy.ndarray
            Distance matrix between datasets
        """
        # Fit all KMD models
        for kmd_set in self.kmds:
            for kmd in tqdm(kmd_set, desc="Fitting KMD models", disable=disable_tqdm):
                kmd.fit()

        if len(self.kmds) > 1:
            return self.calculate_dist_matrix(self.kmds[0], self.kmds[1])
        else:
            return self.calculate_dist_matrix(self.kmds[0])
