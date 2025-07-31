import torch
import numpy as np
import ot
from typing import Literal, Optional, Union, List, Tuple
import torch.nn.utils.parametrize as parametrize
import torch.optim as optim
import torch.nn as nn

def pad_zeros(A,B,device):

    with torch.no_grad():
        dim = max(A.shape[0],B.shape[0])
        A1 = torch.zeros((dim,dim)).float()
        A1[:A.shape[0],:A.shape[1]] += A
        A = A1.float().to(device)

        B1 = torch.zeros((dim,dim)).float()
        B1[:B.shape[0],:B.shape[1]] += B
        B = B1.float().to(device)

    return A,B

class LearnableSimilarityTransform(torch.nn.Module):
    """
    Computes the similarity transform for a learnable orthonormal matrix C
    """
    def __init__(self, n,orthog=True):
        """
        Parameters
        __________
        n : int
            dimension of the C matrix
        """
        super(LearnableSimilarityTransform, self).__init__()
        #initialize orthogonal matrix as identity
        self.C = torch.nn.Parameter(torch.eye(n).float())
        self.orthog = orthog

    def forward(self, B):
        if self.orthog:
            return self.C @ B @ self.C.transpose(-1, -2)
        else:
            return self.C @ B @ torch.linalg.inv(self.C)

class Skew(torch.nn.Module):
    def __init__(self,n,device):
        """
        Computes a skew-symmetric matrix X from some parameters (also called X)

        """
        super().__init__()

        self.L1 = torch.nn.Linear(n,n,bias = False, device = device)
        self.L2 = torch.nn.Linear(n,n,bias = False, device = device)
        self.L3 = torch.nn.Linear(n,n,bias = False, device = device)

    def forward(self, X):
        X = torch.tanh(self.L1(X))
        X = torch.tanh(self.L2(X))
        X = self.L3(X)
        return X - X.transpose(-1, -2)

class Matrix(torch.nn.Module):
    def __init__(self,n,device):
        """
        Computes a matrix X from some parameters (also called X)

        """
        super().__init__()

        self.L1 = torch.nn.Linear(n,n,bias = False, device = device)
        self.L2 = torch.nn.Linear(n,n,bias = False, device = device)
        self.L3 = torch.nn.Linear(n,n,bias = False, device = device)

    def forward(self, X):
        X = torch.tanh(self.L1(X))
        X = torch.tanh(self.L2(X))
        X = self.L3(X)
        return X

class CayleyMap(torch.nn.Module):
    """
    Maps a skew-symmetric matrix to an orthogonal matrix in O(n)
    """
    def __init__(self, n, device):
        """
        Parameters
        __________

        n : int
            dimension of the matrix we want to map

        device : {'cpu','cuda'} or int
            hardware device on which to send the matrix
        """
        super().__init__()
        self.register_buffer("Id", torch.eye(n,device = device))

    def forward(self, X):
        # (I + X)(I - X)^{-1}
        return torch.linalg.solve(self.Id + X, self.Id - X)


class ProcrustesDistance:
    """
    Computes the Procrustes Analysis over Vector Fields
    """
    def __init__(self,
                iters=200,
                score_method: Literal["angular", "euclidean"] = "angular",
                lr=0.01,
                device: Literal["cpu", "cuda"] = 'cuda',
                verbose=False,
                group: Literal["O(n)", "SO(n)", "GL(n)"] = "O(n)"
                ):
        """
        Parameters
        _________
        iters : int
            number of iterations to perform gradient descent

        score_method : {"angular", "euclidean"}
            specifies the type of metric to use

        lr : float
            learning rate

        device : {'cpu', 'cuda'} or int
            hardware device on which to send the matrix

        verbose : bool
            prints when finished optimizing

        group : {'SO(n)', 'O(n)', 'GL(n)'}
            specifies the group of matrices to optimize over
        """
        self.iters = iters
        self.score_method = score_method
        self.lr = lr
        self.verbose = verbose
        self.device = device
        self.C_star = None
        self.A = None
        self.B = None
        self.group = group

    def fit(self,
            A,
            B,
            iters=None,
            lr=None,
            group=None
            ):
        """
        Computes the optimal matrix C over specified group

        Parameters
        __________
        A : np.array or torch.tensor
            first data matrix
        B : np.array or torch.tensor
            second data matrix
        iters : int or None
            number of optimization steps, if None then resorts to saved self.iters
        lr : float or None
            learning rate, if None then resorts to saved self.lr
        group : {'SO(n)', 'O(n)', 'GL(n)'}
            specifies the group of matrices to optimize over

        Returns
        _______
        None
        """
        assert A.shape[0] == A.shape[1]
        assert B.shape[0] == B.shape[1]

        A = A.to(self.device)
        B = B.to(self.device)
        self.A, self.B = A, B
        lr = self.lr if lr is None else lr
        iters = self.iters if iters is None else iters
        group = self.group if group is None else group

        if group in {"SO(n)", "O(n)"}:
            self.losses, self.C_star, self.sim_net = self.optimize_C(A,
                                                                     B,
                                                                     lr, iters,
                                                                     orthog=True,
                                                                     verbose=self.verbose)
        if group == "O(n)":
            # permute the first row and column of B then rerun the optimization
            P = torch.eye(B.shape[0], device=self.device)
            if P.shape[0] > 1:
                P[[0, 1], :] = P[[1, 0], :]
            losses, C_star, sim_net = self.optimize_C(A,
                                                    P @ B @ P.T,
                                                    lr, iters,
                                                    orthog=True,
                                                    verbose=self.verbose)
            if losses[-1] < self.losses[-1]:
                self.losses = losses
                self.C_star = C_star @ P
                self.sim_net = sim_net
        if group == "GL(n)":
            self.losses, self.C_star, self.sim_net = self.optimize_C(A,
                                                                B,
                                                                lr, iters,
                                                                orthog=False,
                                                                verbose=self.verbose)

    def optimize_C(self, A, B, lr, iters, orthog, verbose):
        # parameterize mapping to be orthogonal
        n = A.shape[0]
        sim_net = LearnableSimilarityTransform(n, orthog=orthog).to(self.device)
        if orthog:
            parametrize.register_parametrization(sim_net, "C", Skew(n, self.device))
            parametrize.register_parametrization(sim_net, "C", CayleyMap(n, self.device))
        else:
            parametrize.register_parametrization(sim_net, "C", Matrix(n, self.device))

        simdist_loss = nn.MSELoss(reduction='sum')

        optimizer = optim.Adam(sim_net.parameters(), lr=lr)
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

        losses = []
        A /= torch.linalg.norm(A)
        B /= torch.linalg.norm(B)
        for _ in range(iters):
            # Zero the gradients of the optimizer.
            optimizer.zero_grad()
            # Compute the Frobenius norm between A and the product.
            loss = simdist_loss(A, sim_net(B))

            loss.backward()

            optimizer.step()
            # if _ % 99:
            #     scheduler.step()
            losses.append(loss.item())

        if verbose:
            print("Finished optimizing C")

        C_star = sim_net.C.detach()
        return losses, C_star, sim_net

    def score(self, A=None, B=None, score_method=None, group=None):
        """
        Given an optimal C already computed, calculate the metric

        Parameters
        __________
        A : np.array or torch.tensor or None
            first data matrix, if None defaults to the saved matrix in fit
        B : np.array or torch.tensor or None
            second data matrix if None, defaults to the saved matrix in fit
        score_method : None or {'angular', 'euclidean'}
            overwrites the score method in the object for this application
        group : {'SO(n)', 'O(n)', 'GL(n)'} or None
            specifies the group of matrices to optimize over

        Returns
        _______
        score : float
            similarity of the data under the similarity transform w.r.t C
        """
        assert self.C_star is not None
        A = self.A if A is None else A
        B = self.B if B is None else B
        assert A is not None
        assert B is not None
        assert A.shape == self.C_star.shape
        assert B.shape == self.C_star.shape
        score_method = self.score_method if score_method is None else score_method
        group = self.group if group is None else group
        with torch.no_grad():
            if not isinstance(A, torch.Tensor):
                A = torch.from_numpy(A).float().to(self.device)
            if not isinstance(B, torch.Tensor):
                B = torch.from_numpy(B).float().to(self.device)
            C = self.C_star.to(self.device)

        if group in {"SO(n)", "O(n)"}:
            Cinv = C.T
        elif group in {"GL(n)"}:
            Cinv = torch.linalg.inv(C)
        else:
            raise AssertionError("Need proper group name")

        if score_method == 'angular':
            num = torch.trace(A.T @ C @ B @ Cinv)
            den = torch.norm(A, p='fro') * torch.norm(B, p='fro')
            score = torch.arccos(num/den).cpu().numpy()
            if np.isnan(score):  # around -1 and 1, we sometimes get NaNs due to arccos
                if num/den < 0:
                    score = np.pi
                else:
                    score = 0
        else:
            score = torch.norm(A - C @ B @ Cinv, p='fro').cpu().numpy().item()

        return score

    def compute(self, A, B):
        """
        For efficiency, computes the optimal matrix and returns the score

        Parameters
        __________
        A : np.array or torch.tensor
            first data matrix
        B : np.array or torch.tensor
            second data matrix
        iters : int or None
            number of optimization steps, if None then resorts to saved self.iters
        lr : float or None
            learning rate, if None then resorts to saved self.lr
        score_method : {'angular', 'euclidean'} or None
            overwrites parameter in the class
        group : {'SO(n)', 'O(n)', 'GL(n)'} or None
            specifies the group of matrices to optimize over

        Returns
        _______
        score : float
            similarity of the data under the similarity transform w.r.t C
        """

        if isinstance(A, np.ndarray):
            A = torch.from_numpy(A).float()
        if isinstance(B, np.ndarray):
            B = torch.from_numpy(B).float()

        assert A.shape[0] == A.shape[1]
        assert B.shape[0] == B.shape[1]
        assert A.shape[0] == B.shape[0], "Matrices must be the same size"

        self.fit(A, B, iters=self.iters, lr=self.lr, group=self.group)
        score_star = self.score(self.A, self.B, score_method=self.score_method, group=self.group)

        return score_star


class WassersteinDistance:
    """
    A class for computing Wasserstein distances between distributions using optimal transport.

    This class implements methods for computing Wasserstein distances between
    distributions represented as samples or histograms.
    """

    def __init__(self,
                 p: int = 2,
                 method: Literal['emd', 'sinkhorn'] = 'emd',
                 reg: float = 0.01,
                 device: str = 'cuda',
                 feature_type: Literal['sv', 'eig'] = 'eig'):
        """
        Initialize the WassersteinDistance object.

        Parameters:
        -----------
        p : int, default=2
            Order of the Wasserstein distance (1 for W1, 2 for W2, etc.)
        method : str, default='emd'
            Method to compute the distance: 'emd' (exact) or 'sinkhorn' (regularized)
        reg : float, default=0.01
            Regularization parameter for Sinkhorn algorithm (only used if method='sinkhorn')
        device : str, default='cpu'
            Device to use for computations ('cpu' or 'cuda')
        """
        self.p = p
        self.method = method
        self.reg = reg
        self.device = device
        self.feature_type = feature_type
        # Validate parameters
        if method not in ['emd', 'sinkhorn']:
            raise ValueError("Method must be one of 'emd' or 'sinkhorn'")
        if p < 1:
            raise ValueError("Order p must be at least 1")

    def compute_from_distributions(self,
                X: Union[np.ndarray, torch.Tensor],
                Y: Union[np.ndarray, torch.Tensor],
                a: Optional[Union[np.ndarray, torch.Tensor]] = None,
                b: Optional[Union[np.ndarray, torch.Tensor]] = None) -> float:
        """
        Compute the Wasserstein distance between two distributions.

        Parameters:
        -----------
        X : numpy.ndarray or torch.Tensor
            First distribution samples (shape: [n_samples, n_features])
        Y : numpy.ndarray or torch.Tensor
            Second distribution samples (shape: [m_samples, n_features])
        a : numpy.ndarray or torch.Tensor, optional
            Weights for first distribution (if None, uniform weights are used)
        b : numpy.ndarray or torch.Tensor, optional
            Weights for second distribution (if None, uniform weights are used)

        Returns:
        --------
        distance : float
            Wasserstein distance between the distributions
        """
        # Convert numpy arrays to torch tensors if needed
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float().to(self.device)
        if isinstance(Y, np.ndarray):
            Y = torch.from_numpy(Y).float().to(self.device)

        # Ensure X and Y are 2D
        if X.dim() == 1:
            X = X.view(-1, 1)
        if Y.dim() == 1:
            Y = Y.view(-1, 1)

        # Create uniform weights if not provided
        if a is None:
            a = torch.ones(X.shape[0], device=self.device) / X.shape[0]
        elif isinstance(a, np.ndarray):
            a = torch.from_numpy(a).float().to(self.device)

        if b is None:
            b = torch.ones(Y.shape[0], device=self.device) / Y.shape[0]
        elif isinstance(b, np.ndarray):
            b = torch.from_numpy(b).float().to(self.device)

        # Compute cost matrix (p-th power of Euclidean distance)
        M = ot.dist(X, Y)
        if self.p != 1:
            M = M ** (self.p / 2)  # Since ot.dist returns squared Euclidean distance

        # Compute Wasserstein distance
        if self.method == 'emd':
            distance = ot.emd2(a, b, M)
        else:  # sinkhorn, supports gradient descent
            distance = ot.sinkhorn2(a, b, M, self.reg)

        # Take p-th root for Wp distance
        if self.p != 1:
            distance = distance ** (1.0 / self.p)

        return distance.item()


    def compute(self, X_features: Union[np.ndarray, torch.Tensor],
                      Y_features: Optional[Union[np.ndarray, torch.Tensor]] = None,
                      feature_type: Literal['sv', 'eig'] = None) -> float:
        """
        Compute Wasserstein distance between matrices based on their features (singular values or eigenvalues).

        Parameters:
        -----------
        X_features : numpy.ndarray or torch.Tensor
            First matrix
        Y_features : numpy.ndarray or torch.Tensor, optional
            Second matrix (if None, uses X_features)
        feature_type : str, default='sv'
            Type of features to extract: 'sv' for singular values, 'eig' for eigenvalues

        Returns:
        --------
        distance : float
            Wasserstein distance between the feature distributions
        """
        if Y_features is None:
            raise ValueError("Y_features must be provided")

        # Convert numpy arrays to torch tensors if needed
        if isinstance(X_features, np.ndarray):
            X_features = torch.from_numpy(X_features).float().to(self.device)
        if isinstance(Y_features, np.ndarray):
            Y_features = torch.from_numpy(Y_features).float().to(self.device)
        feature_type = self.feature_type if feature_type is None else feature_type
        # Extract features
        if feature_type == "sv":
            a = torch.svd(X_features).S.view(-1, 1)
            b = torch.svd(Y_features).S.view(-1, 1)
        elif feature_type == "eig":
            a = torch.linalg.eig(X_features).eigenvalues
            a = torch.vstack([a.real, a.imag]).T
            b = torch.linalg.eig(Y_features).eigenvalues
            b = torch.vstack([b.real, b.imag]).T
        else:
            raise ValueError(f"Unknown feature type: {feature_type}. Use 'sv' or 'eig'")

        # Compute Wasserstein distance
        return self.compute_from_distributions(a, b)
