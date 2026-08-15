import numpy as np
import torch

torch.manual_seed(42)

class KMD:
    def __init__(self, data, rank=None, lamb=0.,
            backend='numpy',
            device='cpu',
            verbose=False
        ):
        """
        Base class for Koopman Mode Decomposition.

        Parameters:
        -----------
        data : array-like
            Input data to decompose
        backend : str, optional
            Computational backend to use ('numpy', 'pytorch', or 'cupy')
        device : str, optional
            Device to use for computation when using PyTorch backend
        verbose : bool, optional
            Whether to print verbose output
        send_to_cpu : bool, optional
            Whether to send results to CPU after computation
        """
        self.data = data
        self.backend = backend
        self.device = device
        self.rank = rank
        self.lamb = lamb
        self.verbose = verbose
        self.A_v, self.E, self.S, self.U, self.V, self.Vh, self.W, self.W_prime = None, None, None, None, None, None, None, None

    def init_data(self):
        if isinstance(self.data, np.ndarray):
            self.data = torch.from_numpy(self.data).to(self.device)

        if self.data.ndim == 2:
            self.data = self.data.unsqueeze(0)  # Add trial dimension (1, timesteps, features)
        elif self.data.ndim == 3:
            pass  # Already in the correct format (trials, timesteps, features)
        else:
            raise ValueError(f"Invalid data shape: {self.data.shape}. Expected 2D (samples, features) or 3D (trials, samples, features)")

        self.n_trials, self.n_timesteps, self.n_features = self.data.shape

    def embed(self):

        raise NotImplementedError


    def compute_svd(self):
        """
        Compute the Singular Value Decomposition of the embedded data.

        Parameters:
        -----------
        rank : int, optional
            Truncation rank for SVD. If None, full SVD is computed.

        Returns:
        --------
        self : object
            Returns self for method chaining
        """
        # Flatten embedding across trials if 3D
        E = self.E.reshape(self.E.shape[0] * self.E.shape[1], self.E.shape[2]) if self.E.ndim == 3 else self.E

        U, self.S, self.Vh = torch.linalg.svd(E.T, full_matrices=False)
        self.V = self.Vh.T


    def reduced_rank(self):

        raise NotImplementedError


    def compute_dmd(self):
        """
        Compute the Dynamic Mode Decomposition.
        """
        if self.verbose:
            print("Computing DMD")

        if self.lamb != 0:
            regularization = self.lamb * torch.eye(self.rank).to(self.device)
        else:
            regularization = torch.zeros(self.rank, self.rank).to(self.device)

        self.A_v = (torch.linalg.inv(self.W.T @ self.W + regularization) @ self.W.T @ self.W_prime).T


    def fit(self):
        self.init_data()
        self.embed()
        self.compute_svd()
        if self.rank is not None:
            self.reduced_rank()
        self.compute_dmd()


class KoopSTD(KMD):
    def __init__(self, data, rank=15, lamb=0., win_len=8, hop_size=1,
            backend='numpy',
            device='cpu',
            verbose=False
        ):
        super().__init__(data, rank, lamb, backend, device, verbose)
        self.win_len = win_len
        self.hop_size = hop_size
        self.rank = rank
        self.lamb = lamb
        self.data = data

        self.backend = backend
        self.device = device
        self.verbose = verbose

    def embed(self):
        # multivariate STFT
        stfts = []
        for i in range(self.n_features):
            stft = torch.stft(self.data[:, :, i], n_fft=self.win_len, hop_length=self.hop_size, return_complex=True, normalized=True)
            stfts.append(stft)
        stfts = torch.stack(stfts, dim=1)
        trial, _, _, time_frames = stfts.shape
        stfts = stfts.view(trial, time_frames, -1).real.to(torch.float32)

        self.E = stfts.to(self.device)
        if self.n_trials == 1:
            self.E = self.E.squeeze(0)

    def compute_svd(self):
        if self.E.ndim == 3:
            E = self.E.reshape(self.E.shape[0] * self.E.shape[1], self.E.shape[2])
        else:
            E = self.E

        # z = E.T = U Σ V^T. Residual ranking uses this SVD of z, not the
        # closed-form DMD operator YX^+. torch.linalg.svd returns Vh = V^T.
        self.U, self.S, self.Vh = torch.linalg.svd(E.T, full_matrices=False)
        self.V = self.Vh.T

        if E.shape[0] < E.shape[1]:  # T < N
            E = E[:, :E.shape[0]]
        self.E_minus = E[:-1]
        self.E_plus = E[1:]

    def _gram_matrices(self):
        X_X = torch.matmul(self.E_minus.T.conj(), self.E_minus)
        X_Y = torch.matmul(self.E_minus.T.conj(), self.E_plus)
        Y_Y = torch.matmul(self.E_plus.T.conj(), self.E_plus)
        return X_X, X_Y, Y_Y

    def _select_modes_by_residual(self):
        """
        Rank selection via ||Y - λ X||_v at z's own SVD pairs (σ_j, column j of V^T).

        Returns indices of the `rank` modes with the smallest residuals.
        """
        X_X, X_Y, Y_Y = self._gram_matrices()
        n_modes = self.S.shape[0]
        # Mode j corresponds to column j of V^T (Vh), not a row of V^T / column of V.
        eigenvectors = self.Vh[:, :n_modes]
        eigenvalues = self.S[:n_modes].to(dtype=X_X.dtype)

        YYv = Y_Y @ eigenvectors
        XYv = X_Y @ eigenvectors
        XYHv = X_Y.T.conj() @ eigenvectors
        XXv = X_X @ eigenvectors
        v_conj = eigenvectors.conj()
        yy = (v_conj * YYv).sum(dim=0)
        xy = (v_conj * XYv).sum(dim=0)
        yx = (v_conj * XYHv).sum(dim=0)
        xx = (v_conj * XXv).sum(dim=0)

        numerator = yy - eigenvalues * xy - eigenvalues.conj() * yx + (eigenvalues.abs() ** 2) * xx
        residuals = torch.sqrt(torch.abs(numerator) / torch.abs(xx))
        k = min(self.rank, residuals.numel())
        return torch.topk(residuals, k, largest=False).indices

    def _build_rank_projections(self, topk_indices):
        # Time-mode matrix V (columns of V = rows of V^T), then keep residual-selected modes.
        V = self.V
        rank = topk_indices.numel()

        if self.n_trials > 1:
            V = V.reshape(self.E.shape)
            V_rank = V[:, :, topk_indices]
            new_shape = (self.E.shape[0] * (self.E.shape[1] - 1), rank)
            V_minus_rank = V_rank[:, :-1].reshape(new_shape)
            V_plus_rank = V_rank[:, 1:].reshape(new_shape)
        else:
            V_rank = V[:, topk_indices]
            V_minus_rank = V_rank[:-1]
            V_plus_rank = V_rank[1:]

        return V_minus_rank, V_plus_rank

    def reduced_rank(self):
        topk_indices = self._select_modes_by_residual()
        self.W, self.W_prime = self._build_rank_projections(topk_indices)
        self.rank = self.W.shape[1]

    def residual_dmd(self):
        """
        Residual-ranked DMD using z's SVD pairs rather than the YX^+ closed form.
        For large pairwise comparisons, prefer fit() / reduced_rank().
        """
        self.reduced_rank()
        self.compute_dmd()

    def compute_residuals(self, X_X, X_Y, Y_Y, eigenvalue, eigenvector):
        numerator = torch.matmul(
            eigenvector.conj(),
            torch.matmul(
                Y_Y - eigenvalue * X_Y - torch.conj(eigenvalue) * X_Y.T.conj() + (eigenvalue.abs() ** 2) * X_X,
                eigenvector
            )
        )
        denominator = torch.matmul(eigenvector.conj(), torch.matmul(X_X, eigenvector))

        residual = torch.sqrt(torch.abs(numerator) / torch.abs(denominator))
        return residual
