import numpy as np
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm

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
        self.A_v, self.E, self.S, self.V, self.Vh, self.W, self.W_prime = None, None, None, None, None, None, None

        # TODO: Backends specification
        # if backend == 'numpy':
        #     self.xp = np
        # elif backend == 'cupy':
        #     self.xp = cp
        # elif backend == 'pytorch':
        #     self.xp = torch
        # else:
        #     raise ValueError(f"Unsupported backend: {backend}. Choose from 'numpy', 'pytorch', or 'cupy'")

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

        _, self.S, self.V = torch.linalg.svd(E.T, full_matrices=False)

        if E.shape[0] < E.shape[1]:  # T < N
            E = E[:, :E.shape[0]]
        self.E_minus = E[:-1]
        self.E_plus = E[1:]

    def reduced_rank(self):
        M = torch.matmul(self.E_minus.T.conj(), self.E_minus)
        N = torch.matmul(self.E_minus.T.conj(), self.E_plus)
        O = torch.matmul(self.E_plus.T.conj(), self.E_plus)

        egvalues, egvectors = self.S, self.V
        residuals = []
        for j, (eigenvalue, eigenvector) in enumerate(zip(egvalues, egvectors.T)):
            residual = self.compute_residuals(M, N, O, eigenvalue, eigenvector)
            residuals.append(residual)
        residuals = torch.tensor(residuals)
        topk_indices = torch.topk(-residuals, self.rank, largest=False).indices
        V = self.V.T

        if self.n_trials > 1:
            V = V.reshape(self.E.shape)
            V_rank = V[:, :, topk_indices]
            new_shape = (self.E.shape[0] * (self.E.shape[1] - 1), self.rank)
            V_minus_rank = V_rank[:, :-1].reshape(new_shape)
            V_plus_rank = V_rank[:, 1:].reshape(new_shape)
        else:
            V_rank = V[:, topk_indices]
            V_minus_rank = V_rank[:-1]
            V_plus_rank = V_rank[1:]

        self.W = V_minus_rank
        self.W_prime = V_plus_rank

    def residual_dmd(self):
        """
        Standard implementation of ResDMD, however, for the sake of efficiency,
        we don't recommend it in large dataset comparison.
        """
        self.Vt_minus = self.V[:-1]
        self.Vt_plus = self.V[1:]

        X_X = torch.matmul(self.Vt_plus.T.conj(), self.Vt_plus)
        X_Y = torch.matmul(self.Vt_plus.T.conj(), self.Vt_minus)
        Y_Y = torch.matmul(self.Vt_minus.T.conj(), self.Vt_minus)

        A_full = torch.linalg.inv(self.Vt_minus.T @ self.Vt_minus) @ self.Vt_minus.T @ self.Vt_plus
        _, egvalues, egvectors = torch.linalg.svd(A_full, full_matrices=True)
        residuals = []
        for j, (eigenvalue, eigenvector) in enumerate(zip(egvalues, egvectors.T)):
            residual = self.compute_residual(X_X, X_Y, Y_Y, eigenvalue, eigenvector)
            residuals.append(residual)
        residuals = torch.tensor(residuals)
        topk_indices = torch.topk(-residuals, self.rank, largest=False).indices
        self.A_v = egvalues[topk_indices].view(-1,1)  # direct eigenvalues

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

class TrainResKoopNet:
    def __init__(self, data, rank=15, lamb=0.01, learning_rate=0.001, epochs=10, device='cuda'):
        self.rank = rank
        self.lamb = lamb
        self.data = data
        self.device = device
        win_len = min([x.shape[0] for x in data])
        self.hs_dataset = self.create_hs_dataset(win_len=win_len)
        self.epochs = epochs
        self.reskoopnet = nn.Sequential(
            nn.Linear(self.data[0].shape[-1], rank),
            # nn.Sigmoid(),
            # nn.Linear(512, rank),
            nn.Sigmoid()
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.reskoopnet.parameters(), lr=learning_rate)

    def create_hs_dataset(self, win_len=1000):
        hs_dataset = []
        for i in range(len(self.data)):
            T, _ = self.data[i].shape
            if T < win_len:
                raise ValueError(f"Time dimension {T} is smaller than window size {win_len}.")
            for start in range(T - win_len + 1):
                window = self.data[i][start:start+win_len]
                hs_dataset.append(window)
        hs_dataset = torch.stack(hs_dataset, dim=0)
        return hs_dataset

    def forward(self, x):
        return self.reskoopnet(x)

    def residual_loss(self, batch_data):
        X, Y = batch_data[:, :-1].to(self.device), batch_data[:, 1:].to(self.device)
        psi_x = self.forward(X).reshape(-1, self.rank)
        psi_y = self.forward(Y).reshape(-1, self.rank)
        psi_xT = psi_x.T
        G = torch.matmul(psi_xT, psi_x) + self.lamb * torch.eye(psi_xT.shape[0], device=self.device)
        A = torch.matmul(psi_xT, psi_y)
        try:
            K = torch.linalg.solve(G, A)
        except Exception as e:
            print(e)
            K = torch.linalg.pinv(G) @ A
        _, S, Vh = torch.linalg.svd(K, full_matrices=False)
        S_diag = torch.diag(S)
        psi_x_v = torch.matmul(psi_x, Vh)
        psi_x_v_k = torch.matmul(psi_x_v, S_diag)
        psi_y_v = torch.matmul(psi_y, Vh)
        J = torch.norm(psi_y_v - psi_x_v_k, p='fro')
        return J

    def fit(self, saved_path):
        dataset = TensorDataset(self.hs_dataset)
        dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
        self.reskoopnet.train()
        min_loss = float('inf')
        for epoch in tqdm(range(self.epochs), desc="Training Epochs", leave=False):
            epoch_losses = []
            for i, batch_data in enumerate(dataloader):
                self.optimizer.zero_grad()
                loss = self.residual_loss(batch_data[0])
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            if avg_epoch_loss < min_loss:
                min_loss = avg_epoch_loss
                torch.save(self.reskoopnet.state_dict(), f'{saved_path}/reskoopnet.pt')
            tqdm.write(f"Epoch {epoch+1} of {self.epochs} | avg_loss: {avg_epoch_loss:.4f} | min_loss: {min_loss:.4f}")

class ResidualDMD:
    def __init__(self, data, saved_path, rank=15, lamb=0.01, device='cuda'):
        self.rank = rank
        self.lamb = lamb
        self.device = device
        self.saved_path = saved_path
        self.data = data
        self.reskoopnet = nn.Sequential(
            nn.Linear(self.data[0].shape[-1], rank),
            # nn.Sigmoid(),
            # nn.Linear(512, rank),
            nn.Sigmoid()
        ).to(self.device)
        state_dict = torch.load(f'{self.saved_path}/reskoopnet.pt', map_location=device)
        self.reskoopnet.load_state_dict(state_dict)
        self.A_vs = []

    def forward(self, x):
        return self.reskoopnet(x)

    def approximate_koopman_op(self):
        for d in tqdm(self.data, desc="Approximating Koopman Operator"):
            psi_x = self.forward(d[:-1].to(self.device))
            psi_y = self.forward(d[1:].to(self.device))
            psi_xT = psi_x.T
            G = torch.matmul(psi_xT, psi_x) + self.lamb * torch.eye(psi_xT.shape[0], device=psi_xT.device)
            A = torch.matmul(psi_xT, psi_y)
            K = torch.linalg.solve(G, A)
            self.A_vs.append(K)
        return self.A_vs

class ResKoopNet:
    """
    Residual Koopman Network. To obtain a neural network-based Koopman dictionary for analyzing model representations.
    """
    def __init__(self, data, rank=15, lamb=1e-3, learning_rate=0.001, epochs=10, device='cuda'):
        self.rank = rank
        self.lamb = lamb
        self.data = data
        self.epochs = epochs
        self.make_dataset()
        self.reskoopnet = nn.Sequential(
            nn.Linear(data.shape[-1], rank),
            nn.Sigmoid()
        ).to(device)
        self.optimizer = torch.optim.Adam(self.reskoopnet.parameters(), lr=learning_rate)

        self.A_v = None

    def make_dataset(self):
        if self.data.ndim == 2:
            # For tensor of shape T x D (one trial), generate a dataset using sliding window:
            win_size = 1000  # you can make this parameterizable if needed
            T, D = self.data.shape
            if T < win_size:
                raise ValueError(f"Time dimension {T} is smaller than window size {win_size}.")
            # Stride shape to (B, win_size, D)
            dataset = []
            for start in range(T - win_size + 1):
                window = self.data[start:start+win_size]  # shape: (win_size, D)
                dataset.append(window)
            self.dataset = torch.stack(dataset, dim=0)  # shape: (B, win_size, D)
        elif self.data.ndim == 3:
            self.dataset = torch.stack(self.data, dim=0)
        else:
            raise ValueError(f"Invalid data shape: {self.data.shape}. Expected 2D (samples, features) or 3D (trials, samples, features)")

    def forward(self, x):
        return self.reskoopnet(x)

    def residual_loss(self, batch_data):
        X, Y = batch_data[:, :-1], batch_data[:, 1:]
        psi_x = self.forward(X).reshape(-1, self.rank)
        psi_y = self.forward(Y).reshape(-1, self.rank)
        psi_xT = psi_x.T
        G = torch.matmul(psi_xT, psi_x) + self.lamb * torch.eye(psi_xT.shape[0], device=psi_xT.device)
        A = torch.matmul(psi_xT, psi_y)
        K = torch.linalg.solve(G, A)
        _, S, Vh = torch.linalg.svd(K, full_matrices=False)
        S_diag = torch.diag(S)
        psi_x_v = torch.matmul(psi_x, Vh)
        psi_x_v_k = torch.matmul(psi_x_v, S_diag)
        psi_y_v = torch.matmul(psi_y, Vh)
        J = torch.norm(psi_y_v - psi_x_v_k, p='fro')
        return J

    def fit_koopman_dict(self, saved_path):
        dataset = TensorDataset(self.dataset)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        self.reskoopnet.train()
        from tqdm import tqdm

        from tqdm import trange

        epoch_bar = trange(self.epochs, desc="Training Epochs")
        for epoch in epoch_bar:
            epoch_losses = []
            for batch_data in dataloader:
                self.optimizer.zero_grad()
                loss = self.residual_loss(batch_data[0]) / batch_data[0].shape[0]
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            epoch_bar.set_postfix({'avg_loss': avg_epoch_loss})
        torch.save(self.reskoopnet.state_dict(), f'{saved_path}/reskoopnet.pt')

    def approx_koopman_op(self):
        psi_x = self.forward(self.data[:-1])
        psi_y = self.forward(self.data[1:])
        psi_xT = psi_x.T
        G = torch.matmul(psi_xT, psi_x) + self.lamb * torch.eye(psi_xT.shape[0], device=psi_xT.device)
        A = torch.matmul(psi_xT, psi_y)
        K = torch.linalg.solve(G, A)

        self.A_v = K
