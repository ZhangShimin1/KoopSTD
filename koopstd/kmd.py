import numpy as np
import cupy as cp
import torch
import time



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
        if self.E.ndim == 3: #flatten across trials for 3d
            E = self.E.reshape(self.E.shape[0] * self.E.shape[1], self.E.shape[2])
        else:
            E = self.E
        
        U, self.S, self.V = torch.linalg.svd(E.T, full_matrices=False)
        # print(U.shape, self.S.shape, self.V.shape)    

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
    

class HAVOK(KMD):
    def __init__(self, data, rank=15, lamb=0., n_delays=8, delay_interval=1,
            backend='numpy', 
            device='cpu',
            verbose=False
        ):
        super().__init__(data, rank, lamb, backend, device, verbose)
        self.n_delays = n_delays
        self.delay_interval = delay_interval
        self.rank = rank
        self.lamb = lamb
        self.data = data

    def embed(self):
        # Hankel (delay) Embedding
        if self.data.shape[int(self.data.ndim==3)] - (self.n_delays - 1) * self.delay_interval < 1:
            raise ValueError("The number of delays is too large for the number of time points in the data!")
        
        if self.data.ndim == 3:
            embedding = torch.zeros((self.data.shape[0], self.data.shape[1] - (self.n_delays - 1) * self.delay_interval, self.data.shape[2] * self.n_delays))
        else:
            embedding = torch.zeros((self.data.shape[0] - (self.n_delays - 1) * self.delay_interval, self.data.shape[1] * self.n_delays))
        
        for d in range(self.n_delays):
            index = (self.n_delays - 1 - d) * self.delay_interval
            ddelay = d * self.delay_interval

            if self.data.ndim == 3:
                ddata = d * self.data.shape[2]
                embedding[:,:, ddata: ddata + self.data.shape[2]] = self.data[:,index:self.data.shape[1] - ddelay]
            else:
                ddata = d * self.data.shape[1]
                embedding[:, ddata:ddata + self.data.shape[1]] = self.data[index:self.data.shape[0] - ddelay]
    
        self.E = embedding.to(self.device)
        if self.n_trials == 1:
            self.E = self.E.squeeze(0)

    def reduced_rank(self):
        if self.n_trials > 1:
            V = self.V.reshape(self.E.shape)
            new_shape = (self.E.shape[0] * (self.E.shape[1] - 1), self.E.shape[2])
            V_minus = V[:, :-1].reshape(new_shape)
            V_plus = V[:, 1:].reshape(new_shape)
        else:
            V_minus = V[:-1]
            V_plus = V[1:]
        
        self.W = V_minus[:, :self.rank]
        self.W_prime = V_plus[:, :self.rank]

