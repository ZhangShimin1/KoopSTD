<div align="center">
<h1>KoopSTD </h1>
<h3>Reliable Similarity Analysis between Dynamical Systems via Approximating Koopman Spectrum with Timescale Decoupling</h3>


[Shimin Zhang](https://scholar.google.com/citations?hl=en&user=iDKLHNMAAAAJ)<sup>1</sup> \*,[Ziyuan Ye](https://scholar.google.com/citations?user=CmzVixkAAAAJ)<sup>1</sup> \*, [Yinsong Yan](https://openreview.net/profile?id=~Yinsong_Yan1)<sup>1</sup>, [Zeyang Song](https://scholar.google.com/citations?user=iTf0gegAAAAJ)<sup>1</sup> <sup>2</sup>, [Yujie Wu](https://scholar.google.com/citations?user=-lw0UPkAAAAJ)<sup>1</sup>, [Jibin Wu](https://scholar.google.com/citations?user=QwDyvrgAAAAJ)<sup>1 :email:</sup>

<sup>1</sup>  The Hong Kong Polytechnic University, <sup>2</sup> National University of Singapore

(\*) equal contribution, (<sup>:email:</sup>) corresponding author.

ICML 2025 [(paper)](https://openreview.net/forum?id=29eZ8pWc8E)
</div>


## Environment setup
Python 3.9+ is required. We recommend installing KoopSTD as a package in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch first (CPU or CUDA),:

```bash
# CPU (default PyPI wheels)
python -m pip install "torch>=2.2" "torchvision>=0.17" "torchaudio>=2.2"

# OR CUDA wheels (example: CUDA 12.1)
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  "torch>=2.2" "torchvision>=0.17" "torchaudio>=2.2"

# Install KoopSTD in editable mode
python -m pip install -e .
```

Optional dependency groups:

```bash
# Extra packages used by examples
python -m pip install -e ".[examples]"
```

# Basic usage
For the experiments presented in our paper, we compare the distance between every pair of samples in a dataset.
```python
from koopstd.dynametric import KoopOpMetric
from koopstd.eval import MetricEvaluator

# Step 1: Prepare your data
# data should be a list of numpy arrays or torch tenosrs
# Each array represents a trajectroy with the shape of (n_trials, n_timepoints, n_dimensions) or (n_timepoints, n_dimensions) sampled by the system
data = [trajectory1, trajectory2, ...]

# Step 2: Set up KoopSTD parameters
koopstd_params = {
    'hop_size': 128,        # Window sliding step size of STFT
    'win_len': 1024,        # Window length STFT
    'rank': 5,              # The first rank modes with the smallest residual
    'lamb': 0               # Regularization parameter
}

# Step 3: Set up distance metric parameters
distance_params = {
    'p': 1,                 # Order for Wasserstein distance
    'method': 'emd'         # Earth Mover's Distance
}

# Step 4: Initialize KoopSTD metric
koopstd = KoopOpMetric(
    X=data,                          # Your trajectory data
    kmd_method='koopstd',            
    kmd_params=koopstd_params,       # KoopSTD parameters
    dist='wasserstein',              # Distance metric
    dist_params=distance_params,     # Distance parameters
    device='cuda'                    # Specify the gpu
)

# Step 5: Compute distance matrix
distance_matrix = koopstd.fit_score()
```

## Citation
```
@inproceedings{zhang2025koopstd,
  title={Reliable Similarity Analysis between Dynamical Systems via Approximating Koopman Spectrum with Timescale Decoupling},
  author={Zhang, Shimin and Ye, Ziyuan and Yan, Yinsong and Song, Zeyang and Wu, Yujie and Wu, Jibin},
  booktitle={ICML},
  year={2025}
}
```
