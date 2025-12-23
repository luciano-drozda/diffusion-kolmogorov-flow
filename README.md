# Latent Diffusion Model for super-resolution of JAX-CFD 2D Kolmogorov flow data

![](ldm.gif)

[Create a Python virtual environment](https://docs.python.org/3/library/venv.html#creating-virtual-environments), [activate it](https://docs.python.org/3/library/venv.html#how-venvs-work), and install dependencies via the command below:

```bash
pip install -r requirements.txt
```

## Walkthrough

### Step 1: Generate data

Run `data/generate.ipynb`. It stores the following files in disk:

- `data/data.png`: An image showing snapshots of JAX-CFD simulation data (vorticity field);

- `data/data.h5`: JAX-CFD simulation data with shape `(nb_snapshots, 256, 256)`;

- `data/data_normalized.h5`: A normalized version of JAX-CFD simulation data where values lie in the range `[-1,1]`. It also includes normalized data downsampled to a `(64, 64)` grid (scale factor of `1/4`);

- `data/data_normalized.png`: An image showing snapshots of JAX-CFD normalized simulation data;

- `data/data_normalized_lr.png`: An image showing snapshots of JAX-CFD normalized simulation data downsampled to a `(64, 64)` grid.

### Step 2: Train a Latent Diffusion Model

Run `train.ipynb`. It stores the following files in disk:

- `model_tuned.ckpt`: A version of `CompVis/latent-diffusion` super-resolution model fine-tuned on JAX-CFD data.

### Step 3: Call trained model

Run `inference.ipynb`. It stores the following files in disk:

- `ldm.gif`: An animation comparing LDM output with ground-truth from JAX-CFD simulation data. It also includes downsampled JAX-CFD simulation data as LDM Input / Conditioning;

## Acknowledgements
The code in this repo heavily borrows from [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion) and uses [JAX-CFD](https://github.com/google/jax-cfd) for data generation.