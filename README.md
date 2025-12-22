# Latent Diffusion Model for super-resolution of JAX-CFD 2D Kolmogorov flow data

Create a [Python virtual environment](https://docs.python.org/3/library/venv.html#creating-virtual-environments), [activate it](https://docs.python.org/3/library/venv.html#how-venvs-work), and install dependencies via the command below:

```bash
pip install -r requirements.txt
```

## Walkthrough

### Step 1: Generate data

Run `data/generate.ipynb`. It stores the following files in disk:

- `data/data.png`:

- `data/data.h5`:

- `data/data_normalized.h5`: A normalized version where values lie in the range \[-1,1\].

### Step 2: Train a Latent Diffusion Model

Run `train.ipynb`.

### Step 3: Call trained model

Run `inference.ipynb`.

## Acknowledgements
The code in this repo heavily borrows from [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion) and uses [JAX-CFD](https://github.com/google/jax-cfd) for data generation.