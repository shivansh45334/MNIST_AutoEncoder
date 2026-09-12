# MNIST Autoencoder

A small PyTorch project that implements a **fully connected Autoencoder** from scratch and trains it on the **MNIST handwritten-digit dataset**.

The project focuses on understanding how Autoencoders learn compressed representations of images through an encoder–decoder architecture. The notebook covers the complete workflow, from loading MNIST and training the model to reconstructing images and visualizing the learned latent space.

---

## Project Overview

An Autoencoder learns to reconstruct its input:

```text
Input Image
     │
     ▼
  Encoder
     │
     ▼
Latent Representation
     │
     ▼
  Decoder
     │
     ▼
Reconstructed Image
```

Instead of learning to classify digits, the model learns a compressed representation that preserves the information necessary to reconstruct the original image.

For this project:

```text
784 → 256 → 128 → 256 → 784
```

The `128`-dimensional latent representation acts as the bottleneck of the network.

---

## Project Structure

```text
.
├── pyproject.toml
├── requirements.txt
├── src/
│   └── autoencoder/
│       ├── __init__.py
│       ├── Autoencoder_implimentation.ipynb
│       └── data/
│           └── MNIST/
└── README.md
```

The main implementation is contained in:

```text
src/autoencoder/Autoencoder_implimentation.ipynb
```

The notebook is the executable entry point for the project. The Python package module is currently a minimal placeholder.

---

## Requirements

* Python 3.14 or newer
* PyTorch
* torchvision
* NumPy
* Matplotlib
* scikit-learn
* Jupyter/IPython kernel
* CUDA-capable GPU *(optional)*

The notebook automatically selects CUDA when a compatible GPU is available and otherwise falls back to the CPU.

---

## Setup

### 1. Clone the repository

```powershell
git clone <repository-url>
cd Autoencoder
```

### 2. Create a virtual environment

Using Python's built-in `venv`:

```powershell
py -3.14 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Install the project

To install the package in editable mode:

```powershell
python -m pip install -e .
```

### Using `uv`

The project also contains a `pyproject.toml` configured for `uv`. Install `uv` first if it is not already available.

If `uv` is installed:

```powershell
uv venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
uv pip install -e .
```

---

## Run the Notebook

Open the notebook at:

```text
src/autoencoder/Autoencoder_implimentation.ipynb
```

in **VS Code** or **Jupyter Notebook**.

Then:

1. Select the project's `.venv` as the notebook kernel.
2. Run the cells from top to bottom.
3. Allow `torchvision` to download MNIST automatically.
4. Inspect the reconstruction results.
5. Explore the latent-space visualizations.

The dataset does not need to be downloaded manually.

The dataset cell uses:

```python
root="./data"
```

for the MNIST dataset. This path is relative to the notebook kernel's current working directory, not to the notebook file itself. When the notebook is run from the repository root, the dataset will be stored under:

```text
data/MNIST/
```

If the notebook is launched with a different working directory, MNIST will be downloaded to that directory's `data/MNIST/` folder.

The `autoencoder` console command defined in `pyproject.toml` currently calls a placeholder `main()` function and is not the training entry point. Run the notebook for the actual implementation.

---

## Notebook Workflow

The notebook follows a progressive learning workflow:

### 1. Environment Setup

Import the required libraries and automatically select CPU or CUDA.

### 2. Dataset Preparation

Load MNIST, convert images to tensors, and create batches using a PyTorch `DataLoader`.

### 3. Understand the Input

MNIST images have the shape:

```text
1 × 28 × 28
```

Because the model uses fully connected layers, each image is flattened into:

```text
784 values
```

### 4. Build the Autoencoder

Define separate encoder and decoder networks using PyTorch's `nn.Sequential` and `nn.Linear` layers.

### 5. Train the Model

Train the network using:

* MSE reconstruction loss
* Adam optimizer
* Learning rate: `0.001`
* Batch size: `128`
* Epochs: `20`

### 6. Image Reconstruction

Compare original MNIST images with the images reconstructed by the decoder.

### 7. Extract Latent Representations

Pass the complete training dataset through the encoder and collect the resulting latent vectors.

### 8. Visualize the Latent Space

Analyze the learned representation directly when possible or use PCA when the latent space has more than three dimensions.

---

# Architecture

Each MNIST image is a grayscale image of size:

```text
28 × 28
```

The image is flattened before being passed to the fully connected network.

```text
                    INPUT
                 1 × 28 × 28
                      │
                      ▼
                   Flatten
                      │
                      ▼
                 784 values
                      │
                      ▼
               Linear + ReLU
                      │
                      ▼
                 256 values
                      │
                      ▼
               Linear + ReLU
                      │
                      ▼
               Latent Space
                 128 values
                      │
                      ▼
               Linear + ReLU
                      │
                      ▼
                 256 values
                      │
                      ▼
              Linear + Sigmoid
                      │
                      ▼
                784 values
                      │
                      ▼
                  Reshape
                      │
                      ▼
                28 × 28 IMAGE
```

### Encoder

The encoder performs the compression:

```text
784 → 256 → 128
```

### Latent Representation

The bottleneck contains:

```text
128 values
```

This representation is a compressed version of the original 784-dimensional image.

### Decoder

The decoder reconstructs the image:

```text
128 → 256 → 784
```

The final `Sigmoid` activation produces values in the `[0, 1]` range, matching the normalized pixel values produced by `transforms.ToTensor()`.

---

# How the Autoencoder Learns

## Self-Supervised Reconstruction

The Autoencoder does not require the MNIST digit labels during training.

Instead, the input itself becomes the target:

```text
             ┌───────────────┐
             │               │
             ▼               │
x → Encoder → z → Decoder → x̂
                            │
                            ▼
                     Reconstruction
                         Loss
```

Where:

* `x` = original image
* `z` = latent representation
* `x̂` = reconstructed image

The training objective is to make:

```text
x̂ ≈ x
```

---

## Reconstruction Loss

The project uses **Mean Squared Error (MSE)**:

```text
MSE(x, x̂) = mean((x - x̂)²)
```

The loss compares every pixel of the reconstructed image with the corresponding pixel in the original image.

During training, the optimizer updates the network parameters to minimize this reconstruction error.

---

## Optimization

The model uses the **Adam optimizer**:

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
```

Training configuration:

| Parameter        | Value |
| ---------------- | ----: |
| Batch Size       |   128 |
| Epochs           |    20 |
| Learning Rate    | 0.001 |
| Optimizer        |  Adam |
| Loss             |   MSE |
| Latent Dimension |   128 |
| Hidden Dimension |   256 |

---

# Latent Space Visualization

After training, the encoder is used to generate a latent representation for every MNIST image.

With the default architecture:

```text
60,000 images × 128 latent dimensions
```

The MNIST labels are collected **only for visualization**.

They are not used to train the Autoencoder.

This allows us to investigate whether visually similar digits naturally occupy nearby regions of the learned latent space.

---

## Visualization Strategy

The notebook automatically chooses a visualization based on the latent dimensionality.

### 1D Latent Space

For one-dimensional representations:

* Class-specific histograms
* Latent-value strip plot

### 2D Latent Space

For two-dimensional representations:

```text
z₁ vs z₂
```

is plotted directly.

### 3D Latent Space

For three-dimensional representations, a 3D scatter plot is generated.

### High-Dimensional Latent Space

For latent spaces with more than three dimensions, **Principal Component Analysis (PCA)** is used.

For the default 128-dimensional representation:

```text
128D latent space
        │
        ▼
       PCA
        │
        ▼
     2D space
```

The notebook also reports the explained variance ratio of the selected principal components.

---

# Results

The notebook produces two primary types of results.

## Image Reconstruction

Original images are compared against their reconstructed versions:

```text
Original          Reconstruction
─────────         ───────────────
  digit     →       generated
  image             reconstruction
```

This provides a visual indication of how well the Autoencoder has learned to preserve the important information in the input images.

## Latent Space

The learned latent representations are projected into a space that can be visualized.

By coloring the points according to their MNIST labels, we can inspect whether different digits form distinguishable regions.

> Note: Because the labels are not used during training, any visible grouping is a property of the representation learned through reconstruction rather than supervised classification.

---

# Key Concepts Demonstrated

This project provides a practical implementation of several important deep-learning concepts:

* Neural network encoders and decoders
* Representation learning
* Dimensionality reduction
* Bottleneck architectures
* Self-supervised learning
* Reconstruction loss
* Backpropagation
* Adam optimization
* PyTorch `nn.Module`
* PyTorch `DataLoader`
* GPU/CPU device management
* Latent-space analysis
* PCA visualization

---

# Learning Objective

The primary goal of this project is not simply to train an Autoencoder, but to understand **how neural networks can learn useful representations without explicit class labels**.

The key idea is:

```text
High-dimensional input
        │
        ▼
    Encoder
        │
        ▼
Compact representation
        │
        ▼
    Decoder
        │
        ▼
Reconstructed input
```

The bottleneck forces the network to learn which information is important enough to preserve.

---

# Tech Stack

```text
Python
PyTorch
Torchvision
NumPy
Matplotlib
Scikit-learn
Jupyter Notebook
```

