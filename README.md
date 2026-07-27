# Neural Network Surrogate Model for Bloch Simulations in MR-STAT

This repository contains the code for training and evaluating a neural network surrogate model for MR-STAT signal simulation. The surrogate predicts the MR signal and its derivatives with respect to tissue parameters. 

The surrogate model is implemented in PyTorch, while Bloch simulations for data generation are performed in Julia using BlochSimulators.jl. Communication between Python and Julia is handled through JuliaCall.

## Folder description

### `config/`

Contains all configuration files.

- **experiment.yaml** – experiment-wide settings (random seed, device and output paths)
- **data.yaml** – dataset generation settings (number of samples and sequence parameters)
- **model.yaml** – model architecture settings (number of layers, hidden sizes, and output dimensions)
- **training.yaml** – optimizer and training settings (number of epochs, batch sizes, and learning rate scheduler)
- **testing.yaml** – evaluation and inference benchmark settings (batch sizes and number or runs)

---

### `data/`

Contains all data-related code.

- **data_generation.jl** – Julia implementation for generating Bloch simulation training data and saving it to a HDF5 file
- **data_loading.py** – loading datasets and creating PyTorch `DataLoader`s
- **split_dataset.py** – splitting a pre-generated HDF5 dataset into fixed validation and test sets
- **datasets/** – location where generated HDF5 datasets will be stored
- **rf_data/** – RF pulse sequence data used during data generation

Training data is generated online during training using Julia Bloch simulations. Validation and test datasets are generated separately and stored as fixed HDF5 files.

---

### `outputs/`

Contains automatically generated output files.

- TensorBoard log files
- Saved model checkpoints (`.pt`)

---

### `scripts/`

Contains scripts called by `run.py`.

- **train.py** – trains the surrogate model
- **test_accuracy.py** – evaluates prediction accuracy on the test dataset
- **test_inference.py** – benchmarks surrogate inference speed

---

### `src/`

Contains the core implementation.

- **model.py** – neural network architecture
- **trainer.py** – training loop, validation, checkpointing, and logging
- **loss.py** – loss function definitions
- **utils.py** – helper functions (device selection, model loading, random seeds, etc.)
- **config.py** – configuration file loader

---

### `run.py`


`run.py` is the main entry point of the repository.

Available tasks:

- `generate` – generate validation and test datasets using Bloch simulations
- `train` – train the surrogate model
- `test_accuracy` – evaluate prediction accuracy
- `test_inference` – benchmark inference speed

## Usage

The tasks should be executed in the following order. Each step creates files required by later steps:

### 1. Generate a dataset

```bash
uv run run.py generate
```

This generates a validation and test set using Julia Bloch simulations and stores it in `data/datasets/`.

### 2. Train the surrogate model

```bash
uv run run.py train
```

Model checkpoints and TensorBoard logs are written to `outputs/`.

### 3. Evaluate prediction accuracy

```bash
uv run run.py test_accuracy
```

Evaluates the trained model on the test dataset.

### 4. Benchmark inference speed

```bash
uv run run.py test_inference
```

Measures surrogate inference performance under different execution settings (FP32, FP16, `torch.compile`, etc.).

## Configuration

All settings are stored in the `config/` directory. Hyperparameters can be modified without changing the source code.

## Dependencies

Python dependencies are managed using `uv` and specified in `pyproject.toml`.

Julia is used for Bloch simulations through JuliaCall. The required Julia packages are:

- **BlochSimulators.jl** 
- **HDF5.jl** 
- **ComputationalResources.jl**
- **DelimitedFiles.jl**

## Reference

Architecture based on:  
Liu, H., van der Heide, O., Mandija, S., van den Berg, C. A. T., & Sbrizzi, A. (2022). Acceleration Strategies for MR-STAT: Achieving High-Resolution Reconstructions on a Desktop PC Within 3 Minutes. IEEE transactions on medical imaging, 41(10), 2681-2692. https://doi.org/10.1109/TMI.2022.3168436

