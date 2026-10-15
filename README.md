# NeuralBloch

Neural network surrogate model for Bloch simulations in MR-STAT.

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
- **export.yaml** – model export settings (output paths per target, precision, TensorRT batch size profile, and the model bundle's location, family and version)

---

### `data/`

Contains all data-related code.

- **data_generation.jl** – Julia implementation for generating Bloch simulation training data and saving it to a HDF5 file
- **data_loading.py** – loading datasets and creating PyTorch `DataLoader`s
- **split_dataset.py** – splitting a pre-generated HDF5 dataset into fixed validation and test sets
- **datasets/** – location where generated HDF5 datasets will be stored
- **rf_data/** – RF pulse sequence data used during data generation (which files are used is set by `data.RF_ANGLE_FILE` / `data.RF_PHASE_FILE`)

Training data is generated online during training using Julia Bloch simulations. Validation and test datasets are generated separately and stored as fixed HDF5 files.

Dataset derivatives use `simulate_derivatives_forward_sensitivity` by default. Set
`data.DERIVATIVE_METHOD` to `finite_difference` to use the previous method.
Forward sensitivity requires BlochSimulators v0.10 or newer. It is now as fast
as finite differences, needs no step-size choice, and gives derivatives exact
up to floating-point error rather than a finite-difference approximation.

For a shared-basis model (`model.derivative_mode: forward`), setting
`model.basis_initialization: joint_pca` initializes its decoder from the
leading singular vectors of row-normalized signal and derivative targets.
These span the rank-limited subspace with the smallest squared projection
error on the initialization batch. Training then updates the decoder weights.

---

### `outputs/`

Contains automatically generated output files.

- TensorBoard log files
- Saved model checkpoints (`.pt`)
- Exported ONNX models (`.onnx`) and TensorRT engines (`.engine`), one of each per target
- The model bundle for NeuralBloch.jl (`bundle/`) and its release file (`neuralbloch-model-<family>-v<version>.tar.gz`)

---

### `scripts/`

Contains scripts called by `run.py` and standalone post-training commands.

- **train.py** – trains the surrogate model
- **test_accuracy.py** – evaluates prediction accuracy on the test dataset
- **test_inference.py** – benchmarks surrogate inference speed
- **orthogonalize_basis.py** – optionally orders a trained shared basis and saves smaller-rank checkpoints
- **export_onnx.py** – exports the trained model to ONNX
- **build_engine.py** – builds a TensorRT engine from each exported ONNX model
- **export_bundle.py** – writes the model bundle NeuralBloch.jl reads, and its release file

---

### `src/`

Contains the core implementation.

- **model.py** – neural network architecture
- **basis.py** – post-training shared-basis ordering and rank truncation
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
- `export_onnx` – export the trained model to ONNX, one model per target
- `build_engine` – build a TensorRT engine from each exported ONNX model
- `export_bundle` – write the model bundle NeuralBloch.jl reads

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

For a forward-mode model, basis ordering is an optional step after training.
The trained decoder can lose its initial orthogonality and order. This command
saves an equivalent full-rank checkpoint and separate smaller-rank checkpoints;
the original checkpoint is untouched:

```bash
uv run python -m scripts.orthogonalize_basis \
  --checkpoint outputs/model.pt \
  --fit-dataset data/datasets/dataset_for_validation.h5 \
  --output outputs/model_ordered.pt \
  --truncate 16 24 32
```

The full-rank rewrite preserves predictions up to floating-point error. Smaller
ranks approximate the trained model; compare their reconstruction speed and
accuracy before choosing a rank. Set `experiment.PT_FILE_PATH` to the selected
rank checkpoint when exporting the surrogate for reconstruction.

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

### 5. Export the trained model to ONNX

```bash
uv run run.py export_onnx
```

Reads the trained checkpoint (`experiment.PT_FILE_PATH`) and exports one ONNX model per target, each with a dynamic batch dimension, to the paths in `export.TARGETS`:

| Target | Model method | Outputs |
|---|---|---|
| `signal_and_derivatives` | `forward` | `signal`, `dT1`, `dT2`, each `(batch, NTR, 2)` |
| `signal` | `forward_signal` | `signal`, `(batch, NTR, 2)` |
| `coefficients` | `forward_coefficients` | `coefficients`, `(batch, rank)`: the signal's basis coefficients |

All three take the input `x`, `(batch, 3)` physical parameters `[T1, T2, B1]`. A TensorRT engine computes every output of its model on each call, so a separate model per target lets a caller skip the work it does not need. Each export is verified against the PyTorch model using ONNX Runtime, on parameters sampled from the checkpoint's training domain. With `export.FP16` enabled, the models run internally in FP16 while their inputs and outputs stay FP32.

### 6. Build the TensorRT engines

```bash
uv run run.py build_engine
```

Compiles each exported ONNX model into a serialized TensorRT engine with a dynamic batch size profile, checks that the engine has exactly its target's inputs and outputs, and verifies it against PyTorch. TensorRT networks are strongly typed, so the engine precision is inherited from the ONNX model. Requires a CUDA GPU; the engines are specific to the GPU architecture and TensorRT version they were built with.

### 7. Export the model bundle

```bash
uv run run.py export_bundle
```

Writes the model bundle that [NeuralBloch.jl](https://github.com/NLeSC-COMPAS/NeuralBloch.jl) reads to `export.BUNDLE_DIR`: the three ONNX models of step 5, the RF tables of the training sequence, and a generated `card.toml` (format 3). The card states the model's family and version (`export.MODEL_FAMILY`, `export.MODEL_VERSION`), its readouts and basis rank, each ONNX model's sha256, the training domain, how the training derivatives were computed, the sequence, and where the model came from: the NeuralBloch commit, the checkpoint's epoch, a hash of its configuration, the BlochSimulators version and the precision. Every field comes from the checkpoint or the export; each carries a comment naming its source.

So that the card describes the checkpoint truthfully, the task refuses:

- a checkpoint that does not store its training configuration (`model_config`, `data_config`);
- a `MODEL_FAMILY` whose `rank<N>` is not the rank of the checkpoint's signal decoder;
- ONNX models that do not reproduce the checkpoint (each is verified against it again);
- uncommitted changes in this repository, since the card names its commit.

A family is one contract for callers: the sequence, its number of readouts and the basis rank, named `<sequence><ntr>-rank<rank>` (for example `fisp600-rank16`). Its version is a [semantic version](https://semver.org), written in quotes in `export.yaml`: `MAJOR.MINOR.PATCH`, optionally with a pre-release such as `-beta.1`, and without build metadata (`+…`), because the version is part of the release tag. Each version of a family is a drop-in replacement for the one before:

- PATCH: the same weights, exported or packaged again;
- MINOR: new weights or a wider training domain;
- MAJOR: stays 0 until the family is stable, and 1.0.0 marks its first stable model;
- a pre-release (`-beta.1`, `-rc.1`) marks a candidate.

### 8. Publish a model release

`export_bundle` also writes the bundle's release file next to it: `outputs/neuralbloch-model-<family>-v<version>.tar.gz`, the bundle's files at the root of a reproducible tarball (the same bundle always gives the same bytes). NeuralBloch.jl downloads this file from the release `model-<family>-v<version>` of this repository, as a Julia artifact whose hashes it pins. The task prints the file's sha256 and the command that would create the release; it publishes nothing itself. To publish a model:

1. Push the commit that the card names (step 7 exports from a clean, committed tree).
2. In NeuralBloch.jl, bind the file with `julia --project dev/bind_bundle.jl /path/to/the/file.tar.gz` and run its tests and `scripts/validate.jl`.
3. Create the release with the printed `gh release create` command, or in GitHub's web interface: tag `model-<family>-v<version>` at the card's commit, and attach the tarball as the only file. A version with a pre-release part (`-beta.1`) is published as a pre-release; any other version is not marked latest, so the latest release stays the code's.

Never replace a published file: NeuralBloch.jl pins its hashes, so every commit that bound it would stop installing. A corrected model is a new version.

## Configuration

All settings are stored in the `config/` directory. Hyperparameters can be modified without changing the source code.

## Development

Development guidelines are documented in [README.dev.md](README.dev.md), inlucding dev environment setup, dependency management, code style, git hooks, etc.

## Reference

Architecture based on:
Liu, H., van der Heide, O., Mandija, S., van den Berg, C. A. T., & Sbrizzi, A. (2022). Acceleration Strategies for MR-STAT: Achieving High-Resolution Reconstructions on a Desktop PC Within 3 Minutes. IEEE transactions on medical imaging, 41(10), 2681-2692. https://doi.org/10.1109/TMI.2022.3168436
