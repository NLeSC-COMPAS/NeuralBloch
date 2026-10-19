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
- **export.yaml** – model export settings (output paths for each target, precision, TensorRT batch size profile, and the location, family and version of the model bundle)

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
- Exported ONNX models (`.onnx`) and TensorRT engines (`.engine`): one model and one engine for each target
- The model bundle (`bundle/`) and its release file (`neuralbloch-model-<family>-v<version>.tar.gz`)

---

### `scripts/`

Contains scripts called by `run.py` and standalone post-training commands.

- **train.py** – trains the surrogate model
- **test_accuracy.py** – evaluates prediction accuracy on the test dataset
- **test_inference.py** – benchmarks surrogate inference speed
- **orthogonalize_basis.py** – optionally orders a trained shared basis and saves smaller-rank checkpoints
- **export_onnx.py** – exports the trained model to ONNX
- **build_engine.py** – builds a TensorRT engine from each exported ONNX model
- **export_bundle.py** – writes the model bundle and its release file

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
- `export_onnx` – export the trained model to ONNX (one model for each target)
- `build_engine` – build a TensorRT engine from each exported ONNX model
- `export_bundle` – write the model bundle and its release file

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

Reads the trained checkpoint (`experiment.PT_FILE_PATH`) and exports one ONNX model for each target to the paths in `export.TARGETS`. Each model has a dynamic batch dimension.

| Target | Model method | Outputs |
|---|---|---|
| `signal_and_derivatives` | `forward` | `signal`, `dT1`, `dT2`, each `(batch, NTR, 2)` |
| `signal` | `forward_signal` | `signal`, `(batch, NTR, 2)` |
| `coefficients` | `forward_coefficients` | `coefficients`, `(batch, rank)`, the basis coefficients of the signal |

All three models take the input `x`, `(batch, 3)`, which contains the physical parameters `[T1, T2, B1]`. A TensorRT engine calculates all the outputs of its model on each call. Thus, a separate model for each target lets a caller do only the work that it needs. The task uses ONNX Runtime to compare each export with the PyTorch model. It uses parameters sampled from the training domain of the checkpoint. If `export.FP16` is `true`, the models run internally in FP16, but their inputs and outputs stay FP32.

### 6. Build the TensorRT engines

```bash
uv run run.py build_engine
```

Compiles each exported ONNX model into a serialized TensorRT engine with a dynamic batch size profile. Then the task makes sure that each engine has exactly the inputs and outputs of its target, and compares the engine with PyTorch. TensorRT networks are strongly typed, so each engine gets its precision from the ONNX model. This task requires a CUDA GPU. An engine works only with the GPU architecture and the TensorRT version that built it.

### 7. Export the model bundle

```bash
uv run run.py export_bundle
```

Writes the model bundle to `export.BUNDLE_DIR`: the three ONNX models of step 5, the RF tables of the training sequence, and `model_card.toml`. The card describes the model: its family and version, shapes, training domain, sequence and origin. Next to the bundle, the task writes the release file `neuralbloch-model-<family>-v<version>.tar.gz` (step 8).

The task checks the checkpoint and the ONNX models before it writes. It stops if the repository has uncommitted changes, because the card records the commit. It also stops if the release of the family version exists (step 8), or if it cannot get the tags of `origin` to find out. To continue without `origin`, set `export.OFFLINE: true`. For the card fields, the checks and the version rules, refer to `scripts/export_bundle.py` and `config/export.yaml`.

### 8. Publish a model release

A model release is a GitHub release with the tag `model-<family>-v<version>` and one file: the release file of step 7. To publish a model, do these steps:

1. Set `MODEL_FAMILY` and `MODEL_VERSION` in `config/export.yaml`. Then commit and push.
2. Do steps 5 to 7. Keep the sha256 and the `gh release create` command that step 7 prints.
3. Create the release with this command, or in the GitHub web interface. In the web interface, create the tag at the commit that the card records, not at `main`. Attach the release file. Publish a pre-release version (`-beta.1`) as a pre-release. Do not mark other versions as latest.
4. Make sure that the sha256 of the published file is the printed sha256.

CAUTION: Do not change a published file, because users can pin its sha256. To correct a model, publish a new version.

## Configuration

All settings are stored in the `config/` directory. Hyperparameters can be modified without changing the source code.

## Development

Development guidelines are documented in [README.dev.md](README.dev.md), inlucding dev environment setup, dependency management, code style, git hooks, etc.

## Reference

Architecture based on:
Liu, H., van der Heide, O., Mandija, S., van den Berg, C. A. T., & Sbrizzi, A. (2022). Acceleration Strategies for MR-STAT: Achieving High-Resolution Reconstructions on a Desktop PC Within 3 Minutes. IEEE transactions on medical imaging, 41(10), 2681-2692. https://doi.org/10.1109/TMI.2022.3168436
