# Multiple Instance Learning (MIL) with Attention

This project implements an Attention-based Multiple Instance Learning model for processing patient data where each patient has multiple instances (samples). The model learns to identify important instances through an attention mechanism.

## Project Structure

1. `create_sample_data.py`

   - Generates synthetic training and validation datasets
   - Creates patient instances with both positive and negative labels
   - Each patient has multiple instances with feature vectors
   - Saves data in JSON format in the `data/` directory
   - Includes statistics printing functionality to show dataset characteristics

2. `defs.py`

   - Contains the base data structure definition
   - Defines `PatientInstances` class using Pydantic for data validation
   - Specifies the schema for patient data (name, label, and features)

3. `dataloader.py`

   - Implements the PyTorch Dataset and DataLoader functionality
   - `MILDataset` class handles loading and processing of JSON data
   - Provides padding and masking for variable-length instance sets
   - Includes utility functions `get_train_loader` and `get_val_loader`

4. `model.py`

   - Implements the `AttentionMIL` neural network model
   - Includes:
     - Instance-level feature extraction
     - Attention mechanism for instance importance weighting
     - Final classification layer
   - Includes methods for forward pass and attention weight extraction

5. `training_manager.py`

   - Manages the training process
   - Handles:
     - Training loop implementation
     - Validation
     - Early stopping
     - TensorBoard logging

6. `main.py`
   - Entry point for training and evaluation
   - Sets up hyperparameters and model configuration
   - Initializes training components
   - Handles device selection (CPU/CUDA/MPS)
   - Generates visualization outputs

## Setup and Usage

1. First, generate the sample data:

```bash
python create_sample_data.py
```

This will create:

- `data/train.json`: Training dataset
- `data/validate.json`: Validation dataset

2. Train and evaluate the model:

```bash
python main.py
```

This will:

- Train the attention-based MIL model
- Generate visualizations in `figs/` directory
- Log training metrics to TensorBoard

## Model Architecture

The Attention MIL model processes data in the following steps:

1. Instance-level feature extraction
2. Attention weight computation for each instance
3. Weighted pooling of instance features
4. Final classification

## Output Files

The training process generates:

- Confusion matrix plot (`figs/confusion_matrix.png`)
- ROC curve plot (`figs/ROC_curve.png`)
- Attention visualization plots (`figs/attentions/`)
- TensorBoard logs in `runs/mil_experiment/`

## Requirements

- PyTorch
- NumPy
- Pydantic
- TensorBoard
- Matplotlib
- tqdm

## Data Format

The project expects data in JSON format with the following structure:

```json
[
  {
    "name": "patient_id",
    "label": 0 or 1,
    "features": [[float, ...], ...]  // List of feature vectors
  },
  ...
]
```
