"""This file trains a GNN model for soft-label based power grid topology control.

The script loads processed Grid2Op graph datasets, initializes a GAT model from a
saved configuration, and trains the model using PyTorch Lightning.

"""
import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import grid2op
import pytorch_lightning as pl
import torch.nn as nn
from grid2op.Observation import CompleteObservation
from lightsim2grid import LightSimBackend
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch_geometric.loader import DataLoader

from gnn.gnn_models import GAT
from gnn.torch_geometric_datasets import Grid2opGraphDatasetProcessed

# Project root: every default path below is relative to it.
PROJECT_ROOT = Path(__file__).resolve().parent

# Directory holding the processed graph datasets produced by Data_Processing.ipynb.
# Override with the DATASET_PATH environment variable.
DATASET_PATH = Path(os.environ.get("DATASET_PATH", PROJECT_ROOT / "data" / "processed"))

# Model configuration to train. Defaults to the architecture of the released
# best model; override with the CONFIG_PATH environment variable.
CONFIG_PATH = Path(
    os.environ.get("CONFIG_PATH", PROJECT_ROOT / "data" / "best_model" / "config.json")
)

# Where checkpoints and the resolved config are written.
# Override with the OUTPUT_PATH environment variable.
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", PROJECT_ROOT / "saved_models" / "gat"))

ENV_NAME = os.environ.get("GRID2OP_ENV", "l2rpn_icaps_2021_large")

DATASET_NAME_TRAIN = "soft_target_data_train_processed_merged_norm.pt"
DATASET_NAME_VAL = "soft_target_data_val_processed_merged_norm.pt"


if __name__ == "__main__":
    """Train a GAT model on processed Grid2Op graph data.

    This script loads the training and validation datasets, restores the model
    configuration from a previous trial, initializes the GAT model, and trains it
    with early stopping and model checkpointing.

    """
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    config["loss"] = nn.KLDivLoss(reduction="batchmean")

    # Load datasets
    backend = LightSimBackend()
    env = grid2op.make(ENV_NAME, backend=backend, observation_class=CompleteObservation)

    d_train = Grid2opGraphDatasetProcessed(root=str(DATASET_PATH), dataset_name=DATASET_NAME_TRAIN, env=env,
                                           split="train",
                                           include_disconnected_lines=True)

    d_val = Grid2opGraphDatasetProcessed(root=str(DATASET_PATH), dataset_name=DATASET_NAME_VAL, env=env, split="val",
                                         include_disconnected_lines=True)

    train_dataloader = DataLoader(d_train, batch_size=config["batch_size"], shuffle=True, num_workers=4, drop_last=True)
    val_dataloader = DataLoader(d_val, batch_size=config["batch_size"], shuffle=False, num_workers=4, drop_last=True)

    # Initialize model
    model = GAT(config)

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Save the config actually used for this run next to the checkpoints.
    # The loss is a live object, so it is dropped before serializing.
    config.pop("loss")
    with open(OUTPUT_PATH / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # Callbacks for saving best model and early stopping
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(OUTPUT_PATH),
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best_model-{epoch:02d}-{val_loss:.2f}",
        save_last=True
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=20
    )

    # Define trainer
    trainer = pl.Trainer(
        enable_progress_bar=False,
        max_epochs=1000,
        accelerator="auto",
        devices="auto",
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=True
    )

    # Train model
    trainer.fit(model, train_dataloader, val_dataloader)
