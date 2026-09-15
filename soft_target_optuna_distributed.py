"""This file performs hyperparameter optimization for a GAT model.

The script uses Optuna to search for suitable GNN architecture and training
parameters. Each trial trains a PyTorch Lightning GAT model on processed Grid2Op
graph data and stores the corresponding model checkpoints and configuration.

"""
import os
import json
import optuna
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch.nn as nn
from pathlib import Path

import warnings
from gnn.gnn_models import GAT
import random
from gnn.torch_geometric_datasets import Grid2opGraphDataset, Grid2opGraphDatasetProcessed, Grid2opGraphDatasetGlobal
from lightsim2grid import LightSimBackend
from grid2op.Observation import CompleteObservation
import grid2op

# Directory holding the Optuna study database and per-trial checkpoints.
# Override with the STUDY_DIR environment variable.
STUDY_DIR = os.environ.get("STUDY_DIR", "./saved_models/gat_hypertun")


def objective(trial):
    """Train and evaluate one Optuna trial.

    This function defines the hyperparameter search space for the GAT model,
    creates the corresponding model configuration, loads the processed training
    and validation datasets, and trains the model with PyTorch Lightning. The
    validation loss of the trained model is returned as the optimization target.

    Args:
        trial: Optuna trial object used to sample hyperparameters.

    Returns:
        Validation loss of the trained model for the current trial.

    """

    # Dataset paths
    dataset_path = os.environ.get("DATASET_PATH", "./data/processed")
    dataset_name_train = "soft_target_data_train_processed_merged_norm.pt"
    dataset_name_val = "soft_target_data_val_processed_merged_norm.pt"

    # Hyperparameter space
    learning_rate = trial.suggest_float("lr", 1e-8, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
    
    # GAT layers
    num_gat_layers = trial.suggest_int("num_gat_layers", 2, 8)  # Tune number of GAT layers
    gat_layers = []
    for i in range(num_gat_layers):
        out_channels = trial.suggest_categorical(f"gat_out_channels_{i}", [16, 32, 64, 128])
        heads = trial.suggest_categorical(f"gat_heads_{i}", [1, 2, 4, 8])
        dropout = trial.suggest_float(f"gat_dropout_{i}", 0.0, 0.5)
        concat = i < (num_gat_layers - 1)  # Always concat except for the last layer
        gat_layers.append({
            "out_channels": out_channels,
            "heads": heads,
            "dropout": dropout,
            "concat": concat
        })

    pooling_type = trial.suggest_categorical("pooling_type", ["max", "attention"])#, "sag"])
    
    # Linear layers
    num_linear_layers = trial.suggest_int("num_linear_layers", 1, 6)  # Tune number of linear layers
    linear_layers = []
    for i in range(num_linear_layers):
        out_features = trial.suggest_categorical(f"linear_out_features_{i}", [64, 128, 256, 512, 1024, 2048, 4096])
        dropout = trial.suggest_float(f"linear_dropout_{i}", 0.0, 0.5)
        linear_layers.append({"out_features": out_features, "dropout": dropout})
    # Add final output layer
    linear_layers.append({"out_features": 2000})  # Adjust output size if needed
    
    # Scheduler
    scheduler_params = {
        "mode": "min",
        "factor": 0.99,
        "patience": 20,
        "verbose": True,
        "min_lr": 1e-10
    }
    batch_size = 64
    # Model configuration
    config = {
        "lr_scheduler": {"type": "ReduceLROnPlateau", "params": scheduler_params},
        "kwargs_algorithm": {"lr": learning_rate, "weight_decay": weight_decay},
        "in_channels": 27,
        "gat_layers": gat_layers,
        "linear_layers": linear_layers,
        "dropout": 0.0, #trial.suggest_uniform("global_dropout", 0.0, 0.5),
        "pooling_type": pooling_type,
        #"sag_pooling_ratio": trial.suggest_uniform("sag_pooling_ratio", 0.0, 0.5),
        "gat_activation": "elu",
        "linear_activation": "relu",
        "batch_size": batch_size, #trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        "cluster_assignment": None,
        "out_channels_lin": 2000
    }
        
    config["loss"] = nn.KLDivLoss(reduction='batchmean')
    # Load datasets
    backend = LightSimBackend()
    env = grid2op.make("l2rpn_icaps_2021_large", backend=backend, observation_class=CompleteObservation)

    
    d_train = Grid2opGraphDatasetProcessed(root=dataset_path, dataset_name=dataset_name_train, env=env,
                                           split="train",
                                           include_disconnected_lines=True)

    d_val = Grid2opGraphDatasetProcessed(root=dataset_path, dataset_name=dataset_name_val, env=env, split="val",
                                         include_disconnected_lines=True)

    train_dataloader = DataLoader(d_train, batch_size=batch_size, shuffle=True, num_workers=4, drop_last = True)
    val_dataloader = DataLoader(d_val, batch_size=batch_size, shuffle=False, num_workers=4, drop_last = True)

    # Initialize model
    model = GAT(config)


    # Create a folder for this specific trial
    trial_id = trial.number
    trial_dir = os.path.join(STUDY_DIR, f"trial_{trial_id}")
    os.makedirs(trial_dir, exist_ok=True)

        # Save trial config
    config_path = os.path.join(trial_dir, "config.json")
    config.pop("loss")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    # Callbacks for saving best model and early stopping
    checkpoint_callback = ModelCheckpoint(
        dirpath=trial_dir,
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
        #accelerator="cpu",  # Use the CPU accelerator
        #devices=8,          # Number of CPU processes/cores to use
        #strategy="ddp",  # Use distributed data parallel on CPU
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=True  # Disable logging for faster optimization
    )

    # Train model
    trainer.fit(model, train_dataloader, val_dataloader)


    # Return validation loss
    return trainer.callback_metrics["val_loss"].item()


if __name__ == "__main__":
    """Run the Optuna hyperparameter optimization study.

    This block creates or loads an Optuna study, defines a median pruner, and
    optimizes the objective function for a fixed number of trials.

    """

    os.makedirs(STUDY_DIR, exist_ok=True)

    # Create Optuna study
    pruner = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=50)

    study = optuna.create_study(
        study_name="lightning_gat",
        direction="minimize",  # Or "maximize", depending on your goal
        storage=f"sqlite:///{os.path.join(STUDY_DIR, 'study.db')}",  # Using SQLite for file storage
        load_if_exists=True,
        pruner=pruner

    )
    study.optimize(objective, n_trials=100)
    
    print("Best trial:", study.best_trial.params)
