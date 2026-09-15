import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import TransformerConv, GATConv, global_max_pool, global_mean_pool, global_add_pool, TopKPooling, SAGPooling
from torch_geometric.nn.aggr import AttentionalAggregation


from torch.nn import Linear, Sequential, Dropout, functional as F



class GAT(pl.LightningModule):
    def __init__(self, config=None, lr=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['config'])  # saves hyperparameters to checkpoint

        self.default_config = {
            "in_channels": 27,
            "gat_layers": [
                {"out_channels": 16, "heads": 4, "dropout": 0.2, "concat": True},
                {"out_channels": 256, "heads": 1, "dropout": 0.2, "concat": False}
            ],
            "linear_layers": [
                {"out_features": 512, "dropout": 0.3},
                {"out_features": 1024, "dropout": 0.3},
                {"out_features": 2030}  # change according to output
            ],
            "dropout": 0.0,
            "pooling_type": "max",  # options: "max", "mean", "add", "attention", "sag"
            "gat_activation": "elu",   # Activation function for GAT layers
            "linear_activation": "relu",  # Activation function for linear layers
            "sag_pooling_ratio": 0.5      # optional ratio for SAGPooling
        }

        # Use provided config or fall back to default
        if config is None:
            config = self.default_config
        self.config = config

        self.global_features_size = config.get("global_features_size", 0)

        # Set up GAT layers
        self.gat_layers = nn.ModuleList()
        in_channels = config["in_channels"]
        # Pooling logic and GATv2Conv support
        pooling_type = config.get("pooling_type", "attention")
        self.pool = None
        for layer_cfg in config["gat_layers"]:
            out_channels = layer_cfg["out_channels"]
            heads = layer_cfg["heads"]
            concat = layer_cfg.get("concat", True)
            dropout = layer_cfg.get("dropout", config["dropout"])
            self.gat_layers.append(
                GATConv(in_channels, out_channels, heads=heads, concat=concat, dropout=dropout)
            )
            in_channels = out_channels * heads if concat else out_channels
        
        # Unified pooling creation
        if pooling_type == "TopKPooling":
            self.pool = TopKPooling(in_channels)
        elif pooling_type == "SAGPooling":
            self.pool = SAGPooling(in_channels)
        elif pooling_type == "attention":
            gate_nn = nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.BatchNorm1d(in_channels),
                nn.ReLU(),
                nn.Linear(in_channels, 1)
            )
            self.pool = AttentionalAggregation(gate_nn)
        elif pooling_type == "sag":
            sag_ratio = config.get("sag_pooling_ratio", 0.5)
            self.pool = SAGPooling(in_channels, ratio=sag_ratio)
        elif pooling_type in ["max", "mean", "add"]:
            self.pool = pooling_type  # Will handle in forward
        else:
            raise ValueError(f"Unsupported pooling type: {pooling_type}")
        self.pooling_type = pooling_type

        # Set up linear layers
        self.linear_layers = nn.ModuleList()
        self.linear_dropouts = []

        # Include global features if provided
        in_channels = in_channels + self.global_features_size
        for layer_cfg in config["linear_layers"]:
            out_features = layer_cfg["out_features"]
            dropout = layer_cfg.get("dropout", config["dropout"])
            layer = Linear(in_channels, out_features)
            self.linear_layers.append(layer)
            self.linear_dropouts.append(dropout)
            in_channels = out_features

        # Define activation functions based on config
        self.gat_activation_func = getattr(F, config["gat_activation"], F.elu)
        self.linear_activation_func = getattr(F, config["linear_activation"], F.relu)

        # Define loss function
        self.criterion = self.config["loss"]
        #self.lr = lr

    def forward(self, data):
        # Extract features and edge indices; include global features if provided.
        global_features = None
        if self.global_features_size == 0:
            x, edge_index = data.x, data.edge_index
        else:
            x, edge_index, global_features = data.x, data.edge_index, data.global_features.float().reshape(-1, self.global_features_size)

        # Apply GAT layers with activation
        for layer in self.gat_layers:
            x = self.gat_activation_func(layer(x, edge_index))

        # Pooling logic (parity with gnn_models.py)
        if self.pooling_type in ["TopKPooling", "SAGPooling", "sag"]:
            x, edge_index, _, batch, _, _ = self.pool(x, data.edge_index, None, data.batch)
            x = global_mean_pool(x, batch)
        elif self.pooling_type == "max":
            x = global_max_pool(x, data.batch)
        elif self.pooling_type == "mean":
            x = global_mean_pool(x, data.batch)
        elif self.pooling_type == "add":
            x = global_add_pool(x, data.batch)
        elif self.pooling_type == "attention":
            x = self.pool(x, data.batch)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")

        if self.global_features_size > 0:
            x = torch.cat([x, global_features], dim=1)

        # Apply linear layers with activation and dropout
        for i in range(len(self.linear_layers)):
            if i == len(self.linear_layers) - 1:
                x = self.linear_layers[i](x)
            else:
                x = self.linear_activation_func(self.linear_layers[i](x))
                x = F.dropout(x, p=self.linear_dropouts[i], training=self.training)

        return F.log_softmax(x, dim=1)


    def training_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = self.criterion(logits, batch.y.reshape(self.config["batch_size"],-1))
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, batch_size=self.config["batch_size"])

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = self.criterion(logits, batch.y.reshape(self.config["batch_size"],-1))
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, batch_size=self.config["batch_size"])

        return loss

    def test_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = self.criterion(logits, batch.y.reshape(self.config["batch_size"],-1))
        self.log('test_loss', loss, on_epoch=True, prog_bar=True, batch_size=self.config["batch_size"])

        return loss

    def configure_optimizers(self):
        algorithm_config = self.config.get("kwargs_algorithm", {"lr": 0.01, "weight_decay": 5e-4})
        optimizer = torch.optim.AdamW(self.parameters(), **algorithm_config)
        if "lr_scheduler" in self.config:
            scheduler_cfg = self.config["lr_scheduler"]
            if scheduler_cfg["type"] == "ReduceLROnPlateau":
                     # Robust against configs written by older Optuna / PyTorch versions
                scheduler_params = dict(scheduler_cfg["params"])
                scheduler_params.pop("verbose", None)
    
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    **scheduler_params
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val_loss"
                    }
                }
        return optimizer



class GraphTransformer(pl.LightningModule):
    def __init__(self, config=None, lr=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['config'])  # saves hyperparameters to checkpoint

        self.default_config = {
            "in_channels": 27,
            "gat_layers": [  # This key now configures the Transformer layers.
                {"out_channels": 16, "heads": 4, "dropout": 0.2, "concat": True},
                {"out_channels": 256, "heads": 1, "dropout": 0.2, "concat": False}
            ],
            "linear_layers": [
                {"out_features": 512, "dropout": 0.3},
                {"out_features": 1024, "dropout": 0.3},
                {"out_features": 4}  # change according to output
            ],
            "dropout": 0.0,
            "pooling_type": "max",
            "gat_activation": "elu",   # Activation function for Transformer layers (renamed from GAT layers)
            "linear_activation": "relu"  # Activation function for linear layers
        }

        # Use provided config or fall back to default
        if config is None:
            config = self.default_config
        self.config = config

        self.global_features_size = config.get("global_features_size", 0)

        # Set up Transformer layers (replacing GAT layers)
        self.transformer_layers = nn.ModuleList()
        in_channels = config["in_channels"]
        for layer_cfg in config["gat_layers"]:
            out_channels = layer_cfg["out_channels"]
            heads = layer_cfg["heads"]
            concat = layer_cfg.get("concat", True)
            dropout = layer_cfg.get("dropout", config["dropout"])
            self.transformer_layers.append(
                TransformerConv(in_channels, out_channels, heads=heads, concat=concat, dropout=dropout)
            )
            in_channels = out_channels * heads if concat else out_channels

        # Set up pooling layer: if using attention pooling, instantiate a GlobalAttention module
        self.pooling_type = config["pooling_type"]
        if self.pooling_type == "attention":
            # The gate network produces a scalar score for each node.
            gate_nn = nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.BatchNorm1d(in_channels),
                nn.ReLU(),
                nn.Linear(in_channels, 1)
            )
            self.att_pool = AttentionalAggregation(gate_nn)

        # Set up linear layers
        self.linear_layers = nn.ModuleList()
        self.linear_dropouts = []

        # add global features size if applicable
        in_channels = in_channels + self.global_features_size
        for layer_cfg in config["linear_layers"]:
            out_features = layer_cfg["out_features"]
            dropout = layer_cfg.get("dropout", config["dropout"])
            layer = Linear(in_channels, out_features)
            self.linear_layers.append(layer)
            self.linear_dropouts.append(dropout)
            in_channels = out_features

        # Define activation functions based on config
        self.gat_activation_func = getattr(F, config["gat_activation"], F.elu)
        self.linear_activation_func = getattr(F, config["linear_activation"], F.relu)

        # Define loss function (adjust if needed)
        self.criterion = self.config["loss"]
        #self.lr = lr

    def forward(self, data):
        # Extract features and edge indices; handle global features if provided.
        global_features = None
        if self.global_features_size == 0:
            x, edge_index = data.x, data.edge_index
        else:
            x, edge_index, global_features = data.x, data.edge_index, data.global_features.float().reshape(-1, self.global_features_size)

        # Apply Transformer layers with specified activation
        for layer in self.transformer_layers:
            x = self.gat_activation_func(layer(x, edge_index))

        # Apply pooling based on type
        if self.pooling_type == "max":
            x = global_max_pool(x, data.batch)
        elif self.pooling_type == "mean":
            x = global_mean_pool(x, data.batch)
        elif self.pooling_type == "add":
            x = global_add_pool(x, data.batch)
        elif self.pooling_type == "attention":
            x = self.att_pool(x, data.batch)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")

        if self.global_features_size > 0:
            x = torch.cat([x, global_features], dim=1)

        # Apply linear layers with specified activation and dropout
        for i in range(len(self.linear_layers)):
            if i == len(self.linear_layers) - 1:
                x = self.linear_layers[i](x)
            else:
                x = self.linear_activation_func(self.linear_layers[i](x))
                x = F.dropout(x, p=self.linear_dropouts[i], training=self.training)

        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = self.criterion(logits, batch.y.reshape(self.config["batch_size"],-1))
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, batch_size=self.config["batch_size"])

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = self.criterion(logits, batch.y.reshape(self.config["batch_size"],-1))
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, batch_size=self.config["batch_size"])

        return loss

    def test_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = self.criterion(logits, batch.y.reshape(self.config["batch_size"],-1))
        self.log('test_loss', loss, on_epoch=True, prog_bar=True, batch_size=self.config["batch_size"])

        return loss

    def configure_optimizers(self):
        algorithm_config = self.config.get("kwargs_algorithm", {"lr": 0.01, "weight_decay": 5e-4})
        optimizer = torch.optim.Adam(self.parameters(), **algorithm_config)
        if "lr_scheduler" in self.config:
            scheduler_cfg = self.config["lr_scheduler"]
            if scheduler_cfg["type"] == "ReduceLROnPlateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_cfg["params"])
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val_loss"
                    }
                }
        return optimizer
