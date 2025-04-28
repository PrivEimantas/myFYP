#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2025 Pedro Guijas Bravo.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

"""Simple MLP on PyTorch Lightning for MNIST."""

from typing import Dict, Optional

import lightning as L
import torch
from torchmetrics import Accuracy, Metric, Precision, Recall,F1Score

from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel
from p2pfl.settings import Settings
from p2pfl.utils.seed import set_seed

####
# Example MLP
####


class MLP(L.LightningModule):
    """Multilayer Perceptron (MLP) with configurable parameters."""

    def __init__(
        self,
        input_size: int = 28 * 28,
        hidden_sizes: Optional[list[int]] = None,
        out_channels: int = 10,
        activation: str = "relu",
        metric: type[Metric] = Accuracy,
        lr_rate: float = 0.001,
    ) -> None:
        """Initialize the MLP."""
        super().__init__()
        set_seed(Settings.general.SEED, "pytorch")
        if hidden_sizes is None:
            hidden_sizes = [256, 128]
        self.lr_rate = lr_rate
        if out_channels == 1:
            self.metric = metric(task="binary")
            self.f1score = F1Score(task="binary")
            self.precision = Precision(task="binary")
            self.recall = Recall(task="binary")
        else:
            self.metric = metric(task="multiclass", num_classes=out_channels)
            self.f1score = F1Score(task="multiclass", num_classes=out_channels, average="macro")
            self.precision = Precision(task="multiclass", num_classes=out_channels, average="macro")
            self.recall = Recall(task="multiclass", num_classes=out_channels, average="macro")

        self.layers = torch.nn.ModuleList()

        # Input layer
        self.layers.append(torch.nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(self._get_activation(activation))

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.layers.append(self._get_activation(activation))

        # Output layer
        self.layers.append(torch.nn.Linear(hidden_sizes[-1], out_channels))

    def _get_activation(self, activation_name: str) -> torch.nn.Module:
        if activation_name == "relu":
            return torch.nn.ReLU()
        elif activation_name == "sigmoid":
            return torch.nn.Sigmoid()
        elif activation_name == "tanh":
            return torch.nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP."""
        # Flatten the input
        batch_size, _, _ = x.size()
        x = x.view(batch_size, -1)

        for layer in self.layers:
            x = layer(x)

        x = torch.log_softmax(x, dim=1)
        return x

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.lr_rate)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_id: int) -> torch.Tensor:
        """Training step of the MLP."""
        x = batch["image"].float()
        y = batch["label"]
        loss = torch.nn.functional.cross_entropy(self(x), y)
        self.log("train_loss", loss, prog_bar=False)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_id: int) -> torch.Tensor:
        """Perform validation step for the MLP."""
        raise NotImplementedError("Validation step not implemented")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_id: int) -> torch.Tensor:
        """Test step for the MLP."""
        x = batch["image"].float()
        y = batch["label"]
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(self(x), y)
        out = torch.argmax(logits, dim=1)
        metric = self.metric(out, y)
        test_f1 = self.f1score(out, y)
        test_prec = self.precision(out, y)
        test_rec = self.recall(out, y)
        self.log("test_f1", test_f1, prog_bar=False)
        self.log("test_precision", test_prec, prog_bar=False)
        self.log("test_loss", loss, prog_bar=False)
        self.log("test_recall", test_rec, prog_bar=False)
        self.log("test_metric", metric, prog_bar=False)
        return loss
    
    def on_test_epoch_end(self):
        set_seed(Settings.general.SEED, "pytorch")
        # Compute and log accumulated metric values once at the end of an epoch
        test_acc = self.metric.compute()
        test_f1 = self.f1score.compute()
        test_prec = self.precision.compute()
        test_rec = self.recall.compute()
        self.log("test_accuracy", test_acc, prog_bar=False)
        self.log("test_f1", test_f1, prog_bar=False)
        self.log("test_precision", test_prec, prog_bar=False)
        self.log("test_recall", test_rec, prog_bar=False)


# Export P2PFL model
def model_build_fn(*args, **kwargs) -> LightningModel:
    """Export the model build function."""
    compression = kwargs.pop("compression", None)
    return LightningModel(MLP(*args, **kwargs), compression=compression)
