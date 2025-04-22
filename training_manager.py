import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from typing import Literal
from early_stopping import EarlyStopping


class TrainingManager:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        early_stopping: EarlyStopping,
        logdir: str,
        monitor: Literal["val_loss"] | Literal["val_acc"],
        device: str,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.writer = SummaryWriter(logdir)
        self.model.to(device)

        # Early stopping setup
        self.monitor = monitor
        self.early_stopping = early_stopping

    def train(
        self, num_epochs: int, train_loader: DataLoader, val_loader: DataLoader
    ):
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc = self.validate(val_loader, epoch)

            # Check early stopping
            monitor_value = val_loss if self.monitor == "val_loss" else val_acc
            if self.early_stopping(monitor_value, self.model):
                print(f"\nEarly stopping triggered after epoch {epoch}")
                break

    def train_epoch(
        self, train_loader: DataLoader, epoch: int
    ) -> tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (labels, masks, features) in enumerate(pbar):
            # Move tensors to device
            labels = labels.float().to(self.device)
            masks = masks.to(self.device)
            features = features.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with both features and masks
            predictions, attention_weights = self.model(features, masks)
            predictions = predictions.squeeze()
            loss = self.criterion(predictions, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            predicted = (predictions > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Log attention weights periodically
            if batch_idx % 100 == 0:
                # Only log attention weights for valid instances
                valid_attention = attention_weights[masks.bool()].detach()
                self.writer.add_histogram(
                    f"attention_weights_epoch_{epoch}",
                    valid_attention,
                    batch_idx,
                )

            pbar.set_postfix(
                {
                    "loss": running_loss / (batch_idx + 1),
                    "acc": 100.0 * correct / total,
                }
            )

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total

        # Log metrics to tensorboard
        self.writer.add_scalar("Training Loss", epoch_loss, epoch)
        self.writer.add_scalar("Training Accuracy", epoch_acc, epoch)

        return epoch_loss, epoch_acc

    def validate(
        self, val_loader: DataLoader, epoch: int
    ) -> tuple[float, float]:
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for labels, masks, features in val_loader:
                # Move tensors to device
                labels = labels.float().to(self.device)
                masks = masks.to(self.device)
                features = features.to(self.device)

                # Forward pass with both features and masks
                predictions, attention_weights = self.model(features, masks)
                predictions = predictions.squeeze()
                loss = self.criterion(predictions, labels)

                val_loss += loss.item()
                predicted = (predictions > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * correct / total

        # Log validation metrics to tensorboard
        self.writer.add_scalar("Validation Loss", val_loss, epoch)
        self.writer.add_scalar("Validation Accuracy", val_acc, epoch)

        return val_loss, val_acc

    def get_best_model(self) -> dict:
        if self.early_stopping.best_model is None:
            raise Exception("Model not trained!")

        return self.early_stopping.best_model
