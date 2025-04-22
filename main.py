import torch
import torch.nn as nn
import torch.optim as optim
import os

from model import AttentionMIL
from dataloader import get_train_loader, get_val_loader
from training_manager import TrainingManager
from early_stopping import EarlyStopping
from vis_utils import visualize_model_performance, visualize_attention_weights

# Use MPS if on Mac M1/M2, CUDA if available, else CPU
device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)


def main():
    # Hyperparameters
    input_dim = 192  # Dimension of each instance's features
    learning_rate = 0.001
    num_epochs = 50  # Increased epochs since MIL might need more training
    batch_size = 16  # Reduced batch size due to bag structure
    max_instances = 20  # Maximum number of instances per patient

    # Initialize model, criterion, and optimizer
    model = AttentionMIL(input_dim=input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping with more patience for MIL
    early_stopping = EarlyStopping(
        patience=10, min_delta=1e-4  # Increased patience for MIL
    )

    # Get data loaders with max_instances parameter
    train_loader = get_train_loader(
        batch_size=batch_size, max_instances=max_instances
    )
    val_loader = get_val_loader(
        batch_size=batch_size, max_instances=max_instances
    )

    # Initialize training manager (removed max_instances parameter)
    trainer = TrainingManager(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        early_stopping=early_stopping,
        logdir="runs/mil_experiment",
        monitor="val_loss",
        device=device,
    )

    # Train the model
    print(f"Training on device: {device}")
    print(f"Input dimension: {input_dim}")
    print(f"Max instances per patient: {max_instances}")

    trainer.train(num_epochs, train_loader, val_loader)

    # Load best model
    model.load_state_dict(trainer.get_best_model())

    # Create output directory
    os.makedirs("./figs/attentions", exist_ok=True)

    # Visualize model performance
    cm_fig, roc_fig = visualize_model_performance(
        model=model,
        data_loader=val_loader,
        device=device,
    )
    atten_figs = visualize_attention_weights(
        model=model,
        data_loader=val_loader,
        device=device,
        num_examples=5,
    )

    # Save figures
    cm_fig.savefig("./figs/confusion_matrix.png")
    roc_fig.savefig("./figs/ROC_curve.png")
    for name, fig in atten_figs:
        fig.savefig(f"./figs/attentions/{name}.png")


if __name__ == "__main__":
    main()
