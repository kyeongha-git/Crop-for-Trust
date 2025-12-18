import os
import shutil

import torch
import torch.nn as nn
from tqdm import tqdm


# Utility
def _compute_accuracy(outputs, labels, num_classes: int):
    """
    Compute accuracy for binary or multi-class classification.
    """
    if num_classes == 1:
        # Binary classification
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct = (preds == labels).sum().item()
        total = labels.numel()
    else:
        # Multi-class classification
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == labels).sum().item()
        total = labels.size(0)

    return correct, total


# Training Loop
def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    optimizer,
    device: torch.device,
    num_classes: int,
):
    """Run one training epoch and return average loss and accuracy."""
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for images, labels in tqdm(dataloader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        if num_classes == 1:
            if labels.ndim == 1:
                labels = labels.unsqueeze(1)
            labels = labels.float()
        else:
            labels = labels.long()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct, total = _compute_accuracy(outputs, labels, num_classes)
        total_correct += correct
        total_samples += total

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


# Validation Loop
def validate(
    model: nn.Module,
    dataloader,
    criterion,
    device: torch.device,
    num_classes: int,
):
    """Evaluate the model on validation data."""
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    if len(dataloader.dataset) == 0:
        return 0.0, 0.0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Valid", leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            if num_classes == 1:
                if labels.ndim == 1:
                    labels = labels.unsqueeze(1)
                labels = labels.float()
            else:
                labels = labels.long()

            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            correct, total = _compute_accuracy(outputs, labels, num_classes)
            total_correct += correct
            total_samples += total

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


# Full Training Pipeline
def train_model(
    model: nn.Module,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    device: torch.device,
    epochs: int,
    save_path: str,
    check_path: str,
    num_classes: int,
    wandb_run=None,
):
    """
    Execute the full training + validation loop.

    - Supports binary and multi-class classification
    - Saves last & best checkpoints
    """
    best_acc = 0.0

    check_dir = os.path.dirname(check_path)
    os.makedirs(check_dir, exist_ok=True)

    last_ckpt = check_path
    best_ckpt = check_path.replace("_last.pt", "_best.pt")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            num_classes,
        )

        val_loss, val_acc = validate(
            model,
            valid_loader,
            criterion,
            device,
            num_classes,
        )

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Valid Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "best_val_acc": best_acc,
                }
            )

        torch.save(model.state_dict(), last_ckpt)

        if val_acc > best_acc or epoch == 0:
            best_acc = max(best_acc, val_acc)
            torch.save(model.state_dict(), best_ckpt)
            print(f"Best checkpoint updated: {best_ckpt}")

    if not os.path.exists(best_ckpt) and os.path.exists(last_ckpt):
        print("[WARN] No best checkpoint found. Using last checkpoint instead.")
        shutil.copy2(last_ckpt, best_ckpt)

    if os.path.exists(best_ckpt):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        shutil.copy2(best_ckpt, save_path)
        print(f"Copied final best model to: {save_path}")
    else:
        print(f"[ERROR] Failed to save model to {save_path}")

    print(f"\nTraining Complete! Best Validation Accuracy: {best_acc:.4f}")
    return best_acc
