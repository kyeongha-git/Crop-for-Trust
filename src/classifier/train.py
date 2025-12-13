import os
import shutil

import torch
import torch.nn as nn
from tqdm import tqdm


# ==============================================================
# Training Loop
# ==============================================================
def train_one_epoch(
    model: nn.Module, dataloader, criterion, optimizer, device: torch.device
):
    """Run one training epoch and return average loss and accuracy."""
    model.train()
    total_loss, total_correct = 0.0, 0

    for images, labels in tqdm(dataloader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        if labels.ndim == 1:
            labels = labels.unsqueeze(1)
        labels = labels.float()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        total_correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    avg_acc = total_correct / len(dataloader.dataset)
    return avg_loss, avg_acc


# ==============================================================
# Validation Loop
# ==============================================================
def validate(model: nn.Module, dataloader, criterion, device: torch.device):
    """Evaluate the model on validation data."""
    model.eval()
    total_loss, total_correct = 0.0, 0
    
    # [Safety] 데이터셋이 비어있으면 0.0 반환하고 종료
    if len(dataloader.dataset) == 0:
        return 0.0, 0.0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Valid", leave=False):
            images, labels = images.to(device), labels.to(device)
            if labels.ndim == 1:
                labels = labels.unsqueeze(1)
            labels = labels.float()

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            total_correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    avg_acc = total_correct / len(dataloader.dataset)
    return avg_loss, avg_acc


# ==============================================================
# Full Training Pipeline
# ==============================================================
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
    wandb_run=None,
):
    """
    Execute the full training + validation loop.

    - Saves last checkpoint every epoch
    - Updates best checkpoint when validation accuracy improves
    - Copies final best model to save_path after training
    """
    best_acc = 0.0

    check_dir = os.path.dirname(check_path)
    os.makedirs(check_dir, exist_ok=True)

    last_ckpt = check_path
    best_ckpt = check_path.replace("_last.pt", "_best.pt")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, valid_loader, criterion, device)

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

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_ckpt)
            print(f"Best checkpoint updated: {best_ckpt}")

    if os.path.exists(best_ckpt):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        shutil.copy2(best_ckpt, save_path)
        print(f"Copied final best model to: {save_path}")

    print(f"\nTraining Complete! Best Validation Accuracy: {best_acc:.4f}")
    return best_acc
