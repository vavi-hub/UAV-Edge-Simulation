import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import numpy as np

# =========================
# Config
# =========================
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = os.environ["DATASET"]  

# =========================
# Transforms
# =========================
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet stats
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# Dataset split (correct way)
# =========================
full_dataset = datasets.ImageFolder(DATA_DIR)

indices = torch.randperm(len(full_dataset))
train_size = int(0.8 * len(full_dataset))

train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_dataset = Subset(
    datasets.ImageFolder(DATA_DIR, transform=train_transform),
    train_indices
)

val_dataset = Subset(
    datasets.ImageFolder(DATA_DIR, transform=val_transform),
    val_indices
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# =========================
# Class weights
# =========================
targets = [label for _, label in datasets.ImageFolder(DATA_DIR)]
class_counts = torch.bincount(torch.tensor(targets))
total = class_counts.sum()

weights = total / (2.0 * class_counts.float())
# pos_weight = weights[0].to(DEVICE)  # Fire class weight
pos_weight = torch.tensor([1]).to(DEVICE)

print("Class weights:", weights)

# =========================
# Model: MobileNetV2
# =========================
model = models.mobilenet_v2(pretrained=True)

# Replace classifier
model.classifier[1] = nn.Linear(1280, 1)

model = model.to(DEVICE)

# =========================
# Freeze early layers (recommended)
# =========================
for param in model.features[:10].parameters():
    param.requires_grad = False

# =========================
# Loss + Optimizer
# =========================
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# =========================
# Training loop
# =========================
def train():
    best_val_acc = 0

    for epoch in range(EPOCHS):
        # -------- TRAIN --------
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss /= total
        train_acc = correct / total

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.float().unsqueeze(1).to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_mobilenet_model.pth")

        # Optional: save all checkpoints
        torch.save(model.state_dict(), f"./models/mobilenetmodel_epoch_{epoch+1}.pth")

    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")


# =========================
# Run
# =========================
if __name__ == "__main__":
    train()