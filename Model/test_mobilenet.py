import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from torchvision import models
import os

# =========================
# Config
# =========================
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_DIR = os.environ.get("TESTSET")  #test dataset
print(f"Test Directory: {TEST_DIR}")
MODEL_PATH = "best_mobilenet_model1.pth"

# =========================
# Transforms (same as validation)
# =========================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# Dataset
# =========================
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Classes:", test_dataset.classes)

# =========================
# Model (same as training)
# =========================
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(1280, 1)

model = model.to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# =========================
# Load model
# =========================


# =========================
# Evaluation
# =========================
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)

        outputs = model(images)
        probs = torch.sigmoid(outputs)

        preds = (probs > 0.3).int().cpu().numpy()

        y_true.extend(labels.numpy())
        y_pred.extend(preds.flatten())

# =========================
# Metrics
# =========================
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=test_dataset.classes))

# =========================
# Optional: accuracy
# =========================
accuracy = np.mean(np.array(y_true) == np.array(y_pred))
print(f"\nAccuracy: {accuracy:.4f}")