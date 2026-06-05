import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# =========================
# Config
# =========================
MODEL_PATH = "best_mobilenet_model.pth"
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔥 Set this from your threshold tuning
FIRE_THRESHOLD = 0.4   # replace with your optimal value

# =========================
# Model
# =========================
def load_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(1280, 1)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model

# =========================
# Transform (must match training)
# =========================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# Predict single image
# =========================
def predict_image(model, image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img)
        prob = torch.sigmoid(output).item()

    is_fire = prob > FIRE_THRESHOLD

    print(f"\nImage: {image_path}")
    print(f"Fire Confidence: {prob:.4f}")

    if is_fire:
        print("🔥 FIRE DETECTED")
    else:
        print("🌿 NO FIRE")

    return prob, is_fire

# =========================
# Predict folder (batch)
# =========================
import os

def predict_folder(model, folder_path):
    results = []

    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(folder_path, file)
            prob, is_fire = predict_image(model, path)
            results.append((file, prob, is_fire))

    return results

# =========================
# Run
# =========================
if __name__ == "__main__":
    model = load_model()

    # 🔹 Single image
    # predict_image(model, "resized_test_nofire_frame159.jpg")

    # 🔹 Folder (optional)
    predict_folder(model, "./frames/Testing/Test/No_Fire")