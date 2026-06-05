import torch
import torch.nn as nn
from torchvision import models

# Load model
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(1280, 1)

model.load_state_dict(torch.load("best_mobilenet_model1.pth", map_location="cpu"))
model.eval()

# Dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Export
torch.onnx.export(
    model,
    dummy_input,
    "fire_model.onnx",
    opset_version=12,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch_size"}}
)

print("Exported to ONNX")