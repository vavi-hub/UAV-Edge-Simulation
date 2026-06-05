import onnxruntime as ort
import numpy as np
from PIL import Image
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# =========================
# Config
# =========================
IMAGE_SIZE = 224
FIRE_THRESHOLD = 0.4

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = "/app/best_model.onnx"

# =========================
# Normalization (same as training)
# =========================
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# =========================
# Lazy ONNX Session Loader
# =========================
_session = None
_input_name = None

def _load_session():
    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    return session, input_name

def get_session():
    global _session, _input_name
    if _session is None:
        _session, _input_name = _load_session()
    return _session, _input_name

# =========================
# Preprocessing (matches PyTorch transform)
# =========================
def preprocess(image_path):
    img = Image.open(image_path).convert("RGB")

    # Resize + CenterCrop (manual equivalent)
    img = img.resize((256, 256))
    
    left = (256 - IMAGE_SIZE) // 2
    top = (256 - IMAGE_SIZE) // 2
    img = img.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))

    img = np.array(img).astype(np.float32) / 255.0

    # Normalize
    img = (img - mean) / std

    # HWC → CHW
    img = np.transpose(img, (2, 0, 1))

    # Add batch dim
    img = np.expand_dims(img, axis=0)

    return img.astype(np.float32)

# =========================
# Core Inference Function
# =========================
def predict(image_path):
    """
    Args:
        image_path (str)

    Returns:
        dict:
            {
                "prob": float,
                "is_fire": bool
            }
    """
    session, input_name = get_session()

    img = preprocess(image_path)

    logits = session.run(None, {input_name: img})[0]

    # Apply sigmoid (IMPORTANT)
    prob = 1 / (1 + np.exp(-logits))
    prob = float(prob[0][0])

    return {
        "prob": prob,
        "is_fire": prob > FIRE_THRESHOLD
    }