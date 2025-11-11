import torch
from torchvision import models
import torch.nn as nn
import numpy as np
import json
import os

# Paths
MODEL_PATH = "models/crop_disease_model.pt"
CLASS_PATH = "models/class_names.json"

def load_model():
    """
    Loads the EfficientNet model along with the exact class names 
    from training to avoid class mismatch issues.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    if not os.path.exists(CLASS_PATH):
        raise FileNotFoundError(f"Class names file not found at: {CLASS_PATH}\n"
                                "Run your training script to generate it.")

    # Load class names
    with open(CLASS_PATH, "r") as f:
        class_names = json.load(f)

    num_classes = len(class_names)
    print(f"✅ Loaded {num_classes} classes from class_names.json")

    # Define architecture (must match training)
    model = models.efficientnet_b0(pretrained=False)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)  # type: ignore

    # Load trained weights
    state_dict = torch.load(MODEL_PATH, map_location="cpu")

    # Handle both normal and checkpoint-style saves
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"⚠️ Missing keys: {missing}")
        print(f"⚠️ Unexpected keys: {unexpected}")

    model.eval()
    print("✅ Model loaded successfully and set to eval mode.")
    return model, class_names


def predict_disease(model, image, transform, class_names, threshold=0.5):
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).numpy()[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])

    if confidence < threshold:
        return "Unknown / Not a leaf", confidence

    return class_names[idx], confidence
