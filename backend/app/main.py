from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io
import torch # noqa: F401
import torchvision.transforms as T
import json
from .model_loader import load_model, predict_disease

app = FastAPI(title="Crop Disease Detection API")

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # use your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model + disease info
model, class_names = load_model()
with open("app/disease_info.json", "r", encoding="utf-8") as f:
    DISEASE_INFO = json.load(f)

# Preprocessing (must match training)
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class Prediction(BaseModel):
    disease: str
    confidence: float
    solution: str
    leaf_characteristics: dict

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).numpy()[0]
        idx = int(probs.argmax())

    disease = class_names[idx]
    confidence = float(probs[idx])

    # Build probability dictionary
    all_probs = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

    # Lookup info
    info = DISEASE_INFO.get(disease, {
        "solution": "No info available.",
        "characteristics": {}
    })
    if disease == "Unknown" or confidence < 0.5:
        return {
        "disease": "Unknown / Not a leaf",
        "confidence": confidence,
        "solution": "Please upload a clear image of a healthy or diseased leaf.",
        "leaf_characteristics": {}
         }

    return {
        "disease": disease,
        "confidence": confidence,
        "solution": info["solution"],
        "leaf_characteristics": info["characteristics"],
        "all_probs": all_probs  # Added for visualization
    }
