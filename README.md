# 🌾 Crop Disease Detection System using Machine Learning

## 🧠 Overview
This project is an **AI-powered crop disease detection system** designed to help **farmers identify plant diseases early** and receive **practical solutions** instantly.  

Farmers can simply **upload an image of a diseased leaf**, and the system automatically:
1. Detects the **type of disease**.
2. Shows the **confidence level**.
3. Provides the **recommended solution** and **leaf characteristics**.
4. Displays **visual insights** through charts and history.

Our goal is to make **smart agriculture accessible**, using modern **Machine Learning**, **Deep Learning**, and **Web Technologies**.

---

## 🚀 Tech Stack

| Layer | Technology | Description |
|:------|:------------|:-------------|
| **Frontend** | **React + Vite + Tailwind CSS** | Fast and modern UI for uploading images and visualizing results. |
| **Backend** | **FastAPI (Python)** | Handles image uploads, runs ML model inference, and returns predictions. |
| **Machine Learning** | **PyTorch** | Framework for model training and inference. |
| **Model Architecture** | **Hybrid CNN + Transformer (EfficientNet + CBAM + Transformer Encoder)** | Combines local feature extraction and global attention for higher accuracy. |
| **Dataset** | **Kaggle: PlantVillage Dataset** | ~50,000 labeled images of healthy and diseased leaves. |
| **Visualization** | **Recharts (React)** | For displaying confidence graphs and prediction reports. |

---

## ⚙️ System Architecture
```
  +-----------------------+
  |  Farmer Uploads Image |
  +-----------+-----------+
              |
              ▼
   +----------+-----------+
   |  Frontend (React)    |
   |  - Upload leaf image |
   |  - Display results   |
   |  - Show confidence   |
   +----------+-----------+
              |
 (HTTP POST /predict)
              |
              ▼
   +----------+-----------+
   | Backend (FastAPI)    |
   |  - Receives image    |
   |  - Preprocesses data |
   |  - Runs ML inference |
   |  - Sends JSON result |
   +----------+-----------+
              |
              ▼
   +----------+-----------+
   | ML Model (PyTorch)   |
   |  - EfficientNet-B0   |
   |  - CBAM Attention    |
   |  - Transformer Layer |
   +----------+-----------+
              |
              ▼
   +-----------------------+
   | Result → Disease, Solution,
   | Confidence Graph, Leaf Info |
   +-----------------------------+
```
---

## 🧬 Machine Learning Model

### 🔹 Model Name: `HybridLeafNet`
**Architecture:**
- **Backbone:** EfficientNet-B0 (for local texture recognition)
- **CBAM Attention Module:** Focuses on the most relevant leaf regions (spots, rings, color changes)
- **Transformer Encoder:** Captures global relationships between leaf features
- **Classifier:** Fully connected layer → 16 output classes (15 diseases + Unknown)

### 🧩 Advantages of the Hybrid Model
| Component | Purpose | Benefit |
|:-----------|:----------|:----------|
| **EfficientNet-B0** | Lightweight CNN backbone | High accuracy with low computation |
| **CBAM Attention** | Spatial + channel attention | Focuses on disease-affected regions |
| **Transformer Encoder** | Global feature understanding | Distinguishes similar diseases |
| **Unknown Class** | For non-leaf or unclear images | Avoids false predictions |

---

## 📊 Dataset Details

- **Dataset Used:** [PlantVillage (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Total Classes:** 16  
  (15 disease classes + 1 “Unknown”)
- **Max Images per Class:** 900 (to balance dataset)
- **Train/Validation Split:** 80% / 20%
- **Image Size:** 224×224 px

### Example Classes


---

## 🧠 Training Configuration

| Parameter | Value |
|:-----------|:--------|
| Epochs | 10 |
| Batch Size | 32 |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Loss Function | CrossEntropyLoss |
| Validation Accuracy | ~96% |
| Device | GPU / CUDA supported |

---

## 🌿 Frontend Features (React + Tailwind)

✅ **Upload & Preview:** Users can upload and preview leaf images.  
✅ **Confidence Graph:** Visualizes model confidence for top diseases.  
✅ **Detailed Results:** Displays disease name, solution, and leaf characteristics.  
✅ **Unknown Image Handling:** Warns if image is not a leaf.  
✅ **Recent History:** Shows last 5 predictions with timestamp.  
✅ **Responsive Layout:** Works on desktop and mobile screens.  

---

## 🧩 Backend (FastAPI)

- Accepts uploaded images at `/predict`
- Preprocesses input using PyTorch transforms
- Runs the trained model for inference
- Returns JSON output:

```json
{
  "disease": "Tomato_Early_blight",
  "confidence": 0.92,
  "solution": "Apply fungicide containing chlorothalonil or mancozeb.",
  "leaf_characteristics": {
    "color": "brownish yellow",
    "spots": "concentric rings",
    "texture": "dry and brittle"
  }
}
```

## 📈 Example Output (Frontend)

- ✅ Added Unknown Class to handle invalid or unclear images
- ✅ Introduced Hybrid CNN + Transformer Model for finer pattern recognition
- ✅ Balanced dataset to avoid bias
- ✅ Added interactive visualization for transparency
- ✅ Built an end-to-end pipeline from model → FastAPI → React UI


## 💡 Future Improvements

- 📱 Add live camera capture (for real-time diagnosis on mobile)

- ☁️ Deploy on cloud (AWS / Render / HuggingFace Spaces)

- 🌤️ Integrate local weather data for disease prediction

- 🤖 Experiment with Vision Transformers (ViT)

- 🧾 Add regional language support for farmers

## 📦 Folder Structure

```
backend/
 ├── train_model.py          # Model training script
 ├── hybrid_model.py         # HybridLeafNet architecture
 ├── model_loader.py         # Loads model for FastAPI
 ├── main.py                 # FastAPI backend server
 ├── models/
 │    └── crop_disease_model.pt
 └── data/
      └── train/
           ├── Tomato_Early_blight/
           ├── Potato_Late_blight/
           └── Unknown/
frontend/
 ├── src/
 │    └── App.jsx  # React + Tailwind 
 ├── public/
 │    └── index.html
 └── package.json
```

## 🏆 Results

| Metric                 | Value             |
| :--------------------- | :---------------- |
| Validation Accuracy    | **96.1%**         |
| Average Confidence     | **91%**           |
| Early Blight Precision | **94.8%**         |
| Late Blight Precision  | **95.3%**         |
| Inference Time         | **0.5 sec/image** |


## ✨ Conclusion

This project demonstrates how AI can empower agriculture, enabling farmers to:

- Detect crop diseases early

- Take preventive actions

- Improve yield and reduce losses

By combining Deep Learning, Attention mechanisms, and a clean web interface, this system is both accurate and practical for real-world use.

## 🙌 Credits

- Dataset: PlantVillage (Kaggle)

- Model Base: EfficientNet-B0 (PyTorch)

- Instructor / Guide: Dr. Utkarsh Mahadeo Khaire

