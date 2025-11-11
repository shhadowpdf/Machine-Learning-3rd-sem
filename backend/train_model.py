import os
import random
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from hybrid_model import HybridLeafNet
# Paths
DATA_DIR = "./data/train"
MODEL_SAVE_PATH = "./models/crop_disease_model1.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 1️⃣ Data Transformations
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -------------------------------
# 2️⃣ Load Dataset
# -------------------------------
full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
print(f"Total images before balancing: {len(full_dataset)}")

# -------------------------------
# 3️⃣ Limit each class to max 900 images
# -------------------------------
max_images_per_class = 900
class_indices = {i: [] for i in range(len(full_dataset.classes))}

# Group image indices by class
for idx, (_, label) in enumerate(full_dataset.samples):
    class_indices[label].append(idx)

# Apply the limit per class
balanced_indices = []
for label, indices in class_indices.items():
    if len(indices) > max_images_per_class:
        selected = random.sample(indices, max_images_per_class)
    else:
        selected = indices
    balanced_indices.extend(selected)

# Create a subset dataset
balanced_dataset = Subset(full_dataset, balanced_indices)
print(f"Total images after balancing: {len(balanced_dataset)}")

# -------------------------------
# 4️⃣ Train/Validation Split
# -------------------------------
train_size = int(0.8 * len(balanced_dataset))
val_size = len(balanced_dataset) - train_size
train_ds, val_ds = random_split(balanced_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# -------------------------------
# 5️⃣ Model Setup
# -------------------------------

model = HybridLeafNet(num_classes=len(full_dataset.classes))
model = model.to(DEVICE)0
0

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 10
best_acc = 0

# -------------------------------
# 6️⃣ Training Loop
# -------------------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f}, Val Acc={acc:.2f}%")

    # Save best model
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"✅ Model saved (Accuracy: {best_acc:.2f}%)")

print(f"🎉 Training complete. Best Validation Accuracy: {best_acc:.2f}%")
