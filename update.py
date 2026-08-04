import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ----------------- 1. Device -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------- 2. Image transforms -----------------
transform = transforms.Compose([
    transforms.Resize((160, 160)),   # resize to 160x160
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------------- 3. Datasets & loaders -----------------
train_dataset = datasets.ImageFolder(root="dataset/train", transform=transform)
val_dataset   = datasets.ImageFolder(root="dataset/val",   transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=4, shuffle=False)

print("Classes:", train_dataset.classes)
num_classes = len(train_dataset.classes)

# ----------------- 4. Model (MobileNetV2) -----------------
model = models.mobilenet_v2(weights=None)  # use same arch as training
model.classifier[1] = nn.Linear(model.last_channel, num_classes)  # replace classifier

# Load previous trained weights
model.load_state_dict(torch.load("Garbage_classification.pth"))
model = model.to(device)
model.train()

# ----------------- 5. Optimizer & loss -----------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # small LR for fine-tune
criterion = nn.CrossEntropyLoss()

# ----------------- 6. Training + Validation -----------------
epochs = 3
for epoch in range(epochs):
    # ---- Training ----
    running_loss = 0.0
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    # ---- Validation ----
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")

# ----------------- 7. Save updated model -----------------
torch.save(model.state_dict(), "Garbage_classification_updated.pth")
print("✅ Updated model saved as Garbage_classification_updated.pth")