import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os, glob
from tqdm import tqdm

# ----------------- Settings -----------------
batch_size = 4
image_size = 160
epochs_head = 5
epochs_finetune = 10
use_amp = True
max_checkpoints = 3  # Keep last 3 checkpoints only

# ----------------- Device -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------- Data Augmentation -----------------
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
])

# ----------------- Dataset -----------------
train_dataset = datasets.ImageFolder('dataset/train', transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=2, pin_memory=True
)

val_loader = None
val_path = 'dataset/val'
if os.path.exists(val_path) and any(os.scandir(val_path)):
    val_dataset = datasets.ImageFolder(val_path, transform=transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

# ----------------- Model -----------------
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, 3)
model = model.to(device)
criterion = nn.CrossEntropyLoss()

# ----------------- Checkpoint Utilities -----------------
def save_checkpoint(model, optimizer, scaler, epoch, stage, best_acc=0.0, is_best=False):
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_name = f"checkpoints/checkpoint_{stage}_epoch{epoch+1}.pth"
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scaler_state': scaler.state_dict(),
        'best_acc': best_acc
    }, checkpoint_name)
    
    # Keep only the last N checkpoints
    checkpoint_files = sorted(glob.glob(f"checkpoints/checkpoint_{stage}_epoch*.pth"),
                              key=os.path.getctime)
    while len(checkpoint_files) > max_checkpoints:
        os.remove(checkpoint_files[0])
        checkpoint_files.pop(0)
    
    # Save best model separately
    if is_best:
        torch.save(model.state_dict(), "best_model.pth")
        print(f"🌟 New best model saved at epoch {epoch+1}")

def load_latest_checkpoint(model, optimizer=None, scaler=None):
    checkpoint_files = glob.glob("checkpoints/checkpoint_*.pth")
    if not checkpoint_files:
        print("ℹ️ No checkpoint found, starting from scratch.")
        return model, optimizer, scaler, 0, 0.0

    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    print(f"🔄 Resuming from {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    if optimizer and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    if scaler and 'scaler_state' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state'])
    start_epoch = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_acc', 0.0)
    return model, optimizer, scaler, start_epoch, best_acc

# ----------------- Stage 1: Train Classifier Head -----------------
for param in model.features.parameters():
    param.requires_grad = False

optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
model, optimizer, scaler, start_epoch, best_acc = load_latest_checkpoint(model, optimizer, scaler)

print("\n--- Stage 1: Training classifier head ---")
for epoch in range(start_epoch, epochs_head):
    model.train()
    loop = tqdm(train_loader, desc=f"Head Epoch {epoch+1}/{epochs_head}")
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=loss.item())

    save_checkpoint(model, optimizer, scaler, epoch, stage='head', best_acc=best_acc)

# ----------------- Stage 2: Fine-tuning Full Model -----------------
for param in model.features.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=0.0001)
model, optimizer, scaler, start_epoch, best_acc = load_latest_checkpoint(model, optimizer, scaler)

print("\n--- Stage 2: Fine-tuning full model ---")
for epoch in range(start_epoch, epochs_finetune):
    model.train()
    loop = tqdm(train_loader, desc=f"Finetune Epoch {epoch+1}/{epochs_finetune}")
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # Validation
    acc = 0
    if val_loader:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        print(f"📊 Validation Accuracy: {acc:.2f}%")

    # Save checkpoint every 2 epochs + best model
    if (epoch + 1) % 2 == 0 or (val_loader and acc > best_acc):
        is_best = val_loader and acc > best_acc
        if is_best:
            best_acc = acc
        save_checkpoint(model, optimizer, scaler, epoch, stage='finetune', best_acc=best_acc, is_best=is_best)

# ----------------- Save Final Model -----------------
torch.save(model.state_dict(), 'garbage_classifier.pth')
print("✅ Final model saved as garbage_classifier.pth")
print(f"🌟 Best validation accuracy achieved: {best_acc:.2f}%")