import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import os
import sys
from PIL import Image
from PIL import ImageFile

from collections import Counter
import torch.nn.functional as F

try:
    from PIL import UnidentifiedImageError
except ImportError:
    UnidentifiedImageError = OSError

# ----------------- Device -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------- Model -----------------
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, 3)  # dry, wet, unidentified
model = model.to(device)
model.eval()

# ----------------- Load Final Model -----------------
if os.path.exists("garbage_classifier.pth"):
    print("✅ Loading garbage_classifier.pth")
    model.load_state_dict(torch.load("garbage_classifier.pth", map_location=device))
else:
    raise FileNotFoundError("No trained model found! Run train.py first.")

# ----------------- Preprocessing -----------------
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

labels = ["dry", "wet", "unidentified"]
CONF_THRESHOLD = 0.6  # confidence threshold for dry/wet


# ----------------- Preprocess -----------------
def preprocess(img_pil):
    """Ensure image is RGB PIL → Tensor"""
    img = img_pil.convert("RGB")
    return transform(img).unsqueeze(0).to(device)


# ----------------- Predict -----------------
def predict_label(img_tensor):
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, predicted = torch.max(probs, 1)
        label = labels[predicted.item()]

    if label not in ["dry", "wet"] or conf.item() < CONF_THRESHOLD:
        label = "unidentified"

    return label, conf.item()


# ----------------- Single Image -----------------
def classify_image(img_path, show_debug=True):
    try:
        img = Image.open(img_path)
        img_tensor = preprocess(img)
        label, conf = predict_label(img_tensor)
        if show_debug:
            print(f"✅ {os.path.basename(img_path)} → {label} (confidence: {conf:.2f})")
        return label, conf
    except (UnidentifiedImageError, OSError) as e:
        if show_debug:
            print(f"⚠️ Skipped {img_path} due to error: {e}")
        return "unidentified", 0.0


# ----------------- Folder -----------------
def classify_folder(folder_path):
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not image_files:
        print("❌ No images found in folder:", folder_path)
        return

    print(f"📂 Scanning {len(image_files)} images in {folder_path}...\n")
    predictions = []

    # log file path
    log_path = os.path.join(folder_path, "results_summary.txt")
    with open(log_path, "w") as log_file:
        log_file.write(f"📂 Results for {folder_path}\n\n")

        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            label, conf = classify_image(img_path, show_debug=True)
            predictions.append(label)
            log_file.write(f"{img_file} → {label} (confidence: {conf:.2f})\n")

        # ----------------- Debugging Summary -----------------
        summary = Counter(predictions)
        log_file.write("\n📊 Folder Summary:\n")
        print("\n📊 Folder Summary:")
        for label in labels:
            log_file.write(f"{label.capitalize()}: {summary.get(label, 0)}\n")
            print(f"{label.capitalize()}: {summary.get(label, 0)}")

    print(f"\n📝 Results saved to {log_path}")


# ----------------- Webcam -----------------
def classify_webcam():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        # Convert frame → PIL
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = preprocess(img_pil)
        label, conf = predict_label(img_tensor)

        # Show label on video
        cv2.putText(frame, f"{label} ({conf:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Garbage Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ----------------- Main -----------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        if os.path.isdir(input_path):
            classify_folder(input_path)
        elif os.path.isfile(input_path):
            classify_image(input_path, show_debug=True)
        else:
            print("❌ Invalid path:", input_path)
    else:
        classify_webcam()