import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import os
import sys
from PIL import Image
from collections import Counter
import csv
import time

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
CONF_THRESHOLD = 0.6

# ----------------- Preprocess -----------------
def preprocess(img_pil):
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
    except OSError as e:
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

    log_txt_path = os.path.join(folder_path, "results_summary.txt")
    log_csv_path = os.path.join(folder_path, "results_summary.csv")

    with open(log_txt_path, "w") as log_file_txt, \
         open(log_csv_path, mode="w", newline="") as log_file_csv:

        writer = csv.writer(log_file_csv)
        writer.writerow(["Image", "Label", "Confidence"])
        log_file_txt.write(f"📂 Results for {folder_path}\n\n")

        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            label, conf = classify_image(img_path, show_debug=True)
            predictions.append(label)
            log_file_txt.write(f"{img_file} → {label} (confidence: {conf:.2f})\n")
            writer.writerow([img_file, label, f"{conf:.2f}"])

        summary = Counter(predictions)
        log_file_txt.write("\n📊 Folder Summary:\n")
        print("\n📊 Folder Summary:")
        for label in labels:
            log_file_txt.write(f"{label.capitalize()}: {summary.get(label, 0)}\n")
            print(f"{label.capitalize()}: {summary.get(label, 0)}")

    print(f"\n📝 Results saved to {log_txt_path} and {log_csv_path}")

# ----------------- Webcam -----------------
def classify_webcam(save_frames=False, confidence_threshold=CONF_THRESHOLD):
    cap = cv2.VideoCapture(0)
    log_file = "webcam_log.csv"
    frames_dir = "frames"
    if save_frames:
        os.makedirs(frames_dir, exist_ok=True)

    prediction_counts = Counter()
    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Label", "Confidence", "FrameFile"])

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break

            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tensor = preprocess(img_pil)
            label, conf = predict_label(img_tensor)
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            frame_file = ""

            if save_frames and conf >= confidence_threshold:
                frame_file = os.path.join(frames_dir, f"{timestamp}_{label}_{frame_count}.jpg")
                cv2.imwrite(frame_file, frame)

            if conf >= confidence_threshold:
                writer.writerow([timestamp, label, f"{conf:.2f}", frame_file])
                prediction_counts[label] += 1

            cv2.putText(frame, f"{label} ({conf:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            y_offset = 60
            for lbl in labels:
                cv2.putText(frame, f"{lbl}: {prediction_counts[lbl]}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                y_offset += 30

            cv2.imshow("Garbage Classification", frame)
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n📝 Webcam predictions saved to {log_file}")
    if save_frames:
        print(f"🖼️ Frames saved to {frames_dir}/")

# ----------------- Main -----------------
if __name__ == "__main__":
    print("\nChoose mode:\n1 - Webcam\n2 - Single Image\n3 - Folder")
    choice = input("Enter choice (1/2/3): ")

    if choice == "1":
        classify_webcam(save_frames=True, confidence_threshold=CONF_THRESHOLD)
    elif choice == "2":
        path = input("Enter image path: ").strip()
        if os.path.isfile(path):
            classify_image(path, show_debug=True)
        else:
            print("❌ Invalid image path")
    elif choice == "3":
        path = input("Enter folder path: ").strip()
        if os.path.isdir(path):
            classify_folder(path)
        else:
            print("❌ Invalid folder path")
    else:
        print("❌ Invalid choice")