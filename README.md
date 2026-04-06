# Face Detection & Recognition System

![Python](https://img.shields.io/badge/Python-3.10-blue) ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![RealTime](https://img.shields.io/badge/Real--Time-Webcam-blueviolet)

A real-time face detection and recognition system built using OpenCV and a Convolutional Neural Network (CNN) trained on the Labeled Faces in the Wild (LFW) dataset. Supports live webcam recognition and static image input.

---

## Results

| Model | Recognition Accuracy | Inference Speed |
|-------|---------------------|-----------------|
| Haar Cascade only (baseline) | 78% detection | ~30 FPS |
| CNN + OpenCV (Final) | **94% recognition** | ~24 FPS |

Tested across 50 individuals, 200 images per person.

---

## Project Structure

```
face-detection-recognition/
│
├── data/
│   └── lfw_dataset/              # LFW dataset (subset, 50 individuals)
│
├── notebooks/
│   ├── 01_EDA.ipynb              # Dataset exploration & face sample visualization
│   ├── 02_Preprocessing.ipynb    # Face alignment, resizing, augmentation
│   └── 03_CNN_Training.ipynb     # CNN architecture, training & evaluation
│
├── src/
│   ├── detect.py                 # Face detection using OpenCV Haar Cascade
│   ├── recognize.py              # Face recognition using trained CNN
│   └── webcam_live.py            # Real-time webcam inference
│
├── model/
│   └── face_cnn_model.h5         # Saved trained CNN weights
│
├── requirements.txt
└── README.md
```

---

## How It Works

### 1. Face Detection
- Used OpenCV's Haar Cascade classifier to locate faces in each frame
- Cropped and resized detected faces to 128x128 pixels for CNN input

### 2. Data Preprocessing & Augmentation
- Applied horizontal flipping, random rotation (±15°), and brightness shifts
- Normalized pixel values to [0, 1]
- Split: 80% train, 10% validation, 10% test

### 3. CNN Architecture
```
Input (128x128x3)
→ Conv2D(32, 3x3) + ReLU + MaxPooling
→ Conv2D(64, 3x3) + ReLU + MaxPooling
→ Conv2D(128, 3x3) + ReLU + MaxPooling
→ Flatten
→ Dense(256) + Dropout(0.4)
→ Dense(50, Softmax)   ← 50 identity classes
```
- Trained for 20 epochs, batch size 32, Adam optimizer (lr=0.001)
- Used early stopping to prevent overfitting

### 4. Real-Time Recognition
- Webcam captures frames → Haar Cascade detects face → CNN predicts identity
- Displays name label and confidence score as overlay on the frame

---

## Setup & Run

```bash
# Clone the repo
git clone https://github.com/vedantamzare/face-detection-recognition.git
cd face-detection-recognition

# Install dependencies
pip install -r requirements.txt

# Run on a static image
python src/recognize.py --image path/to/image.jpg

# Run real-time webcam recognition
python src/webcam_live.py
```

Press `Q` to quit the webcam window.

---

## Tech Stack

- **Language:** Python 3.10
- **Computer Vision:** OpenCV 4.x
- **Deep Learning:** TensorFlow 2.x, Keras
- **Data:** NumPy, Pandas
- **Visualization:** Matplotlib

---

## Dataset

Used a subset of the **Labeled Faces in the Wild (LFW)** public dataset — 50 individuals, ~200 images per person. Available at: http://vis-www.cs.umass.edu/lfw/

---

## Author

**Vedant Anil Amzare**
- LinkedIn: [linkedin.com/in/vedant-amzare-949a45235](https://linkedin.com/in/vedant-amzare-949a45235)
- Email: vvedantamzare@gmail.com
