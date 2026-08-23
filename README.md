# 🤟 Real-Time Sign Language Translator

> A client-side, real-time sign language translation web application: **Live Camera → Hand Landmark Detection → ASL Letter Recognition → Live Captions → Text-to-Speech**.

---

## 📌 Overview

This project provides an end-to-end pipeline for real-time American Sign Language (ASL) recognition in the browser. It detects hand landmarks directly from the user's camera feed using **MediaPipe Hands**, runs real-time gesture classification using an optimized **TensorFlow.js** neural network, and translates signs into captions with voice synthesis via the **Web Speech API**.

The model was trained on the **Kaggle ASL Alphabet dataset** (87,000 images), achieving **98.30% test accuracy** on 28 sign classes (`A`–`Z`, `del`, `space`).

---

## 🛠️ Tech Stack

### Python (Data Pipeline & Training)
- **OpenCV (`cv2`)**: Image ingestion and preprocessing
- **MediaPipe (`mp.tasks.vision`)**: Batch hand landmark extraction (21 3D coordinates = 63 features)
- **Scikit-Learn**: Dataset validation, stratified train/test splitting, and RandomForest baseline
- **TensorFlow / Keras (`tf_keras`)**: Lightweight Sequential Neural Network training and export
- **TensorFlow.js Converter**: HDF5/Keras to TF.js JSON model and binary weight shard serialization

### JavaScript (Client-Side Web App)
- **MediaPipe Hands JS (CDN)**: Real-time 21-landmark tracking & glowing skeleton canvas overlay
- **TensorFlow.js (CDN)**: In-browser GPU/WebGL-accelerated neural network inference (~200ms latency)
- **Web Speech API**: Text-to-speech synthesis for vocalizing translated words and sentences
- **Vanilla HTML5 / CSS3 / ES6 JavaScript**: Zero-build, dependency-free mobile-first UI

---

## 📂 Project Structure

```
sign-translator/
├── python-training/              # Data processing and model training pipeline
│   ├── dataset_raw/              # (Gitignored) Raw Kaggle ASL image folders (A-Z, del, space)
│   ├── dataset/                  # (Gitignored) Extracted landmark CSV files per class
│   ├── process_dataset.py        # High-throughput MediaPipe batch landmark extractor
│   ├── collect_landmarks.py      # Interactive webcam landmark recorder for custom signs
│   ├── train_model.py            # RandomForest baseline trainer & evaluator
│   └── convert_to_tfjs.py        # Keras neural network trainer & TF.js exporter
│
├── web-app/                      # Lightweight browser-based translation app
│   ├── index.html                # App shell, video viewport, HUD, and live prediction cards
│   ├── style.css                 # Dark-mode glassmorphic mobile-first styles
│   ├── app.js                    # Application orchestrator & UI bindings
│   ├── model/                    # Committed TensorFlow.js model artifacts
│   │   ├── model.json            # Model architecture & weight graph definition
│   │   ├── group1-shard1of1.bin  # 71 KB quantized model weight shard
│   │   └── labels.json           # Ordered array of 28 sign classes
│   └── js/
│       ├── camera.js             # Mobile rear/front camera stream manager & permissions
│       ├── landmarks.js          # MediaPipe Hands lifecycle & skeleton canvas overlay
│       ├── classifier.js         # TF.js inference engine & getCurrentPrediction()
│       ├── sentenceBuilder.js    # Gesture accumulation & debounce engine
│       └── speech.js             # Web Speech text-to-speech engine
│
├── .gitignore                    # Excludes heavy datasets, venvs, and cache files
└── README.md                     # Project documentation
```

---

## 🚀 Quick Start (Running the Web App)

The web app is entirely static and runs directly in any modern web browser (desktop or mobile) with **no build tools or npm installs required**.

### 1. Start a Local Server

Open your terminal in the project directory:

```bash
cd web-app
python -m http.server 8080
```

### 2. Open in Browser

- **Desktop**: Visit [`http://localhost:8080`](http://localhost:8080)
- **Mobile (same Wi-Fi network)**: Visit `http://<YOUR_COMPUTER_LOCAL_IP>:8080` (e.g., `http://192.168.1.50:8080`)
  > *Note: For mobile camera access over local HTTP, enable Chrome's `Insecure origins treated as secure` flag at `chrome://flags/#unsafely-treat-insecure-origin-as-secure`.*

---

## 📊 Dataset & Model Performance

- **Dataset**: [Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) (87,000 images across 29 classes).
- **Extracted Landmark Frames**: 63,580 landmark rows saved (73.1% detection rate).
- **Filtered Classes**: 28 classes (`A`–`Z`, `del`, `space`). The `nothing` class was cleanly dropped as MediaPipe correctly finds 0 hands in empty backgrounds.

| Model Architecture | Parameters / Size | Test Accuracy | Test Loss |
|:---|:---:|:---:|:---:|
| **RandomForest Baseline** (200 trees) | ~14 MB (`.pkl`) | 97.84% | — |
| **Keras Sequential NN** (3 Dense Layers) | **71.4 KB** (`.bin`) | **98.30%** | **0.0608** |

---

## 🚦 Current Status & Roadmap

- [x] **Camera & Streaming**: Dual camera support (rear `environment` default + front toggle) with error recovery.
- [x] **Hand Tracking**: Real-time 21-landmark tracking with glowing neon skeleton canvas overlay.
- [x] **Live Classification**: In-browser TensorFlow.js inference with real-time confidence bar and HUD display.
- [ ] **Sentence Builder**: Gesture debouncing, word formation, and backspace handling.
- [ ] **Speech Synthesis**: Text-to-speech playback of translated sentences.

---

## 📄 License

MIT License — free for educational and non-commercial use.
