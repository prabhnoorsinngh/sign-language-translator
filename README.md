# Real-Time Sign Language Translator

> A client-side, real-time American Sign Language (ASL) recognition and speech synthesis web application: **Live Camera → Hand Landmark Detection → ASL Letter Classification → Sentence Construction → Text-to-Speech**.

---

## Overview

This project provides an end-to-end client-side pipeline for real-time American Sign Language (ASL) fingerspelling recognition in the browser. The system tracks 21 3D hand landmarks from a live camera feed using **MediaPipe**, classifies static gestures across 28 sign classes using an optimized **TensorFlow.js** neural network, accumulates signs into structured sentences using a temporal hold-to-lock confirmation algorithm, and vocalizes the result via the **Web Speech API**.

The model was trained on the **Kaggle ASL Alphabet dataset** (87,000 images), achieving **98.30% test accuracy** on 28 classes (`A`–`Z`, `del`, `space`).

---

## Features

- **Real-Time Video Capture**: Fluid video stream support with front and rear camera toggle capabilities.
- **Live 21-Landmark Skeleton Tracking**: High-precision hand joint extraction and animated canvas overlay powered by MediaPipe.
- **In-Browser ASL Classification**: Client-side inference via TensorFlow.js running 28 target classes (`A`–`Z`, `del`, `space`) with 98.30% test accuracy.
- **Hold-to-Lock Sentence Builder**: Robust gesture debouncing with visual hold-progress confirmation to eliminate sign flicker, transient transitions, and repeated keystroke spam.
- **Sentence Editing Controls**: Dedicated interface controls for Clear, Backspace, Space insertion, and Clipboard export.
- **Text-to-Speech Synthesis**: Native vocalization of assembled sentences and words using the browser's Web Speech API.
- **Zero-Dependency Deployment**: Lightweight static application requiring no build steps, backend servers, or external package managers at runtime.

---

## How to Use

1. **Launch the Application**: Start a local web server (see [Quick Start](#quick-start)) and open the application in a modern desktop or mobile browser.
2. **Grant Camera Permissions**: When prompted, allow camera access. Use the camera toggle switch to choose between front and rear cameras as needed.
3. **Form a Hand Sign**: Position your hand clearly within the camera viewport. The green landmark skeleton will track your hand in real time.
4. **Hold to Lock the Letter**: Maintain the gesture steadily. The confirmation progress ring will charge; once filled (0.8 seconds hold), the letter is locked and appended to the sentence buffer.
5. **Format Your Sentence**: Use the `space` sign (or the on-screen **Space** button) to separate words, and the `del` sign (or **Backspace** button) to correct mistakes.
6. **Vocalize**: Tap **Speak** to listen to the synthesized sentence aloud, or tap **Copy** to export the text to the clipboard.

---

## Tech Stack

### Python (Data Pipeline & Model Training)
- **OpenCV (`cv2`)**: Image ingestion and preprocessing
- **MediaPipe (`mp.tasks.vision`)**: Batch hand landmark extraction (21 3D coordinates = 63 normalized features)
- **Scikit-Learn**: Stratified train/test dataset splitting and RandomForest baseline evaluation
- **TensorFlow / Keras (`tf_keras`)**: Lightweight Sequential Neural Network architecture and training
- **TensorFlow.js Converter**: Serialization of Keras models into web-optimized TF.js JSON graph and binary weight shards

### JavaScript (Client-Side Web Application)
- **MediaPipe Hands JS**: Real-time 21-landmark tracking and canvas rendering
- **TensorFlow.js**: In-browser WebGL/GPU-accelerated neural network inference (~200ms latency)
- **Web Speech API**: In-browser speech synthesis for real-time vocal output
- **Vanilla HTML5 / CSS3 / ES6 JavaScript**: High-performance, mobile-first responsive interface without external framework overhead

---

## Project Structure

```
sign-translator/
├── python-training/              # Data extraction and machine learning pipeline
│   ├── dataset_raw/              # (Gitignored) Raw Kaggle ASL image directories (A-Z, del, space)
│   ├── dataset/                  # (Gitignored) Extracted landmark CSV feature files
│   ├── process_dataset.py        # MediaPipe batch landmark extraction pipeline
│   ├── collect_landmarks.py      # Interactive webcam landmark recorder for custom samples
│   ├── train_model.py            # RandomForest baseline trainer & evaluator
│   └── convert_to_tfjs.py        # Keras neural network trainer & TF.js exporter
│
├── web-app/                      # Client-side web application
│   ├── index.html                # Application shell, video viewport, HUD, and sentence controls
│   ├── style.css                 # Dark-mode responsive design and animations
│   ├── app.js                    # Application orchestrator and event controller
│   ├── model/                    # Committed TensorFlow.js model artifacts
│   │   ├── model.json            # Model architecture & weight graph specification
│   │   ├── group1-shard1of1.bin  # 71 KB quantized model weight shard
│   │   └── labels.json           # Ordered array of 28 sign classes
│   └── js/
│       ├── camera.js             # Mobile/desktop camera stream management
│       ├── landmarks.js          # MediaPipe Hands lifecycle & skeleton canvas overlay
│       ├── classifier.js         # TensorFlow.js inference engine & prediction dispatcher
│       ├── sentenceBuilder.js    # Hold-to-lock gesture accumulator & sentence state manager
│       └── speech.js             # Web Speech text-to-speech synthesis wrapper
│
├── pyproject.toml                # Python environment and type-checking configuration
├── pyrefly.toml                  # Pyrefly language server configuration
├── pyrightconfig.json            # Pyright type-checking configuration
├── .gitignore                    # Excludes heavy datasets, venvs, and cache files
└── README.md                     # Project documentation
```

---

## Quick Start

The web application is entirely static and runs directly in any modern browser with **no build tools or package installations required**.

### 1. Start a Local Server

Open a terminal in the project directory:

```bash
cd web-app
python -m http.server 8080
```

### 2. Open in Browser

- **Desktop**: Navigate to `http://localhost:8080`
- **Mobile (Local Wi-Fi)**: Navigate to `http://<HOST_LOCAL_IP>:8080` (e.g., `http://192.168.1.50:8080`)
  > *Note: For mobile browsers requiring secure origins for camera access over local HTTP, enable Chrome's `Insecure origins treated as secure` setting at `chrome://flags/#unsafely-treat-insecure-origin-as-secure`.*

---

## Dataset & Model Performance

- **Dataset**: [Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) (87,000 images across 29 classes).
- **Extracted Landmark Frames**: 63,580 landmark rows saved (73.1% detection rate).
- **Filtered Classes**: 28 classes (`A`–`Z`, `del`, `space`). The `nothing` class was removed because MediaPipe naturally yields zero detected hands when no hand is in frame.

| Model Architecture | Artifact Format / Size | Test Accuracy | Test Loss |
|:---|:---:|:---:|:---:|
| **RandomForest Baseline** (200 trees) | `.pkl` (~14 MB) | 97.84% | — |
| **Keras Sequential NN** (3 Dense Layers) | **`.bin` (71.4 KB)** | **98.30%** | **0.0608** |

---

## Current Status & Roadmap

- [x] **Camera & Streaming**: Dual camera support (rear `environment` default and front toggle) with fallback error recovery.
- [x] **Hand Tracking**: Real-time 21-landmark tracking with dynamic skeleton canvas rendering.
- [x] **Live Classification**: In-browser TensorFlow.js inference with real-time confidence metrics and HUD display.
- [x] **Sentence Builder**: Hold-to-lock gesture debouncing, letter accumulation, space insertion, and backspace handling.
- [x] **Speech Synthesis**: Real-time text-to-speech vocalization via the Web Speech API.

---

## License

MIT License — Free for educational and non-commercial use.
