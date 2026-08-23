"""
convert_to_tfjs.py
------------------
Trains a small Keras neural network on hand-landmark CSV data collected from
dataset/*.csv and exports it to TensorFlow.js format for browser inference.

Option A Implementation:
  1. Load and combine all CSV files from dataset/
  2. Encode labels and split into 63 features (21 landmarks x,y,z) and one-hot targets
  3. Stratified 80/20 train/test split
  4. Build a Keras Sequential model (63 inputs -> Dense(128, ReLU) -> Dense(64, ReLU) -> Dense(N, Softmax))
  5. Train the neural network with early stopping
  6. Evaluate test accuracy and compare against RandomForest baseline (model.pkl) if present
  7. Export model to TensorFlow.js format (web-app/model/model.json + weight shards)
  8. Copy/save class labels to web-app/model/labels.json

Dependencies:
    pip install tensorflow tf-keras tensorflowjs pandas scikit-learn joblib

Usage:
    cd python-training
    python convert_to_tfjs.py
"""

import json
import os
import shutil
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tf_keras as keras
import tensorflowjs as tfjs

# ---------------------------------------------------------------------------
# Directory and File Paths
# ---------------------------------------------------------------------------
DATASET_DIR     = "dataset"
RF_MODEL_PATH   = "model.pkl"
SAVED_MODEL_DIR = "keras_saved_model"
TFJS_OUTPUT_DIR = os.path.join("..", "web-app", "model")
LABELS_SRC      = "labels.json"
LABELS_DST      = os.path.join(TFJS_OUTPUT_DIR, "labels.json")

# Warning threshold: if Keras accuracy is more than 5% lower than RF baseline
RF_ACCURACY_WARN_DELTA = 0.05


# ===========================================================================
# Step 1: Load and combine all CSV files from dataset/
# ===========================================================================
def load_dataset(dataset_dir: str) -> pd.DataFrame:
    """
    Scans the dataset directory and merges all per-class CSV files.
    Each CSV is expected to have columns: label, x0,y0,z0, ..., x20,y20,z20
    """
    if not os.path.exists(dataset_dir):
        print(f"\n[ERROR] Directory '{dataset_dir}' not found.")
        print("  Please run collect_landmarks.py to create and populate the dataset directory.\n")
        sys.exit(1)

    csv_files = sorted([
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if f.endswith(".csv")
    ])

    if not csv_files:
        print(
            f"\n[ERROR] No CSV files found in '{dataset_dir}'.\n"
            "  Run collect_landmarks.py first to record at least 2 sign classes.\n"
        )
        sys.exit(1)

    frames = []
    for path in csv_files:
        df = pd.read_csv(path)
        if df.empty:
            print(f"  [SKIP] {path} is empty.")
            continue
        frames.append(df)
        print(f"  [LOAD] {path:50s} rows: {len(df)}")

    if not frames:
        print(
            "\n[ERROR] All CSV files in dataset/ are empty.\n"
            "  Run collect_landmarks.py and hold [r] while gesturing to record samples.\n"
        )
        sys.exit(1)

    combined_df = pd.concat(frames, ignore_index=True)
    return combined_df


# ===========================================================================
# Step 2: Extract features, normalize & encode labels
# ===========================================================================
def normalize_landmarks(X: np.ndarray) -> np.ndarray:
    """
    Normalizes hand landmarks to be position- and scale-invariant:
      1. Translates all landmarks relative to the wrist (landmark 0),
         so wrist becomes (0, 0, 0).
      2. Scales all coordinates by the maximum Euclidean distance from
         the wrist to any other landmark (with epsilon protection).
    """
    is_1d = X.ndim == 1
    if is_1d:
        X = X.reshape(1, -1)

    n_samples = X.shape[0]
    # Reshape (N, 63) -> (N, 21, 3)
    coords = X.reshape(n_samples, 21, 3)

    # Wrist is landmark 0
    wrist = coords[:, 0:1, :]  # shape (N, 1, 3)
    rel_coords = coords - wrist  # shape (N, 21, 3)

    # Compute Euclidean distance from wrist for each of the 21 landmarks
    dists = np.linalg.norm(rel_coords, axis=2)  # shape (N, 21)
    max_d = np.max(dists, axis=1, keepdims=True)  # shape (N, 1)
    max_d = np.where(max_d > 1e-6, max_d, 1.0)

    # Scale coordinates
    norm_coords = rel_coords / max_d[:, :, np.newaxis]
    out = norm_coords.reshape(n_samples, 63)
    return out[0] if is_1d else out


def prepare_features_and_labels(df: pd.DataFrame):
    """
    Separates 63 coordinate features from label strings, applies wrist-relative
    and scale normalization to features, and converts string classes into
    integer indices and one-hot vectors.
    """
    # 63 features: 21 landmarks x (x, y, z) coordinates
    X_raw = df.drop(columns=["label"]).values.astype("float32")
    X = normalize_landmarks(X_raw)
    y_raw = df["label"].values

    # Encode string names into 0..N-1 class indices
    le = LabelEncoder()
    y_int = le.fit_transform(y_raw)
    num_classes = len(le.classes_)

    if num_classes < 2:
        print(
            f"\n[ERROR] Only {num_classes} class found: {list(le.classes_)}\n"
            "  At least 2 distinct sign classes are required to train a classifier.\n"
            "  Please run collect_landmarks.py to record an additional sign.\n"
        )
        sys.exit(1)

    # One-hot encode targets for categorical crossentropy training
    y_onehot = keras.utils.to_categorical(y_int, num_classes=num_classes)

    return X, y_onehot, y_int, y_raw, le


# ===========================================================================
# Step 3: Train / Test Split (80/20 Stratified)
# ===========================================================================
def split_dataset(X, y_onehot, y_int, y_raw, test_size=0.2, random_state=42):
    """
    Splits features and labels into 80% training and 20% validation/testing sets.
    Stratified on integer labels to guarantee proportional representation.
    """
    X_train, X_test, y_train, y_test, y_raw_train, y_raw_test = train_test_split(
        X, y_onehot, y_raw,
        test_size=test_size,
        stratify=y_int,
        random_state=random_state,
    )
    print(f"  Training samples  : {len(X_train)}")
    print(f"  Testing samples   : {len(X_test)}\n")
    return X_train, X_test, y_train, y_test, y_raw_test


# ===========================================================================
# Step 4: Build Keras Sequential Neural Network
# ===========================================================================
def build_keras_model(input_dim: int, num_classes: int) -> keras.Model:
    """
    Constructs a lightweight Feed-Forward Neural Network:
      - Input: 63 normalized landmark coordinates
      - Dense(128, ReLU) + Dropout(0.3)
      - Dense(64,  ReLU) + Dropout(0.2)
      - Dense(num_classes, Softmax) for sign probability distribution
    """
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(input_dim,), name="landmarks_input"),
        keras.layers.Dense(128, activation="relu", name="dense_hidden_1"),
        keras.layers.Dropout(0.3, name="dropout_1"),
        keras.layers.Dense(64, activation="relu", name="dense_hidden_2"),
        keras.layers.Dropout(0.2, name="dropout_2"),
        keras.layers.Dense(num_classes, activation="softmax", name="sign_probabilities"),
    ], name="sign_language_classifier")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


# ===========================================================================
# Step 5: Train Model with Early Stopping
# ===========================================================================
def train_neural_network(model: keras.Model, X_train, y_train, X_test, y_test, epochs=200):
    """
    Trains the model with early stopping on validation loss to prevent overfitting.
    Restores the weights from the best performing epoch automatically.
    """
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1,
    )
    return history


# ===========================================================================
# Step 6: Evaluate & Compare with RandomForest Baseline
# ===========================================================================
def evaluate_model(model: keras.Model, X_test, y_test, y_raw_test) -> float:
    """
    Evaluates final test accuracy of the Keras model.
    If model.pkl exists, evaluates the RandomForest model and warns if Keras
    accuracy is notably lower.
    """
    loss, keras_acc = model.evaluate(X_test, y_test, verbose=0)
    print("\n" + "-" * 60)
    print(f"  >>> Final Keras Test Accuracy: {keras_acc * 100:.2f}% (Loss: {loss:.4f}) <<<")
    print("-" * 60 + "\n")

    # Check if a trained scikit-learn RandomForest model is available
    if os.path.exists(RF_MODEL_PATH):
        try:
            rf_model = joblib.load(RF_MODEL_PATH)
            rf_acc = rf_model.score(X_test, y_raw_test)
            print(f"  [BASELINE] RandomForest accuracy on test set: {rf_acc * 100:.2f}%")

            if (rf_acc - keras_acc) > RF_ACCURACY_WARN_DELTA:
                gap = (rf_acc - keras_acc) * 100
                print(
                    f"\n  [WARNING] Keras neural net accuracy is {gap:.1f}% lower than RandomForest.\n"
                    "            Consider recording more samples per gesture class (~200+ samples).\n"
                    "            Proceeding with TF.js conversion as planned.\n"
                )
            else:
                print("  [OK] Keras accuracy is competitive with RandomForest.\n")
        except Exception as err:
            print(f"  [INFO] Could not evaluate against {RF_MODEL_PATH}: {err}\n")
    else:
        print(f"  [INFO] No {RF_MODEL_PATH} found for comparison (run train_model.py to benchmark).\n")

    return keras_acc


# ===========================================================================
# Step 7: Convert and Export Model to TensorFlow.js
# ===========================================================================
def export_to_tfjs(model: keras.Model, output_dir: str, saved_model_dir: str) -> None:
    """
    Exports model directly to TensorFlow.js web format (model.json + binary shards).
    Uses tfjs.converters.save_keras_model with CLI fallback.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Exporting TensorFlow.js model to: {os.path.abspath(output_dir)} ...")

    converted = False
    try:
        # Direct Python converter API
        tfjs.converters.save_keras_model(model, output_dir)
        converted = True
    except Exception as e:
        print(f"  [INFO] Direct save note ({e}), attempting CLI conversion fallback...")

    if not converted:
        # Save temporary SavedModel for converter CLI
        if os.path.exists(saved_model_dir):
            shutil.rmtree(saved_model_dir)
        model.save(saved_model_dir)

        # Execute tensorflowjs_converter CLI
        for fmt in ["tf_saved_model", "keras_saved_model"]:
            cmd = [
                sys.executable, "-m", "tensorflowjs.converters.converter",
                f"--input_format={fmt}",
                saved_model_dir,
                output_dir,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                converted = True
                break

    if not converted:
        print("\n[ERROR] Failed to convert model to TensorFlow.js format.")
        sys.exit(1)

    print("\n  Exported TensorFlow.js artifacts:")
    for file_name in sorted(os.listdir(output_dir)):
        file_path = os.path.join(output_dir, file_name)
        size_kb = os.path.getsize(file_path) / 1024
        print(f"    - {file_name:40s} ({size_kb:6.1f} KB)")
    print()


# ===========================================================================
# Step 8: Save and Synchronize labels.json
# ===========================================================================
def export_labels(le: LabelEncoder, dest_path: str, src_path: str) -> None:
    """
    Saves the ordered class label list to web-app/model/labels.json and
    python-training/labels.json so the browser and Python scripts stay synchronized.
    """
    labels_list = list(le.classes_)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # Save to web-app/model/labels.json
    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(labels_list, f, indent=2)

    # Save to python-training/labels.json
    with open(src_path, "w", encoding="utf-8") as f:
        json.dump(labels_list, f, indent=2)

    print(f"  Saved labels to : {os.path.abspath(dest_path)}")
    print(f"  Class labels    : {labels_list}\n")


# ===========================================================================
# Main Pipeline Execution
# ===========================================================================
def main() -> None:
    print("\n" + "=" * 65)
    print("  Sign-Language Neural Network Trainer & TensorFlow.js Exporter")
    print("=" * 65 + "\n")

    # Step 1: Load CSVs
    print("[ Step 1/8 ] Loading landmark CSV files from dataset/ ...")
    df = load_dataset(DATASET_DIR)
    print(f"  Total frames loaded: {len(df)}\n")

    # Step 1b: Drop classes with too few samples (prevents stratified-split crash)
    # 'nothing' gets 0-1 rows because MediaPipe finds no hand in those images.
    MIN_SAMPLES = 5
    counts = df["label"].value_counts()
    rare = counts[counts < MIN_SAMPLES]
    if not rare.empty:
        print(f"  [WARNING] Dropping classes with < {MIN_SAMPLES} samples:")
        for lbl, cnt in rare.items():
            print(f"    - '{lbl}': {cnt} sample(s)")
        df = df[~df["label"].isin(rare.index)].reset_index(drop=True)
        print(f"  Remaining samples : {len(df)}")
        print(f"  Remaining classes : {sorted(df['label'].unique())}\n")

    # Step 2: Prepare features and labels
    print("[ Step 2/8 ] Encoding labels & preparing feature tensors ...")
    X, y_onehot, y_int, y_raw, le = prepare_features_and_labels(df)
    num_features = X.shape[1]
    num_classes = len(le.classes_)
    print(f"  Detected classes ({num_classes}) : {list(le.classes_)}")
    print(f"  Feature count per frame  : {num_features} (21 landmarks x 3 coords)\n")

    # Step 3: Stratified train/test split
    print("[ Step 3/8 ] Splitting dataset into train (80%) and test (20%) sets ...")
    X_train, X_test, y_train, y_test, y_raw_test = split_dataset(X, y_onehot, y_int, y_raw)

    # Step 4: Build Keras Sequential model
    print("[ Step 4/8 ] Constructing Keras Sequential neural network ...")
    model = build_keras_model(input_dim=num_features, num_classes=num_classes)
    print()

    # Step 5: Train model
    print("[ Step 5/8 ] Training neural network (with EarlyStopping on val_loss) ...")
    train_neural_network(model, X_train, y_train, X_test, y_test)
    print()

    # Step 6: Evaluate
    print("[ Step 6/8 ] Evaluating model test accuracy ...")
    evaluate_model(model, X_test, y_test, y_raw_test)

    # Step 7: Convert & export to TF.js
    print("[ Step 7/8 ] Converting and exporting to TensorFlow.js ...")
    export_to_tfjs(model, TFJS_OUTPUT_DIR, SAVED_MODEL_DIR)

    # Step 8: Save labels.json
    print("[ Step 8/8 ] Writing class labels mapping ...")
    export_labels(le, LABELS_DST, LABELS_SRC)

    print("=" * 65)
    print("  [SUCCESS] Conversion complete!")
    print(f"  Model files: {os.path.abspath(TFJS_OUTPUT_DIR)}/")
    print(f"  Labels file: {os.path.abspath(LABELS_DST)}")
    print("\n  Browser loading example:")
    print("    const model  = await tf.loadLayersModel('model/model.json');")
    print("    const labels = await fetch('model/labels.json').then(res => res.json());")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
