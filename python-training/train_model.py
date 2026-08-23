"""
train_model.py
--------------
Loads all per-class landmark CSVs from the dataset/ directory, trains a
Random Forest classifier, evaluates it, and saves the model + label list.

Dependencies:
    pip install pandas scikit-learn joblib

Usage:
    python train_model.py
    (Run from the python-training/ directory, or any directory that contains
     a 'dataset/' sub-folder with at least two class CSVs.)
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_DIR = "dataset"
MODEL_PATH  = "model.pkl"
LABELS_PATH = "labels.json"


# ---------------------------------------------------------------------------
# Step 1 - Load all CSVs from dataset/
# ---------------------------------------------------------------------------
def load_dataset(dataset_dir: str) -> pd.DataFrame:
    """
    Read every .csv file found in dataset_dir and concatenate them into one
    DataFrame.  Each CSV is expected to have the format written by
    collect_landmarks.py:
        label, x0, y0, z0, x1, y1, z1, ..., x20, y20, z20
    """
    csv_files = [
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if f.endswith(".csv")
    ]

    if not csv_files:
        print(
            "\n[ERROR] No CSV files found in dataset/.\n"
            "  Run collect_landmarks.py first to record at least 2 sign classes.\n"
        )
        sys.exit(1)

    frames = []
    for path in sorted(csv_files):
        df = pd.read_csv(path)
        # Skip empty files or files with only a header row
        if df.empty:
            print(f"  [SKIP] {path} is empty.")
            continue
        frames.append(df)
        print(f"  [LOAD] {path:50s}  rows: {len(df)}")

    if not frames:
        print(
            "\n[ERROR] All CSV files were empty.\n"
            "  Hold [r] in collect_landmarks.py to actually save frames.\n"
        )
        sys.exit(1)

    # pd.concat merges the list of DataFrames row-wise (axis=0).
    combined = pd.concat(frames, ignore_index=True)
    return combined


# ---------------------------------------------------------------------------
# Step 2 - Validate that we have enough classes to train
# ---------------------------------------------------------------------------
def validate_classes(df: pd.DataFrame) -> None:
    """
    A classifier needs at least 2 distinct classes.  Print a friendly message
    and exit if the dataset doesn't meet this requirement.
    """
    classes = df["label"].unique()
    n_classes = len(classes)

    if n_classes < 2:
        print(
            f"\n[ERROR] Only {n_classes} class(es) found: {list(classes)}\n"
            "  A classifier needs at least 2 different sign labels.\n"
            "  Run collect_landmarks.py again and record a second sign.\n"
        )
        sys.exit(1)

    print(f"\n  Classes found ({n_classes}): {sorted(classes)}")
    print(f"  Total samples  : {len(df)}\n")


# ---------------------------------------------------------------------------
# Step 2b - Drop classes with too few samples for stratified splitting
# ---------------------------------------------------------------------------
MIN_SAMPLES_PER_CLASS = 5  # stratify= requires at least 2; use 5 for safety

def drop_rare_classes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes any class whose total sample count is below MIN_SAMPLES_PER_CLASS.

    WHY: sklearn's train_test_split with stratify=y requires every class to
    have at least 2 samples (1 for train, 1 for test).  Classes with almost
    no samples (e.g. 'nothing' when MediaPipe detected zero hands) cause a
    ValueError and can't be stratified meaningfully anyway.

    Prints a clear warning listing every dropped class and its sample count.
    Exits if dropping leaves fewer than 2 classes.
    """
    counts = df["label"].value_counts()
    rare   = counts[counts < MIN_SAMPLES_PER_CLASS]

    if not rare.empty:
        print(f"  [WARNING] The following classes have fewer than {MIN_SAMPLES_PER_CLASS} samples")
        print(  "            and will be DROPPED before training:\n")
        for label, count in rare.items():
            print(f"    - '{label}': {count} sample(s)")
        print()

        # Remove rows whose label is in the rare set
        df = df[~df["label"].isin(rare.index)].reset_index(drop=True)
        print(f"  Remaining samples after drop : {len(df)}")
        print(f"  Remaining classes            : {sorted(df['label'].unique())}\n")
    else:
        print(f"  All classes have >= {MIN_SAMPLES_PER_CLASS} samples. Nothing dropped.\n")

    # Final guard: still need >= 2 classes
    if df["label"].nunique() < 2:
        print(
            "\n[ERROR] After dropping rare classes, fewer than 2 classes remain.\n"
            "  Collect more samples before training.\n"
        )
        sys.exit(1)

    return df


# ---------------------------------------------------------------------------
# Step 3 - Normalize coordinates (wrist-relative & scale-invariant)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Step 3b - Split into features (X) and labels (y)
# ---------------------------------------------------------------------------
def split_features_labels(df: pd.DataFrame):
    """
    The 'label' column is the target (y).
    All other columns are landmark coordinates (X).
    Applies wrist-relative and scale normalization to X.
    """
    # Drop the label column to get the feature matrix
    X_raw = df.drop(columns=["label"]).values.astype("float32")  # shape: (n_samples, 63)
    X = normalize_landmarks(X_raw)
    y = df["label"].values                  # shape: (n_samples,)
    return X, y


# ---------------------------------------------------------------------------
# Step 4 - Train / test split
# ---------------------------------------------------------------------------
def make_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Split data into 80 % training and 20 % test sets.

    stratify=y   -> each class keeps the same proportion in both splits,
                    so a rare class isn't accidentally left out of the test set.
    random_state -> fixed seed for reproducibility.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    print(f"  Train samples  : {len(X_train)}")
    print(f"  Test  samples  : {len(X_test)}\n")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Step 5 - Train the Random Forest
# ---------------------------------------------------------------------------
def train(X_train, y_train, n_estimators=200, random_state=42):
    """
    RandomForestClassifier:
      n_estimators  - number of decision trees in the forest; more trees =
                      better accuracy (up to a point) but slower training.
      random_state  - fixed seed so results are reproducible.
      n_jobs=-1     - use all available CPU cores for parallel tree training.
    """
    print(f"  Training RandomForestClassifier (n_estimators={n_estimators}) ...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    # .fit() is where the actual training happens: each tree is built on a
    # bootstrap sample of the training data.
    clf.fit(X_train, y_train)
    print("  Training complete.\n")
    return clf


# ---------------------------------------------------------------------------
# Step 6 - Evaluate
# ---------------------------------------------------------------------------
def evaluate(clf, X_test, y_test) -> None:
    """
    Evaluate the trained model on the held-out test set and print:
      * overall accuracy
      * per-class precision, recall, F1
      * confusion matrix
    """
    # clf.score() computes (correct predictions) / (total predictions)
    accuracy = clf.score(X_test, y_test)
    print(f"  Test Accuracy : {accuracy * 100:.2f}%\n")

    # y_pred contains the model's predicted label for each test sample
    y_pred = clf.predict(X_test)

    # classification_report shows precision / recall / F1 per class,
    # which is more informative than accuracy alone for multi-class problems.
    print("  Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # confusion_matrix[i][j] = how many samples of class i were predicted as j.
    # Off-diagonal cells reveal which pairs of signs are confused most often.
    classes = sorted(clf.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    print("  Confusion Matrix (rows = true label, cols = predicted label):")
    # Pretty-print with class names as column headers
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    print(cm_df.to_string())
    print()


# ---------------------------------------------------------------------------
# Step 7 - Save model and label list
# ---------------------------------------------------------------------------
def save_artifacts(clf, model_path: str, labels_path: str) -> None:
    """
    Persist two artefacts needed for inference:
      model.pkl    - the trained RandomForest (loaded by the app at runtime)
      labels.json  - ordered list of class names so we can map clf.predict()
                     integer indices back to human-readable sign names
    """
    # joblib.dump() serialises the sklearn model to a file.
    # It is preferred over pickle for numpy arrays because it is faster and
    # produces smaller files.
    joblib.dump(clf, model_path)
    print(f"  Model saved to  : {os.path.abspath(model_path)}")

    # clf.classes_ is a numpy array of class labels in the order the model
    # uses internally.  We convert to a plain Python list before serialising.
    labels = list(clf.classes_)
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"  Labels saved to : {os.path.abspath(labels_path)}")
    print(f"  Label order     : {labels}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("\n" + "=" * 60)
    print("  Sign-Language Model Trainer")
    print("=" * 60 + "\n")

    # 1. Load
    print("[ Step 1 ] Loading CSV files from dataset/ ...")
    df = load_dataset(DATASET_DIR)

    # 2. Validate
    print("[ Step 2 ] Validating classes ...")
    validate_classes(df)

    # 2b. Drop classes with too few samples (prevents stratified-split ValueError)
    print("[ Step 2b ] Dropping rare classes (< 5 samples) ...")
    df = drop_rare_classes(df)

    # 3. Split features / labels
    print("[ Step 3 ] Splitting features and labels ...")
    X, y = split_features_labels(df)

    # 4. Train / test split
    print("[ Step 4 ] Creating train / test split (80 / 20) ...")
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)

    # 5. Train
    print("[ Step 5 ] Training ...")
    clf = train(X_train, y_train)

    # 6. Evaluate
    print("[ Step 6 ] Evaluation on test set ...")
    evaluate(clf, X_test, y_test)

    # 7. Save
    print("[ Step 7 ] Saving artefacts ...")
    save_artifacts(clf, MODEL_PATH, LABELS_PATH)

    print("=" * 60)
    print("  Done!  Run your app to start classifying signs.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
