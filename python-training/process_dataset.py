"""
process_dataset.py
------------------
Batch processes an image dataset of sign language gestures (e.g. Kaggle ASL Alphabet)
using MediaPipe Hands to extract 21 (x, y, z) normalized landmarks per image and
exports them into per-label CSV files in the format expected by train_model.py and
convert_to_tfjs.py.

Directory structure expected for --input:
    input_folder/
        A/
            A1.jpg
            A2.jpg
            ...
        B/
            B1.jpg
            ...
        hello/
            ...

Output structure for --output:
    dataset/
        A.csv
        B.csv
        hello.csv
        ...

Each CSV row:
    label, x0,y0,z0, x1,y1,z1, ..., x20,y20,z20

Dependencies:
    pip install opencv-python mediapipe

Usage:
    python process_dataset.py --input "C:/Users/kirta/Downloads/ASL Alphabet/asl_alphabet_train/asl_alphabet_train" --output dataset/
    python process_dataset.py -i dataset_raw/asl_alphabet_train -o dataset/
"""

import argparse
import csv
import os
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode

# ---------------------------------------------------------------------------
# Constants & Model Configuration
# ---------------------------------------------------------------------------
NUM_LANDMARKS = 21  # Standard 21 hand landmarks per MediaPipe Hand model
MODEL_FILENAME = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# Supported image file extensions
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_model_file(model_path: str) -> str:
    """
    Checks if the MediaPipe HandLandmarker .task file exists.
    If missing, downloads it automatically from Google Cloud Storage.
    """
    if os.path.exists(model_path):
        return model_path

    # Check script's directory if not found in current working directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    alt_path = os.path.join(script_dir, MODEL_FILENAME)
    if os.path.exists(alt_path):
        return alt_path

    print(f"  [+] Downloading MediaPipe model file to '{model_path}'...")
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("  [+] Model download complete.\n")
        return model_path
    except Exception as e:
        print(f"\n[ERROR] Failed to download model from {MODEL_URL}: {e}")
        sys.exit(1)


def build_csv_header() -> list:
    """
    Constructs the standard CSV column headers:
    ['label', 'x0', 'y0', 'z0', ..., 'x20', 'y20', 'z20']
    """
    header = ["label"]
    for i in range(NUM_LANDMARKS):
        header.extend([f"x{i}", f"y{i}", f"z{i}"])
    return header


def extract_row(label: str, hand_landmarks) -> list:
    """
    Flattens 21 NormalizedLandmark objects into a single list of floats:
    [label, x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]
    """
    row = [label]
    for lm in hand_landmarks:
        row.extend([lm.x, lm.y, lm.z])
    return row


def process_dataset(input_dir: str, output_dir: str, max_hands: int = 1, min_confidence: float = 0.5):
    """
    Iterates through all class subdirectories in input_dir, runs landmark detection
    on every valid image, and saves extracted landmark rows into output_dir/<label>.csv.
    """
    if not os.path.exists(input_dir):
        print(f"\n[ERROR] Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    header = build_csv_header()
    model_path = ensure_model_file(MODEL_FILENAME)

    # -----------------------------------------------------------------------
    # Step 1: Discover class subfolders
    # -----------------------------------------------------------------------
    entries = sorted(os.listdir(input_dir))
    class_folders = [
        d for d in entries
        if os.path.isdir(os.path.join(input_dir, d))
    ]

    if not class_folders:
        print(
            f"\n[ERROR] No subfolders found in '{input_dir}'.\n"
            "  Expected subfolders named by class label (e.g. 'A/', 'B/', 'hello/').\n"
        )
        sys.exit(1)

    print("\n" + "=" * 65)
    print("  ASL Dataset Landmark Batch Processor")
    print("=" * 65)
    print(f"  Input Directory   : {os.path.abspath(input_dir)}")
    print(f"  Output Directory  : {os.path.abspath(output_dir)}")
    print(f"  Classes Found ({len(class_folders):2d}): {', '.join(class_folders)}")
    print("=" * 65 + "\n")

    # -----------------------------------------------------------------------
    # Step 2: Initialize MediaPipe HandLandmarker in static IMAGE mode
    # We create a single detector instance and reuse it for all images (high efficiency).
    # -----------------------------------------------------------------------
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.IMAGE,  # IMAGE mode for independent static frames
        num_hands=max_hands,
        min_hand_detection_confidence=min_confidence,
        min_hand_presence_confidence=min_confidence,
    )

    total_images_all = 0
    total_saved_all = 0
    total_skipped_all = 0
    start_time_all = time.time()

    with mp_vision.HandLandmarker.create_from_options(options) as detector:
        for idx, label in enumerate(class_folders, 1):
            folder_path = os.path.join(input_dir, label)
            image_files = sorted([
                f for f in os.listdir(folder_path)
                if os.path.splitext(f.lower())[1] in VALID_EXTENSIONS
            ])

            num_images = len(image_files)
            if num_images == 0:
                print(f"[{idx}/{len(class_folders)}] Class '{label}': No valid images found, skipping.")
                continue

            csv_file_path = os.path.join(output_dir, f"{label}.csv")
            print(f"[{idx}/{len(class_folders)}] Processing Class '{label}' ({num_images} images) -> {csv_file_path}")

            saved_count = 0
            skipped_count = 0
            label_start_time = time.time()

            # Open/create CSV with header
            with open(csv_file_path, "w", newline="", encoding="utf-8") as csv_f:
                writer = csv.writer(csv_f)
                writer.writerow(header)

                for img_idx, img_name in enumerate(image_files, 1):
                    img_path = os.path.join(folder_path, img_name)

                    try:
                        # 1. Read image with OpenCV (BGR)
                        bgr_img = cv2.imread(img_path)
                        if bgr_img is None:
                            skipped_count += 1
                            continue

                        # 2. Convert BGR -> RGB as required by MediaPipe
                        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

                        # 3. Wrap in MediaPipe Image container
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

                        # 4. Perform hand landmark detection
                        detection_result = detector.detect(mp_image)

                        # 5. Extract landmarks if at least one hand detected
                        if detection_result.hand_landmarks:
                            # Save all detected hands (typically 1 for ASL letters)
                            for hand in detection_result.hand_landmarks:
                                row = extract_row(label, hand)
                                writer.writerow(row)
                                saved_count += 1
                        else:
                            skipped_count += 1

                    except Exception as err:
                        # Catch corrupted images or decode failures gracefully
                        skipped_count += 1

                    # Print progress every 500 images
                    if img_idx % 500 == 0 or img_idx == num_images:
                        elapsed = time.time() - label_start_time
                        fps = img_idx / max(elapsed, 0.001)
                        print(
                            f"   Progress: {img_idx:5d}/{num_images:5d} "
                            f"(Saved: {saved_count:5d}, Skipped: {skipped_count:4d}) "
                            f"- {fps:5.1f} img/s"
                        )

            total_images_all += num_images
            total_saved_all += saved_count
            total_skipped_all += skipped_count
            label_time = time.time() - label_start_time
            print(f"   Done '{label}' in {label_time:.1f}s | Saved: {saved_count} | Skipped: {skipped_count}\n")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total_time = time.time() - start_time_all
    avg_fps = total_images_all / max(total_time, 0.001)
    detection_rate = (total_saved_all / max(total_images_all, 1)) * 100

    print("=" * 65)
    print("  Processing Summary")
    print("=" * 65)
    print(f"  Total Images Processed : {total_images_all:,}")
    print(f"  Landmarks Saved (Rows) : {total_saved_all:,} ({detection_rate:.1f}% detection rate)")
    print(f"  Images Skipped (No hand): {total_skipped_all:,}")
    print(f"  Total Time Elapsed     : {total_time:.1f} seconds (~{total_time/60:.1f} min)")
    print(f"  Average Processing Speed: {avg_fps:.1f} images/second")
    print(f"  Output CSVs Saved In   : {os.path.abspath(output_dir)}/")
    print("=" * 65 + "\n")


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch extract 21 MediaPipe hand landmarks from an image folder of sign classes into CSVs."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="dataset_raw/asl_alphabet_train",
        help="Path to root folder containing label subdirectories with images.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="dataset",
        help="Path to output folder where <label>.csv files will be saved.",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=1,
        help="Maximum number of hands to detect per image (default: 1).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence score between 0.0 and 1.0 (default: 0.5).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_dataset(
        input_dir=args.input,
        output_dir=args.output,
        max_hands=args.max_hands,
        min_confidence=args.min_confidence,
    )
