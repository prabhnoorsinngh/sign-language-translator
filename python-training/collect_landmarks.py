"""
collect_landmarks.py
--------------------
Collects hand-landmark data from a live webcam feed and saves it to CSV files
for use as a sign-language training dataset.

Dependencies:
    pip install mediapipe opencv-python

Usage:
    python collect_landmarks.py

MediaPipe version note:
    mediapipe >= 0.10 / 1.x uses the new "Tasks" API (mp.tasks.vision).
    The older mp.solutions.hands API was removed.
    This script uses mp.tasks.vision.HandLandmarker.
"""

import csv
import os
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode

# ---------------------------------------------------------------------------
# MediaPipe Task model file
# ---------------------------------------------------------------------------
# The new API requires a .task model file downloaded from MediaPipe.
# We download it automatically on first run if it isn't present.
MODEL_FILENAME = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_DIR   = "dataset"   # folder where CSV files are saved
NUM_LANDMARKS = 21          # MediaPipe always gives exactly 21 hand landmarks


def download_model_if_needed(path: str) -> None:
    """Download the HandLandmarker .task model file if it is not present."""
    if os.path.exists(path):
        return
    print(f"  [DL] Downloading model to {path} ...")
    import urllib.request
    urllib.request.urlretrieve(MODEL_URL, path)
    print("  [DL] Download complete.\n")


def print_instructions() -> None:
    """Print startup instructions to the terminal."""
    print("\n" + "=" * 60)
    print("  Sign-Language Landmark Collector")
    print("=" * 60)
    print("  1. Type a label when prompted (e.g. 'hello', 'A', 'thankyou')")
    print("  2. A webcam window will open.")
    print("  3. Hold  [r]  to RECORD frames for the current label.")
    print("  4. Press [q]  to QUIT the program.")
    print("  5. Rows are saved to  dataset/<label>.csv  automatically.")
    print("  6. A running sample count is printed in the terminal.")
    print("=" * 60 + "\n")


def get_label() -> str:
    """Prompt the user to type a label; strip whitespace and validate."""
    while True:
        label = input("Enter label to record (e.g. 'hello'): ").strip()
        if label:
            return label
        print("  [!] Label cannot be empty, please try again.")


def ensure_csv(label: str) -> str:
    """
    Create the dataset directory and return the path to <label>.csv.
    If the file doesn't exist yet it is created with a header row.
    """
    os.makedirs(DATASET_DIR, exist_ok=True)
    filepath = os.path.join(DATASET_DIR, f"{label}.csv")

    if not os.path.exists(filepath):
        # Build column names: label, x0,y0,z0, x1,y1,z1, ..., x20,y20,z20
        header = ["label"]
        for i in range(NUM_LANDMARKS):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        with open(filepath, "w", newline="") as f:
            csv.writer(f).writerow(header)
        print(f"  [+] Created new CSV: {filepath}")
    else:
        print(f"  [~] Appending to existing CSV: {filepath}")

    return filepath


def count_existing_samples(filepath: str) -> int:
    """Return the number of data rows already in a CSV (excludes header)."""
    if not os.path.exists(filepath):
        return 0
    with open(filepath, "r", newline="") as f:
        return max(0, sum(1 for _ in f) - 1)


def landmarks_to_row(label: str, hand_landmarks) -> list:
    """
    Flatten one hand's 21 NormalizedLandmark objects into a CSV row.

    In the Tasks API, hand_landmarks is a list of 21 NormalizedLandmark objects,
    each having:
        .x  - horizontal position (0.0 = left edge, 1.0 = right edge)
        .y  - vertical position   (0.0 = top,       1.0 = bottom)
        .z  - depth estimate relative to the wrist (negative = closer)

    Returns: [label, x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]
    """
    row = [label]
    for lm in hand_landmarks:   # iterates over all 21 NormalizedLandmark items
        row += [lm.x, lm.y, lm.z]
    return row


def draw_landmarks_on_frame(frame, detection_result) -> None:
    """
    Manually draw the 21 hand landmarks and their connections onto the frame.

    detection_result.hand_landmarks is a list of hands, where each hand is a
    list of 21 NormalizedLandmark objects with .x/.y in [0, 1].  We scale
    them to pixel coordinates before drawing.
    """
    # Standard MediaPipe hand connections: pairs of landmark indices to connect.
    HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),         # thumb
        (0,5),(5,6),(6,7),(7,8),         # index finger
        (0,9),(9,10),(10,11),(11,12),    # middle finger
        (0,13),(13,14),(14,15),(15,16),  # ring finger
        (0,17),(17,18),(18,19),(19,20),  # pinky
        (5,9),(9,13),(13,17),            # palm
    ]

    h, w = frame.shape[:2]

    for hand in detection_result.hand_landmarks:
        # Convert normalised coords -> pixel coords for this hand
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

        # Draw connections (skeleton lines)
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, pts[start_idx], pts[end_idx], (0, 200, 0), 2)

        # Draw landmark dots
        for pt in pts:
            cv2.circle(frame, pt, 4, (0, 0, 255), -1)   # filled red dot
            cv2.circle(frame, pt, 4, (255, 255, 255), 1) # white outline


def main() -> None:
    print_instructions()
    download_model_if_needed(MODEL_FILENAME)

    label = get_label()
    csv_path = ensure_csv(label)
    sample_count = count_existing_samples(csv_path)
    print(f"  Starting with {sample_count} existing samples for '{label}'.\n")

    # -----------------------------------------------------------------------
    # Open the default webcam (device index 0).
    # Change the argument to 1, 2, ... if you have multiple cameras.
    # -----------------------------------------------------------------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("[ERROR] Could not open webcam. Check that it is connected.")

    # -----------------------------------------------------------------------
    # Build the HandLandmarker using the new MediaPipe Tasks API.
    #
    # HandLandmarkerOptions parameters:
    #   base_options          - points to the .task model file on disk
    #   running_mode          - VIDEO = process frames with timestamps (faster
    #                           than IMAGE mode for live streams because it
    #                           uses temporal tracking between frames)
    #   num_hands             - maximum number of hands to detect per frame
    #   min_hand_detection_confidence - minimum score (0-1) for initial detect
    #   min_hand_presence_confidence  - minimum score (0-1) to keep a tracked
    #                                   hand visible across frames
    #   min_tracking_confidence       - minimum IOU score to maintain tracking
    # -----------------------------------------------------------------------
    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_FILENAME)
    options   = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=RunningMode.VIDEO,       # use VIDEO for webcam streams
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print("  [Webcam open]  Hold [r] to record  |  Press [q] to quit\n")

    # HandLandmarker is used as a context manager to ensure resources are freed.
    with mp_vision.HandLandmarker.create_from_options(options) as detector:

        while True:
            # -----------------------------------------------------------------
            # Read one frame from the webcam.
            # cap.read() returns (success_flag, BGR_image_array).
            # cap.get(cv2.CAP_PROP_POS_MSEC) gives the timestamp in ms.
            # -----------------------------------------------------------------
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Failed to read frame; skipping.")
                continue

            # Flip horizontally so the image acts like a mirror.
            frame = cv2.flip(frame, 1)

            # -----------------------------------------------------------------
            # The Tasks API wraps images in an mp.Image container.
            # mp.ImageFormat.SRGB tells MediaPipe the pixel layout is RGB.
            # We first convert from OpenCV's BGR to RGB.
            # -----------------------------------------------------------------
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # -----------------------------------------------------------------
            # detector.detect_for_video(mp_image, timestamp_ms)
            #
            # This is the core inference call for VIDEO running mode.
            # The timestamp_ms MUST be strictly monotonically increasing.
            #
            # WHY NOT cap.get(cv2.CAP_PROP_POS_MSEC):
            #   On Windows, OpenCV reports position 0 for the first several
            #   frames from a live webcam.  Passing the same timestamp twice
            #   causes detect_for_video to block forever (deadlock).
            #
            # FIX: use wall-clock time (nanoseconds -> milliseconds).
            #   time.time_ns() is always strictly increasing and has enough
            #   resolution (~100 ns on Windows) to be unique every frame.
            # -----------------------------------------------------------------
            timestamp_ms = time.time_ns() // 1_000_000  # ns -> ms, always increases
            result = detector.detect_for_video(mp_image, timestamp_ms)

            # -----------------------------------------------------------------
            # Draw the hand skeleton on the frame if any hands were found.
            # result.hand_landmarks is an empty list when nothing is detected.
            # -----------------------------------------------------------------
            if result.hand_landmarks:
                draw_landmarks_on_frame(frame, result)

            # -----------------------------------------------------------------
            # Check keyboard input.
            # waitKey(1) waits 1 ms; & 0xFF is needed on 64-bit systems.
            # -----------------------------------------------------------------
            key = cv2.waitKey(1) & 0xFF

            # ---- QUIT -------------------------------------------------------
            if key == ord("q"):
                print("\n  [q] Quit requested. Goodbye!")
                break

            # ---- RECORD -----------------------------------------------------
            # Held-down keys repeat on most OSes, so recording is continuous
            # while the key is held.
            if key == ord("r"):
                if result.hand_landmarks:
                    with open(csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        # Record ALL detected hands in this frame.
                        for hand in result.hand_landmarks:
                            row = landmarks_to_row(label, hand)
                            writer.writerow(row)
                            sample_count += 1

                    print(
                        f"\r  [REC] Samples saved for '{label}': {sample_count}",
                        end="", flush=True,
                    )
                else:
                    print(
                        f"\r  [!]  No hand detected - nothing saved this frame.   ",
                        end="", flush=True,
                    )

            # -----------------------------------------------------------------
            # HUD overlay on the frame.
            # -----------------------------------------------------------------
            hud_color = (0, 60, 255) if key == ord("r") else (200, 200, 200)
            rec_text  = "REC" if key == ord("r") else "---"

            cv2.putText(
                frame,
                f"Label: {label}  |  Samples: {sample_count}  |  {rec_text}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, hud_color, 2, cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "Hold [r] to record  |  [q] to quit",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA,
            )

            cv2.imshow("Sign Translator - Landmark Collector", frame)

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  Done. Total samples for '{label}': {sample_count}")
    print(f"  Saved to: {os.path.abspath(csv_path)}\n")


if __name__ == "__main__":
    main()
