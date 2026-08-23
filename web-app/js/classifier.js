/**
 * classifier.js
 * -------------
 * TensorFlow.js sign language inference engine.
 *
 * Responsibilities:
 * 1. Load the exported Keras Sequential model from `model/model.json`.
 * 2. Load `model/labels.json` class mapping (28 classes: A-Z, del, space).
 * 3. Poll `getCurrentLandmarks()` and `getCurrentHandedness()` from `landmarks.js` every ~200ms.
 * 4. If the detected hand is "Left" (camera perspective), mirror x-coordinates so the
 *    gesture matches the right-hand training data distribution. See mirrorLandmarksIfLeft().
 * 5. Apply wrist-relative and scale normalization (matching python-training).
 * 6. Run inference using `tf.tidy()` to prevent GPU/WebGL memory leaks.
 * 7. Expose `getCurrentPrediction()` returning `{ label, confidence, handedness }` or `null`.
 */

(function (window) {
  'use strict';

  // Model and labels state
  let model = null;
  let labels = [];
  let isModelLoading = false;
  let isModelReady = false;

  // Inference state
  let inferenceIntervalId = null;
  let latestPrediction = null; // { label: string, confidence: number } or null
  const listeners = [];

  const MODEL_PATH = 'model/model.json';
  const LABELS_PATH = 'model/labels.json';
  const INFERENCE_INTERVAL_MS = 200; // ~5 inferences/second for low latency & power efficiency

  /**
   * Loads the TensorFlow.js model and class labels.
   *
   * @param {Object} options - Config options (e.g. modelPath, labelsPath, onStatusChange)
   * @returns {Promise<{model: tf.LayersModel, labels: string[]}>}
   */
  async function initClassifier(options = {}) {
    const modelUrl = options.modelPath || MODEL_PATH;
    const labelsUrl = options.labelsPath || LABELS_PATH;
    const onStatus = options.onStatusChange || (() => {});

    if (isModelReady) {
      return { model, labels };
    }

    if (isModelLoading) {
      return new Promise((resolve) => {
        const checkReady = setInterval(() => {
          if (isModelReady) {
            clearInterval(checkReady);
            resolve({ model, labels });
          }
        }, 50);
      });
    }

    isModelLoading = true;
    onStatus('loading', 'Loading AI gesture model...');

    // 1. Verify TensorFlow.js availability
    if (typeof window.tf === 'undefined') {
      const err = new Error(
        '[Classifier] TensorFlow.js (tf) not found. Ensure @tensorflow/tfjs CDN is included in index.html.'
      );
      console.error(err);
      onStatus('error', err.message);
      isModelLoading = false;
      throw err;
    }

    try {
      // 2. Fetch class labels JSON
      console.log(`[Classifier] Fetching class labels from ${labelsUrl}...`);
      const labelsRes = await fetch(labelsUrl);
      if (!labelsRes.ok) {
        throw new Error(`Failed to load labels from ${labelsUrl} (status: ${labelsRes.status})`);
      }
      labels = await labelsRes.json();
      console.log(`[Classifier] Loaded ${labels.length} class labels:`, labels);

      // 3. Load TensorFlow.js layers model
      console.log(`[Classifier] Loading TF.js model from ${modelUrl}...`);
      model = await window.tf.loadLayersModel(modelUrl);
      console.log('[Classifier] Model loaded successfully.');

      // 4. Warm up model with a dummy inference to compile WebGL shaders
      window.tf.tidy(() => {
        const dummyInput = window.tf.zeros([1, 63]);
        model.predict(dummyInput);
      });

      isModelReady = true;
      isModelLoading = false;
      onStatus('ready', `Model ready (${labels.length} classes)`);

      // 5. Start continuous inference loop
      startInferenceLoop();

      return { model, labels };
    } catch (err) {
      isModelLoading = false;
      isModelReady = false;
      console.error('[Classifier] Initialization failed:', err);
      onStatus('error', `Failed to load model: ${err.message}`);
      throw err;
    }
  }

  /**
   * Mirrors the x-coordinate of all 21 landmarks by computing (1 - x).
   *
   * WHY THIS IS NEEDED — Training data bias:
   *   The Kaggle ASL Alphabet dataset was captured almost exclusively with right hands.
   *   The model therefore learned sign shapes as they appear for a right-handed signer.
   *   For a left-handed user the hand shape appears as a horizontal mirror image, so
   *   without correction every letter would be systematically misclassified.
   *
   *   MediaPipe reports handedness from the CAMERA's perspective (not the user's).
   *   A hand the USER sees as their right hand is on the LEFT of the camera frame,
   *   so MediaPipe labels it "Left". We therefore mirror when MediaPipe says "Left".
   *
   *   Mirroring is applied BEFORE wrist-relative normalization so the downstream
   *   math sees geometrically equivalent right-hand coordinates.
   *
   * @param {Float32Array} raw - 63 raw landmark coordinates
   * @returns {Float32Array} x-mirrored copy
   */
  function mirrorLandmarksIfLeft(raw) {
    const mirrored = new Float32Array(raw);
    for (let i = 0; i < 21; i++) {
      mirrored[i * 3] = 1.0 - raw[i * 3]; // x_new = 1 - x_old; y and z unchanged
    }
    return mirrored;
  }

  /**
   * Normalizes hand landmarks to be wrist-relative and scale-invariant.
   * Matches the exact Python implementation:
   *   1. Subtract wrist (landmark 0) x, y, z from all 21 landmarks.
   *   2. Divide all coordinates by the maximum Euclidean distance from the wrist to any landmark.
   *
   * @param {Float32Array|number[]} raw - 63 raw landmark coordinates [x0,y0,z0,...,x20,y20,z20]
   * @returns {Float32Array} 63 normalized coordinates
   */
  function normalizeLandmarks(raw) {
    if (!raw || raw.length !== 63) return raw;

    const wristX = raw[0];
    const wristY = raw[1];
    const wristZ = raw[2];

    const rel = new Float32Array(63);
    let maxDist = 0;

    // Step 1: Subtract wrist coordinates & find max Euclidean distance from wrist
    for (let i = 0; i < 21; i++) {
      const idx = i * 3;
      const rx = raw[idx] - wristX;
      const ry = raw[idx + 1] - wristY;
      const rz = raw[idx + 2] - wristZ;

      rel[idx] = rx;
      rel[idx + 1] = ry;
      rel[idx + 2] = rz;

      const dist = Math.sqrt(rx * rx + ry * ry + rz * rz);
      if (dist > maxDist) {
        maxDist = dist;
      }
    }

    // Step 2: Scale coordinates by max Euclidean distance from wrist
    const scale = maxDist > 1e-6 ? maxDist : 1.0;
    for (let j = 0; j < 63; j++) {
      rel[j] /= scale;
    }

    return rel;
  }

  /**
   * Executes a single classification pass on the provided 63-element landmark vector.
   *
   * @param {Float32Array|number[]} landmarks - 63 landmark coordinates [x0,y0,z0,...,x20,y20,z20]
   * @returns {{ label: string, confidence: number }|null}
   */
  function predictSign(landmarks) {
    if (!model || !isModelReady || !landmarks || landmarks.length !== 63) {
      return null;
    }

    // Normalize coordinates to be wrist-relative and scale-invariant
    const normalizedCoords = normalizeLandmarks(landmarks);

    // Wrap tensor operations in tf.tidy() to prevent memory leaks on mobile GPUs
    const probabilities = window.tf.tidy(() => {
      // Convert flat array to 2D tensor of shape [1, 63]
      const inputTensor = window.tf.tensor2d([normalizedCoords], [1, 63], 'float32');
      const outputTensor = model.predict(inputTensor);
      return outputTensor.dataSync(); // Synchronous array extract
    });

    // Find class with highest probability (argmax)
    let maxIdx = 0;
    let maxProb = probabilities[0];
    for (let i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > maxProb) {
        maxProb = probabilities[i];
        maxIdx = i;
      }
    }

    const predictedLabel = labels[maxIdx] || 'Unknown';
    const confidenceScore = maxProb; // 0.0 to 1.0

    return {
      label: predictedLabel,
      confidence: confidenceScore,
    };
  }

  /**
   * Runs prediction every INFERENCE_INTERVAL_MS using getCurrentLandmarks().
   */
  function startInferenceLoop() {
    if (inferenceIntervalId) return;

    inferenceIntervalId = setInterval(() => {
      if (!isModelReady || typeof window.getCurrentLandmarks !== 'function') {
        latestPrediction = null;
        notifyListeners(null);
        updateHandednessBadge(null);
        return;
      }

      const landmarks = window.getCurrentLandmarks();
      if (!landmarks) {
        latestPrediction = null;
        notifyListeners(null);
        updateHandednessBadge(null);
        return;
      }

      // Retrieve MediaPipe handedness (camera perspective: "Left" or "Right")
      const handedness = typeof window.getCurrentHandedness === 'function'
        ? window.getCurrentHandedness()
        : null;

      // Apply left-hand mirroring if needed (see mirrorLandmarksIfLeft for full rationale)
      const processedLandmarks = handedness === 'Left'
        ? mirrorLandmarksIfLeft(landmarks)
        : landmarks;

      // Update the on-screen handedness badge for debugging / demo transparency
      updateHandednessBadge(handedness);

      const result = predictSign(processedLandmarks);
      // Attach handedness to the prediction result so consumers can read it
      if (result) result.handedness = handedness;
      latestPrediction = result;
      notifyListeners(result);
    }, INFERENCE_INTERVAL_MS);
  }

  /**
   * Stops the inference loop.
   */
  function stopInferenceLoop() {
    if (inferenceIntervalId) {
      clearInterval(inferenceIntervalId);
      inferenceIntervalId = null;
    }
    latestPrediction = null;
    notifyListeners(null);
  }

  /**
   * Registers a callback listener for new predictions.
   *
   * @param {Function} callback - Function receiving ({ label, confidence } | null)
   * @returns {Function} Unsubscribe function
   */
  function onPrediction(callback) {
    if (typeof callback === 'function') {
      listeners.push(callback);
    }
    return () => {
      const idx = listeners.indexOf(callback);
      if (idx !== -1) listeners.splice(idx, 1);
    };
  }

  function notifyListeners(prediction) {
    for (let i = 0; i < listeners.length; i++) {
      try {
        listeners[i](prediction);
      } catch (e) {
        console.error('[Classifier] Listener error:', e);
      }
    }
  }

  /**
   * Updates the on-screen handedness badge element with the current hand label.
   * Renders "Right Hand" or "Left Hand" with distinct styling for demo clarity.
   * Hides the badge when no hand is detected.
   *
   * @param {"Left"|"Right"|null} handedness
   */
  function updateHandednessBadge(handedness) {
    const badge = document.getElementById('handednessBadge');
    if (!badge) return;

    if (!handedness) {
      badge.textContent = '';
      badge.className = 'handedness-badge hidden';
      return;
    }

    // MediaPipe "Left" from camera perspective = user's right hand (and vice versa).
    // We display from the USER's perspective for intuitive feedback.
    const userLabel = handedness === 'Left' ? 'Right Hand' : 'Left Hand';
    const isMirrored = handedness === 'Left'; // left-hand mirroring was applied

    badge.textContent = userLabel + (isMirrored ? ' (mirrored)' : '');
    badge.className = 'handedness-badge ' + (isMirrored ? 'badge-left' : 'badge-right');
  }

  /**
   * Returns the latest predicted sign and confidence score.
   *
   * @returns {{ label: string, confidence: number, handedness: string|null }|null}
   */
  function getCurrentPrediction() {
    return latestPrediction;
  }

  // Export to global window object
  window.initClassifier = initClassifier;
  window.predictSign = predictSign;
  window.getCurrentPrediction = getCurrentPrediction;
  window.onPrediction = onPrediction;
  window.startInferenceLoop = startInferenceLoop;
  window.stopInferenceLoop = stopInferenceLoop;
})(window);
