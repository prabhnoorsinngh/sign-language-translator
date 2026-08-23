/**
 * classifier.js
 * -------------
 * TensorFlow.js sign language inference engine.
 *
 * Responsibilities:
 * 1. Load the exported Keras Sequential model from `model/model.json`.
 * 2. Load `model/labels.json` class mapping (28 classes: A-Z, del, space).
 * 3. Poll `getCurrentLandmarks()` from `landmarks.js` every ~200ms.
 * 4. Run inference using `tf.tidy()` to prevent GPU/WebGL memory leaks.
 * 5. Normalization note:
 *    As verified in `python-training/convert_to_tfjs.py` and `process_dataset.py`,
 *    the model was trained directly on raw MediaPipe normalized coordinates
 *    (x in [0, 1], y in [0, 1], z relative depth) without additional scaling.
 *    `getCurrentLandmarks()` outputs this exact 63-element feature vector.
 * 6. Expose `getCurrentPrediction()` returning `{ label, confidence }` or `null`.
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
   * Executes a single classification pass on the provided 63-element landmark vector.
   *
   * @param {Float32Array|number[]} landmarks - 63 landmark coordinates [x0,y0,z0,...,x20,y20,z20]
   * @returns {{ label: string, confidence: number }|null}
   */
  function predictSign(landmarks) {
    if (!model || !isModelReady || !landmarks || landmarks.length !== 63) {
      return null;
    }

    // Wrap tensor operations in tf.tidy() to prevent memory leaks on mobile GPUs
    const probabilities = window.tf.tidy(() => {
      // Convert flat array to 2D tensor of shape [1, 63]
      const inputTensor = window.tf.tensor2d([landmarks], [1, 63], 'float32');
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
        return;
      }

      const landmarks = window.getCurrentLandmarks();
      if (!landmarks) {
        latestPrediction = null;
        notifyListeners(null);
        return;
      }

      const result = predictSign(landmarks);
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
   * Returns the latest predicted sign and confidence score.
   *
   * @returns {{ label: string, confidence: number }|null}
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
