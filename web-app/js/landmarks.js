/**
 * landmarks.js
 * ------------
 * Real-time hand landmark detection and skeleton overlay rendering
 * using MediaPipe Hands (via browser CDN).
 *
 * Responsibilities:
 * 1. Initialize the MediaPipe Hands solution with CDN assets.
 * 2. Continuously process video frames from camera.js using requestAnimationFrame.
 * 3. Render 21 hand landmarks and connection lines on the overlay canvas.
 * 4. Expose `getCurrentLandmarks()` returning a flat 63-element array [x0, y0, z0, ..., x20, y20, z20]
 *    for the primary detected hand (or null if no hands in frame).
 */

(function (window) {
  'use strict';

  // Stores the latest detected 63-element landmark coordinates (or null when no hand)
  let latestLandmarks = null;

  // MediaPipe and canvas tracking state
  let hands = null;
  let videoElement = null;
  let canvasElement = null;
  let canvasCtx = null;
  let isTracking = false;
  let isProcessingFrame = false;

  /**
   * Initializes MediaPipe Hands detector and binds to the video and canvas elements.
   *
   * @param {HTMLVideoElement} video - The source webcam video element
   * @param {HTMLCanvasElement} canvas - The overlay canvas element
   * @param {Object} options - Optional overrides for confidence thresholds and complexity
   */
  function initLandmarks(video, canvas, options = {}) {
    videoElement = video;
    canvasElement = canvas;
    canvasCtx = canvas.getContext('2d');

    // 1. Verify MediaPipe library availability
    if (typeof window.Hands === 'undefined') {
      console.error(
        '[Landmarks] MediaPipe Hands library not loaded. Make sure @mediapipe/hands script is included in index.html.'
      );
      return null;
    }

    // 2. Instantiate MediaPipe Hands solution
    // locateFile downloads WASM binaries and models dynamically from CDN
    hands = new window.Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });

    // 3. Set MediaPipe Hands options
    hands.setOptions({
      maxNumHands: options.maxNumHands || 2, // Track up to 2 hands (primary hand extracted for inference)
      modelComplexity: options.modelComplexity !== undefined ? options.modelComplexity : 1, // 0 = Lite (fastest), 1 = Full (accurate)
      minDetectionConfidence: options.minDetectionConfidence || 0.5, // Threshold to initiate hand detection
      minTrackingConfidence: options.minTrackingConfidence || 0.5, // Threshold to maintain tracking across frames
    });

    // 4. Register frame results callback
    hands.onResults(onHandResults);

    console.log('[Landmarks] MediaPipe Hands initialized successfully.');
    startTracking();

    return hands;
  }

  /**
   * Callback invoked whenever MediaPipe Hands finishes processing a frame.
   * Clears canvas, renders hand skeleton overlay, and updates `latestLandmarks`.
   *
   * @param {Object} results - MediaPipe detection results containing `multiHandLandmarks`
   */
  function onHandResults(results) {
    if (!canvasCtx || !canvasElement) return;

    // Clear previous frame overlay
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    const detectedHands = results.multiHandLandmarks;

    if (detectedHands && detectedHands.length > 0) {
      // 1. Extract 63 flat coordinate features (x, y, z for all 21 landmarks) from the primary hand
      const primaryHand = detectedHands[0];
      const flatCoords = new Float32Array(63);

      for (let i = 0; i < 21; i++) {
        const landmark = primaryHand[i];
        const offset = i * 3;
        flatCoords[offset] = landmark.x;
        flatCoords[offset + 1] = landmark.y;
        flatCoords[offset + 2] = landmark.z;
      }

      latestLandmarks = flatCoords;

      // 2. Draw visual hand skeleton overlay on canvas
      for (let h = 0; h < detectedHands.length; h++) {
        const handLandmarks = detectedHands[h];

        // Draw connections / bones (neon cyan with glow)
        if (window.drawConnectors && window.HAND_CONNECTIONS) {
          window.drawConnectors(canvasCtx, handLandmarks, window.HAND_CONNECTIONS, {
            color: h === 0 ? '#00e5ff' : '#a855f7', // Primary hand = Cyan, Secondary = Purple
            lineWidth: 3,
          });
        }

        // Draw landmark joints / keypoints (glowing coral/white)
        if (window.drawLandmarks) {
          window.drawLandmarks(canvasCtx, handLandmarks, {
            color: '#ffffff',
            fillColor: h === 0 ? '#ff3b80' : '#c084fc',
            lineWidth: 1.5,
            radius: 4,
          });
        }
      }
    } else {
      // No hands detected in current frame
      latestLandmarks = null;
    }

    canvasCtx.restore();
  }

  /**
   * Animation frame loop that continuously feeds video frames into MediaPipe.
   */
  async function processVideoFrame() {
    if (!isTracking) return;

    if (
      videoElement &&
      videoElement.readyState >= 2 &&
      !videoElement.paused &&
      !videoElement.ended &&
      hands &&
      !isProcessingFrame
    ) {
      isProcessingFrame = true;
      try {
        await hands.send({ image: videoElement });
      } catch (err) {
        console.warn('[Landmarks] Error sending frame to MediaPipe Hands:', err);
      } finally {
        isProcessingFrame = false;
      }
    }

    // Schedule next frame
    requestAnimationFrame(processVideoFrame);
  }

  /**
   * Starts the landmark detection loop.
   */
  function startTracking() {
    if (isTracking) return;
    isTracking = true;
    requestAnimationFrame(processVideoFrame);
  }

  /**
   * Stops the landmark detection loop and clears overlay.
   */
  function stopTracking() {
    isTracking = false;
    latestLandmarks = null;
    if (canvasCtx && canvasElement) {
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    }
  }

  /**
   * Returns the latest detected hand landmarks as a flat array of 63 numbers:
   * [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]
   *
   * @returns {Float32Array|null} 63 normalized coordinates, or null if no hand detected.
   */
  function getCurrentLandmarks() {
    return latestLandmarks;
  }

  // Export functions to global scope
  window.initLandmarks = initLandmarks;
  window.startLandmarksTracking = startTracking;
  window.stopLandmarksTracking = stopTracking;
  window.getCurrentLandmarks = getCurrentLandmarks;
})(window);
