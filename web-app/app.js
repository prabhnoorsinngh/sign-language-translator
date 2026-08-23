/**
 * app.js
 * ------
 * Application entry point for Checkpoint 3 (Live gesture classification & UI bindings).
 */

document.addEventListener("DOMContentLoaded", async () => {
  // DOM Elements
  const video = document.getElementById("webcam");
  const canvas = document.getElementById("outputCanvas");
  const statusOverlay = document.getElementById("statusOverlay");
  const statusSpinner = document.getElementById("statusSpinner");
  const statusMessage = document.getElementById("statusMessage");
  const retryBtn = document.getElementById("retryBtn");
  const switchCameraBtn = document.getElementById("switchCameraBtn");

  // Status Badges
  const cameraStatusBadge = document.getElementById("cameraStatusBadge");
  const trackingStatusBadge = document.getElementById("trackingStatusBadge");
  const modelStatusBadge = document.getElementById("modelStatusBadge");
  const resolutionBadge = document.getElementById("resolutionBadge");

  // Prediction HUD Elements (on Camera Viewport)
  const hudSignText = document.getElementById("hudSignText");
  const hudConfidenceText = document.getElementById("hudConfidenceText");

  // Dedicated Live Prediction Card Elements
  const liveSignSymbol = document.getElementById("liveSignSymbol");
  const liveSignLabel = document.getElementById("liveSignLabel");
  const confidencePercent = document.getElementById("confidencePercent");
  const confidenceBarFill = document.getElementById("confidenceBarFill");

  let landmarksInitialized = false;

  // 1. Create and configure CameraManager instance
  const camera = new CameraManager(video, canvas, {
    facingMode: "environment", // prefer rear camera on mobile
    onStatusChange: (status, message) => {
      switch (status) {
        case "loading":
          statusOverlay.classList.remove("hidden");
          statusSpinner.classList.remove("hidden");
          retryBtn.classList.add("hidden");
          statusMessage.textContent = message;
          cameraStatusBadge.className = "badge badge-warning";
          cameraStatusBadge.textContent = "Connecting...";
          break;

        case "active":
          statusOverlay.classList.add("hidden");
          cameraStatusBadge.className = "badge badge-success";
          cameraStatusBadge.textContent = "Live";

          // Initialize MediaPipe hand tracking once video is live
          if (!landmarksInitialized && typeof window.initLandmarks === "function") {
            window.initLandmarks(video, canvas);
            landmarksInitialized = true;
          }
          break;

        case "error":
          statusOverlay.classList.remove("hidden");
          statusSpinner.classList.add("hidden");
          retryBtn.classList.remove("hidden");
          statusMessage.textContent = message;
          cameraStatusBadge.className = "badge badge-danger";
          cameraStatusBadge.textContent = "Error";
          break;
      }
    },
    onResolutionChange: (resolutionStr) => {
      resolutionBadge.textContent = resolutionStr;
      resolutionBadge.className = "badge badge-neutral";
    },
  });

  // 2. Initialize TensorFlow.js Gesture Classifier
  if (typeof window.initClassifier === "function") {
    window.initClassifier({
      onStatusChange: (status, msg) => {
        if (!modelStatusBadge) return;
        switch (status) {
          case "loading":
            modelStatusBadge.className = "badge badge-warning";
            modelStatusBadge.textContent = "Loading Model...";
            break;
          case "ready":
            modelStatusBadge.className = "badge badge-success";
            modelStatusBadge.textContent = msg || "Model Ready";
            break;
          case "error":
            modelStatusBadge.className = "badge badge-danger";
            modelStatusBadge.textContent = "Model Error";
            break;
        }
      },
    }).catch((err) => {
      console.error("[App] Classifier initialization error:", err);
    });

    // 3. Listen for live inference results (~200ms)
    window.onPrediction((prediction) => {
      if (prediction && prediction.confidence >= 0.25) {
        const sign = prediction.label;
        const confPct = Math.round(prediction.confidence * 100);

        // Update Camera HUD
        if (hudSignText) hudSignText.textContent = sign;
        if (hudConfidenceText) hudConfidenceText.textContent = `${confPct}%`;

        // Update Live Prediction Card
        if (liveSignSymbol) liveSignSymbol.textContent = sign === "space" ? "␣" : sign === "del" ? "⌫" : sign;
        if (liveSignLabel) liveSignLabel.textContent = sign === "space" ? "Space (␣)" : sign === "del" ? "Delete (⌫)" : `Letter ${sign}`;
        if (confidencePercent) confidencePercent.textContent = `${confPct}%`;
        if (confidenceBarFill) {
          confidenceBarFill.style.width = `${confPct}%`;
          if (confPct >= 80) {
            confidenceBarFill.style.background = "linear-gradient(90deg, #10b981, #059669)";
          } else if (confPct >= 50) {
            confidenceBarFill.style.background = "linear-gradient(90deg, #3b82f6, #10b981)";
          } else {
            confidenceBarFill.style.background = "linear-gradient(90deg, #f59e0b, #ef4444)";
          }
        }
      } else {
        // No hand or low confidence
        if (hudSignText) hudSignText.textContent = "--";
        if (hudConfidenceText) hudConfidenceText.textContent = "0%";

        if (liveSignSymbol) liveSignSymbol.textContent = "--";
        if (liveSignLabel) liveSignLabel.textContent = "Waiting for hand...";
        if (confidencePercent) confidencePercent.textContent = "0%";
        if (confidenceBarFill) confidenceBarFill.style.width = "0%";
      }
    });
  }

  // 4. Periodically update Hand Tracking badge
  setInterval(() => {
    if (!trackingStatusBadge || typeof window.getCurrentLandmarks !== "function") return;
    const landmarks = window.getCurrentLandmarks();
    if (landmarks) {
      trackingStatusBadge.className = "badge badge-success";
      trackingStatusBadge.textContent = "Hand Tracked (21 pts)";
    } else {
      trackingStatusBadge.className = "badge badge-neutral";
      trackingStatusBadge.textContent = "Searching for hand...";
    }
  }, 200);

  // 5. Button Event Listeners
  retryBtn.addEventListener("click", () => {
    camera.start().catch(() => {});
  });

  switchCameraBtn.addEventListener("click", () => {
    camera.switchCamera().catch(() => {});
  });

  // 6. Automatically start camera on page load
  try {
    await camera.start();
  } catch (e) {
    // Handled in camera.js onStatusChange
  }
});
