/**
 * app.js
 * ------
 * Application entry point for Checkpoint 4 (Live captions, debounced sentence building, and gesture classification).
 */

document.addEventListener("DOMContentLoaded", async () => {
  // -------------------------------------------------------------------------
  // DOM Elements
  // -------------------------------------------------------------------------
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

  // Prediction HUD Elements (Camera Overlay)
  const hudSignText = document.getElementById("hudSignText");
  const hudConfidenceText = document.getElementById("hudConfidenceText");

  // Live Prediction Debug Card Elements
  const liveSignSymbol = document.getElementById("liveSignSymbol");
  const liveSignLabel = document.getElementById("liveSignLabel");
  const confidencePercent = document.getElementById("confidencePercent");
  const confidenceBarFill = document.getElementById("confidenceBarFill");

  // Live Sentence Builder & Caption Elements
  const sentenceText = document.getElementById("sentenceText");
  const captionDisplayBox = document.getElementById("captionDisplayBox");
  const stabilityText = document.getElementById("stabilityText");
  const stabilityBarFill = document.getElementById("stabilityBarFill");
  const clearSentenceBtn = document.getElementById("clearSentenceBtn");
  const backspaceBtn = document.getElementById("backspaceBtn");
  const addSpaceBtn = document.getElementById("addSpaceBtn");
  const copySentenceBtn = document.getElementById("copySentenceBtn");
  const copyBtnLabel = document.getElementById("copyBtnLabel");
  const speakSentenceBtn = document.getElementById("speakSentenceBtn");
  const speakBtnLabel = document.getElementById("speakBtnLabel");

  let landmarksInitialized = false;

  // -------------------------------------------------------------------------
  // 1. Camera Manager Setup
  // -------------------------------------------------------------------------
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

  // -------------------------------------------------------------------------
  // 2. Sentence Builder UI Listeners
  // -------------------------------------------------------------------------
  if (window.SentenceBuilder) {
    // A. Sentence content update listener
    window.SentenceBuilder.onSentenceChange((sentence) => {
      if (!sentenceText) return;

      if (sentence && sentence.length > 0) {
        sentenceText.textContent = sentence;
        sentenceText.classList.remove("empty");
      } else {
        sentenceText.textContent = "Begin signing to generate captions...";
        sentenceText.classList.add("empty");
      }

      // Auto-scroll to the bottom of the caption display box as text grows
      if (captionDisplayBox) {
        captionDisplayBox.scrollTop = captionDisplayBox.scrollHeight;
      }
    });

    // B. Stability progress / hold-to-lock listener
    window.SentenceBuilder.onStabilityChange((data) => {
      if (!stabilityText || !stabilityBarFill) return;

      if (data.isCommitted) {
        stabilityText.textContent = `Locked: '${data.label}'`;
        stabilityBarFill.style.width = "100%";
        stabilityBarFill.classList.add("locked");
      } else if (data.label && data.count > 0) {
        stabilityText.textContent = `Holding '${data.label}' (${data.count}/${data.target})`;
        stabilityBarFill.style.width = `${data.percent}%`;
        stabilityBarFill.classList.remove("locked");
      } else {
        stabilityText.textContent = "Hold sign to lock";
        stabilityBarFill.style.width = "0%";
        stabilityBarFill.classList.remove("locked");
      }
    });
  }

  // -------------------------------------------------------------------------
  // 3. TensorFlow.js Gesture Classifier Setup
  // -------------------------------------------------------------------------
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

    // Feed live predictions into sentenceBuilder & update HUD
    window.onPrediction((prediction) => {
      // 1. Process prediction through sentence debounce engine
      if (window.SentenceBuilder) {
        window.SentenceBuilder.processPrediction(prediction);
      }

      // 2. Update real-time debug HUD & prediction card
      if (prediction && prediction.confidence >= 0.25) {
        const sign = prediction.label;
        const confPct = Math.round(prediction.confidence * 100);

        if (hudSignText) hudSignText.textContent = sign;
        if (hudConfidenceText) hudConfidenceText.textContent = `${confPct}%`;

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
        // Hand absent or low confidence
        if (hudSignText) hudSignText.textContent = "--";
        if (hudConfidenceText) hudConfidenceText.textContent = "0%";

        if (liveSignSymbol) liveSignSymbol.textContent = "--";
        if (liveSignLabel) liveSignLabel.textContent = "Waiting for hand...";
        if (confidencePercent) confidencePercent.textContent = "0%";
        if (confidenceBarFill) confidenceBarFill.style.width = "0%";
      }
    });
  }

  // -------------------------------------------------------------------------
  // 4. Status Indicator Poller
  // -------------------------------------------------------------------------
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

  // -------------------------------------------------------------------------
  // 5. Button Event Listeners
  // -------------------------------------------------------------------------
  retryBtn.addEventListener("click", () => {
    camera.start().catch(() => {});
  });

  switchCameraBtn.addEventListener("click", () => {
    camera.switchCamera().catch(() => {});
  });

  // Caption Toolbar Actions
  if (clearSentenceBtn) {
    clearSentenceBtn.addEventListener("click", () => {
      if (window.SentenceBuilder) window.SentenceBuilder.resetSentence();
    });
  }

  if (backspaceBtn) {
    backspaceBtn.addEventListener("click", () => {
      if (window.SentenceBuilder) window.SentenceBuilder.deleteLastChar();
    });
  }

  if (addSpaceBtn) {
    addSpaceBtn.addEventListener("click", () => {
      if (window.SentenceBuilder) window.SentenceBuilder.addSpace();
    });
  }

  if (speakSentenceBtn) {
    speakSentenceBtn.addEventListener("click", () => {
      if (!window.SpeechEngine) return;

      const sentence = window.SentenceBuilder ? window.SentenceBuilder.getCurrentSentence() : "";

      if (!window.SpeechEngine.isSupported()) {
        if (speakBtnLabel) {
          const original = speakBtnLabel.textContent;
          speakBtnLabel.textContent = "Not Supported";
          setTimeout(() => (speakBtnLabel.textContent = original), 2000);
        }
        return;
      }

      if (!sentence || !sentence.trim()) {
        if (speakBtnLabel) {
          const original = speakBtnLabel.textContent;
          speakBtnLabel.textContent = "No Text!";
          setTimeout(() => (speakBtnLabel.textContent = original), 1500);
        }
        return;
      }

      window.SpeechEngine.speak(sentence, {
        onStart: () => {
          if (speakBtnLabel) speakBtnLabel.textContent = "Speaking...";
          speakSentenceBtn.classList.add("speaking");
        },
        onEnd: () => {
          if (speakBtnLabel) speakBtnLabel.textContent = "Speak";
          speakSentenceBtn.classList.remove("speaking");
        },
        onError: (err) => {
          console.warn("[App] Speech synthesis error:", err);
          if (speakBtnLabel) speakBtnLabel.textContent = "Speak";
          speakSentenceBtn.classList.remove("speaking");
        },
      });
    });
  }

  if (copySentenceBtn) {
    copySentenceBtn.addEventListener("click", async () => {
      const text = window.SentenceBuilder ? window.SentenceBuilder.getCurrentSentence() : "";
      if (!text) return;

      try {
        await navigator.clipboard.writeText(text);
        if (copyBtnLabel) {
          const originalText = copyBtnLabel.textContent;
          copyBtnLabel.textContent = "Copied!";
          setTimeout(() => {
            copyBtnLabel.textContent = originalText;
          }, 1500);
        }
      } catch (err) {
        console.warn("[App] Failed to copy to clipboard:", err);
      }
    });
  }

  // -------------------------------------------------------------------------
  // 6. Automatically start camera on page load
  // -------------------------------------------------------------------------
  try {
    await camera.start();
  } catch (e) {
    // Handled in camera.js onStatusChange
  }
});
