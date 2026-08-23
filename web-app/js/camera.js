/**
 * camera.js
 * ---------
 * Manages webcam access, rear/front camera selection, stream lifecycle,
 * canvas dimension synchronization, and user permission error handling.
 */

class CameraManager {
  constructor(videoElement, canvasElement, options = {}) {
    this.video = videoElement;
    this.canvas = canvasElement;
    this.currentFacingMode = options.facingMode || "environment"; // default to rear camera on mobile
    this.stream = null;
    this.onStatusChange = options.onStatusChange || (() => {});
    this.onResolutionChange = options.onResolutionChange || (() => {});
  }

  /**
   * Initializes camera stream with rear camera preferred, falling back to any available camera.
   */
  async start() {
    this.stop(); // Stop any active stream before starting a new one

    // 1. Verify Browser Support & Secure Context
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      const isHttps = window.location.protocol === "https:" || window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
      const errorMsg = isHttps
        ? "Your browser does not support camera streaming via navigator.mediaDevices.getUserMedia."
        : "Camera access requires HTTPS or localhost. If accessing over a local Wi-Fi IP (e.g. http://192.168.x.x), please use HTTPS or enable Chrome's unsafely-treat-insecure-origin-as-secure flag in chrome://flags.";
      
      this.onStatusChange("error", errorMsg);
      throw new Error(errorMsg);
    }

    this.onStatusChange("loading", "Requesting camera access...");

    // 2. Try rear camera (environment) first
    try {
      const constraints = {
        video: {
          facingMode: { ideal: this.currentFacingMode },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      };

      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (primaryErr) {
      console.warn(`[CameraManager] Failed with facingMode='${this.currentFacingMode}', falling back to default camera:`, primaryErr);

      // 3. Fallback to basic generic video constraint if rear camera constraint fails
      try {
        this.stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      } catch (fallbackErr) {
        return this.handleError(fallbackErr);
      }
    }

    // 4. Attach stream to <video> element
    try {
      this.video.srcObject = this.stream;

      // Wait for video metadata (dimensions) to load
      await new Promise((resolve) => {
        if (this.video.readyState >= 2) {
          resolve();
        } else {
          this.video.onloadedmetadata = () => resolve();
        }
      });

      // Play video (required for mobile autoplay)
      await this.video.play();

      // Sync canvas dimensions to match video stream
      this.syncCanvasDimensions();
      window.addEventListener("resize", () => this.syncCanvasDimensions());

      const resText = `${this.video.videoWidth} x ${this.video.videoHeight}`;
      this.onResolutionChange(resText);
      this.onStatusChange("active", "Camera connected");
      return this.stream;
    } catch (playErr) {
      return this.handleError(playErr);
    }
  }

  /**
   * Resizes canvas internal buffer to exactly match the video's stream resolution.
   */
  syncCanvasDimensions() {
    if (this.video.videoWidth && this.video.videoHeight && this.canvas) {
      this.canvas.width = this.video.videoWidth;
      this.canvas.height = this.video.videoHeight;
    }
  }

  /**
   * Toggles between rear ('environment') and front ('user') camera.
   */
  async switchCamera() {
    this.currentFacingMode = this.currentFacingMode === "environment" ? "user" : "environment";
    return this.start();
  }

  /**
   * Stops active camera streams and releases hardware.
   */
  stop() {
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
    if (this.video) {
      this.video.srcObject = null;
    }
  }

  /**
   * Formats and relays human-readable error messages for UI presentation.
   */
  handleError(err) {
    let message = "Could not access camera.";

    switch (err.name) {
      case "NotAllowedError":
      case "PermissionDeniedError":
        message = "Camera permission was denied. Please allow camera permissions in your browser address bar or settings, then click Retry.";
        break;
      case "NotFoundError":
      case "DevicesNotFoundError":
        message = "No camera device was found on this device.";
        break;
      case "NotReadableError":
      case "TrackStartError":
        message = "Camera is currently in use by another application or tab.";
        break;
      case "OverconstrainedError":
      case "ConstraintNotSatisfiedError":
        message = "The requested camera settings are not supported by your device.";
        break;
      default:
        message = err.message || message;
    }

    console.error("[CameraManager] Error:", err);
    this.onStatusChange("error", message);
    throw err;
  }
}

// Attach to window for global access across vanilla JS modules
window.CameraManager = CameraManager;
