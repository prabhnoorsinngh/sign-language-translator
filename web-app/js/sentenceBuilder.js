/**
 * sentenceBuilder.js
 * ------------------
 * Manages sign gesture debouncing, stability verification, and sentence accumulation.
 *
 * Responsibilities:
 * 1. Tracks consecutive prediction frames for stability (requires 15 consecutive
 *    matching predictions (~3 seconds @ 200ms) before committing).
 * 2. Filters out transient flickers and false positives.
 * 3. Commits characters to a running sentence:
 *    - "space" -> adds a single whitespace ' '
 *    - "del"   -> deletes the last character
 *    - "A"-"Z" -> appends the letter
 * 4. Lock mechanism: Prevents the same letter from duplicate spamming when held continuously
 *    until the hand is relaxed/lost or a new sign is formed.
 * 5. Exposes public API:
 *    - getCurrentSentence()
 *    - resetSentence()
 *    - deleteLastChar()
 *    - addSpace()
 *    - onSentenceChange(callback)
 *    - onStabilityChange(callback)
 */

(function (window) {
  'use strict';

  // Configuration Constants
  const REQUIRED_CONSECUTIVE_CHECKS = 15; // 15 checks @ ~200ms = ~3.0s hold time
  const MIN_CONFIDENCE_THRESHOLD = 0.5;   // Ignore predictions with low confidence

  // State
  let currentSentence = '';
  let candidateLabel = null;
  let consecutiveCount = 0;
  let isCommittedLock = false; // Lock active once a sign has been committed until hand is released/changed
  let lastCommittedLabel = null;

  // Listeners
  const sentenceListeners = [];
  const stabilityListeners = [];

  /**
   * Processes a new prediction from classifier.js.
   *
   * @param {{ label: string, confidence: number }|null} prediction
   */
  function processPrediction(prediction) {
    // 1. Hand Lost or Low Confidence -> Reset candidate tracking and unlock commit
    if (!prediction || prediction.confidence < MIN_CONFIDENCE_THRESHOLD) {
      candidateLabel = null;
      consecutiveCount = 0;
      isCommittedLock = false; // Unlocks so the same sign can be made again after releasing hand
      notifyStabilityChange({
        label: null,
        count: 0,
        target: REQUIRED_CONSECUTIVE_CHECKS,
        percent: 0,
        isCommitted: false,
      });
      return;
    }

    const label = prediction.label;

    // 2. Candidate Label Matching & Stability Counting
    if (label === candidateLabel) {
      consecutiveCount++;
    } else {
      // Switched to a new sign -> restart stability count
      candidateLabel = label;
      consecutiveCount = 1;
      isCommittedLock = false;
    }

    const progressPct = Math.min(100, Math.round((consecutiveCount / REQUIRED_CONSECUTIVE_CHECKS) * 100));

    // 3. Threshold Check: Commit sign when stability reaches 15 frames and not locked
    if (consecutiveCount >= REQUIRED_CONSECUTIVE_CHECKS && !isCommittedLock) {
      commitSign(candidateLabel);
      isCommittedLock = true; // Lock until sign changes or hand is removed
      lastCommittedLabel = candidateLabel;
    }

    // 4. Notify UI of current stability progress
    notifyStabilityChange({
      label: candidateLabel,
      count: consecutiveCount,
      target: REQUIRED_CONSECUTIVE_CHECKS,
      percent: progressPct,
      isCommitted: isCommittedLock,
    });
  }

  /**
   * Commits the accepted sign gesture to the running sentence.
   *
   * @param {string} sign - The accepted sign label (e.g. 'A', 'space', 'del')
   */
  function commitSign(sign) {
    if (!sign) return;

    if (sign === 'space') {
      // Don't add duplicate leading or consecutive spaces
      if (currentSentence.length > 0 && !currentSentence.endsWith(' ')) {
        currentSentence += ' ';
      }
    } else if (sign === 'del') {
      // Backspace / Delete last character
      if (currentSentence.length > 0) {
        currentSentence = currentSentence.slice(0, -1);
      }
    } else {
      // Standard letter (A-Z)
      // Prevent immediate duplicate character if the last char matches and user didn't reset
      currentSentence += sign;
    }

    console.log(`[SentenceBuilder] Committed sign: '${sign}' -> Full sentence: "${currentSentence}"`);
    notifySentenceChange(currentSentence, sign);
  }

  /**
   * Returns the current running sentence.
   * @returns {string}
   */
  function getCurrentSentence() {
    return currentSentence;
  }

  /**
   * Resets and clears the current sentence.
   */
  function resetSentence() {
    currentSentence = '';
    candidateLabel = null;
    consecutiveCount = 0;
    isCommittedLock = false;
    lastCommittedLabel = null;
    notifySentenceChange(currentSentence, null);
    notifyStabilityChange({
      label: null,
      count: 0,
      target: REQUIRED_CONSECUTIVE_CHECKS,
      percent: 0,
      isCommitted: false,
    });
  }

  /**
   * Manually deletes the last character.
   */
  function deleteLastChar() {
    if (currentSentence.length > 0) {
      currentSentence = currentSentence.slice(0, -1);
      notifySentenceChange(currentSentence, 'del');
    }
  }

  /**
   * Manually appends a space character.
   */
  function addSpace() {
    if (currentSentence.length > 0 && !currentSentence.endsWith(' ')) {
      currentSentence += ' ';
      notifySentenceChange(currentSentence, 'space');
    }
  }

  /**
   * Subscribes to sentence changes.
   * @param {Function} callback - Function(sentence: string, committedSign: string|null)
   * @returns {Function} Unsubscribe function
   */
  function onSentenceChange(callback) {
    if (typeof callback === 'function') {
      sentenceListeners.push(callback);
    }
    return () => {
      const idx = sentenceListeners.indexOf(callback);
      if (idx !== -1) sentenceListeners.splice(idx, 1);
    };
  }

  /**
   * Subscribes to stability/lock progress updates.
   * @param {Function} callback - Function({ label, count, target, percent, isCommitted })
   * @returns {Function} Unsubscribe function
   */
  function onStabilityChange(callback) {
    if (typeof callback === 'function') {
      stabilityListeners.push(callback);
    }
    return () => {
      const idx = stabilityListeners.indexOf(callback);
      if (idx !== -1) stabilityListeners.splice(idx, 1);
    };
  }

  function notifySentenceChange(sentence, lastSign) {
    for (let i = 0; i < sentenceListeners.length; i++) {
      try {
        sentenceListeners[i](sentence, lastSign);
      } catch (e) {
        console.error('[SentenceBuilder] SentenceListener error:', e);
      }
    }
  }

  function notifyStabilityChange(data) {
    for (let i = 0; i < stabilityListeners.length; i++) {
      try {
        stabilityListeners[i](data);
      } catch (e) {
        console.error('[SentenceBuilder] StabilityListener error:', e);
      }
    }
  }

  // Export to global window object
  window.SentenceBuilder = {
    processPrediction,
    getCurrentSentence,
    resetSentence,
    deleteLastChar,
    addSpace,
    onSentenceChange,
    onStabilityChange,
  };

  // Direct global helpers as requested
  window.getCurrentSentence = getCurrentSentence;
  window.resetSentence = resetSentence;
})(window);
