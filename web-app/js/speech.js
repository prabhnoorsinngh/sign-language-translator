/**
 * speech.js
 * ---------
 * Text-to-Speech (TTS) engine using the browser's native Web Speech API
 * (window.speechSynthesis & SpeechSynthesisUtterance).
 *
 * Responsibilities:
 * 1. Checks for Web Speech API browser compatibility.
 * 2. Fetches and configures natural English voices.
 * 3. Vocalizes the accumulated sentence aloud in English.
 * 4. Manages speech playback states (start, end, error, cancel).
 * 5. Handles empty sentence inputs and unsupported browsers gracefully.
 */

(function (window) {
  'use strict';

  const isSpeechSupported = 'speechSynthesis' in window && 'SpeechSynthesisUtterance' in window;
  let availableVoices = [];
  let preferredVoice = null;
  let isCurrentlySpeaking = false;

  /**
   * Initializes and caches the list of available TTS voices.
   */
  function loadVoices() {
    if (!isSpeechSupported) return;

    availableVoices = window.speechSynthesis.getVoices();

    // Select the best natural-sounding English voice available
    preferredVoice =
      availableVoices.find((v) => v.lang.startsWith('en') && (v.name.includes('Google') || v.name.includes('Natural') || v.name.includes('Samantha') || v.name.includes('Karen'))) ||
      availableVoices.find((v) => v.lang.startsWith('en')) ||
      availableVoices[0] ||
      null;
  }

  if (isSpeechSupported) {
    loadVoices();
    if (window.speechSynthesis.onvoiceschanged !== undefined) {
      window.speechSynthesis.onvoiceschanged = loadVoices;
    }
  } else {
    console.warn('[Speech] Web Speech API (SpeechSynthesis) is not supported in this browser.');
  }

  /**
   * Speaks a provided text string aloud.
   *
   * @param {string} text - The sentence to vocalize.
   * @param {Object} [options] - Optional configurations.
   * @param {number} [options.rate=1.0] - Speech rate (0.1 to 10).
   * @param {number} [options.pitch=1.0] - Speech pitch (0 to 2).
   * @param {Function} [options.onStart] - Callback when speech begins.
   * @param {Function} [options.onEnd] - Callback when speech completes.
   * @param {Function} [options.onError] - Callback if speech fails.
   * @returns {{ success: boolean, reason?: string }} Status object
   */
  function speak(text, options = {}) {
    if (!isSpeechSupported) {
      const err = 'Web Speech API is not supported in this browser.';
      console.warn(`[Speech] ${err}`);
      if (options.onError) options.onError(err);
      return { success: false, reason: 'unsupported' };
    }

    const cleanText = (text || '').trim();
    if (!cleanText) {
      const msg = 'No sentence to speak.';
      if (options.onError) options.onError(msg);
      return { success: false, reason: 'empty' };
    }

    // Cancel any ongoing speech before starting new utterance
    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.lang = 'en-US';
    utterance.rate = options.rate || 1.0;
    utterance.pitch = options.pitch || 1.0;

    if (preferredVoice) {
      utterance.voice = preferredVoice;
    }

    utterance.onstart = () => {
      isCurrentlySpeaking = true;
      if (options.onStart) options.onStart();
    };

    utterance.onend = () => {
      isCurrentlySpeaking = false;
      if (options.onEnd) options.onEnd();
    };

    utterance.onerror = (event) => {
      isCurrentlySpeaking = false;
      console.error('[Speech] SpeechSynthesisUtterance error:', event);
      if (options.onError) options.onError(event.error || 'Speech error');
    };

    window.speechSynthesis.speak(utterance);
    return { success: true };
  }

  /**
   * Helper that retrieves the current sentence from SentenceBuilder and speaks it.
   *
   * @param {Object} [options] - Speech options.
   * @returns {{ success: boolean, reason?: string }}
   */
  function speakCurrentSentence(options = {}) {
    const text = window.SentenceBuilder ? window.SentenceBuilder.getCurrentSentence() : '';
    return speak(text, options);
  }

  /**
   * Cancels any active speech playback.
   */
  function cancelSpeech() {
    if (isSpeechSupported) {
      window.speechSynthesis.cancel();
      isCurrentlySpeaking = false;
    }
  }

  /**
   * Returns whether speech audio is currently playing.
   * @returns {boolean}
   */
  function isSpeaking() {
    return isCurrentlySpeaking;
  }

  /**
   * Checks if the Web Speech API is supported on the current browser.
   * @returns {boolean}
   */
  function isSupported() {
    return isSpeechSupported;
  }

  // Export to global window object
  window.SpeechEngine = {
    speak,
    speakCurrentSentence,
    cancel: cancelSpeech,
    isSpeaking,
    isSupported,
  };

  // Direct global shortcut
  window.speakCurrentSentence = speakCurrentSentence;
})(window);
