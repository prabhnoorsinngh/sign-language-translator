/**
 * speech.js
 *
 * PURPOSE:
 *   Converts the built sentence into audible speech using the
 *   Web Speech API (SpeechSynthesis).
 *
 * FUTURE LOGIC:
 *   - Accept a text string from sentenceBuilder.getSentence()
 *   - Create a SpeechSynthesisUtterance with the given text
 *   - Configure voice, rate, pitch, and volume settings
 *   - Call window.speechSynthesis.speak() to play the audio
 *   - Expose speak(text) as the public interface
 *   - Handle browsers that do not support SpeechSynthesis gracefully
 */
