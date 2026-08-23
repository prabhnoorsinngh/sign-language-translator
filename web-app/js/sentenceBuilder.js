/**
 * sentenceBuilder.js
 *
 * PURPOSE:
 *   Accumulates individual sign predictions into a readable sentence.
 *
 * FUTURE LOGIC:
 *   - Maintain an internal buffer of confirmed sign predictions
 *   - Debounce / deduplicate repeated signs (hold-duration threshold)
 *   - Append new signs to the sentence displayed in the UI
 *   - Expose addSign(label) to append a word/letter
 *   - Expose getSentence() → string for speech.js to read aloud
 *   - Expose clearSentence() for the "Clear" button in index.html
 */
