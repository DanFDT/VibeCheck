/*
 * model.js
 *
 * Loads the SigLIP zero-shot image classification pipeline via Transformers.js
 * directly from a CDN, no build step. The first call downloads the model
 * (around 100 MB quantized) with a progress callback; later calls on the same
 * device read from the browser's IndexedDB cache and finish quickly.
 *
 * Usage:
 *   import { loadModel } from './model.js';
 *   const classifier = await loadModel((event) => { ... });
 *   const result = await classifier(imageUrl, moodPromptArray);
 */

// SigLIP-base is the strong zero-shot image classifier (~100 MB quantized).
// First visit downloads once to IndexedDB; later visits load from cache.
// Preload on page open so the download happens while the user is reading
// the hero rather than after they click Analyze.
const MODEL_ID = 'Xenova/siglip-base-patch16-224';
const TRANSFORMERS_CDN = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0';

let cachedClassifier = null;
let inFlight = null;
const progressHandlers = new Set();

export function onModelProgress(handler) {
    if (typeof handler === 'function') progressHandlers.add(handler);
    return () => progressHandlers.delete(handler);
}

export function isModelReady() {
    return cachedClassifier !== null;
}

export async function loadModel() {
    if (cachedClassifier) return cachedClassifier;
    if (inFlight) return inFlight;

    inFlight = (async () => {
        let pipeline;
        try {
            ({ pipeline } = await import(TRANSFORMERS_CDN));
        } catch (err) {
            inFlight = null;
            throw new Error('Could not load Transformers.js. Check your connection.');
        }

        try {
            const classifier = await pipeline(
                'zero-shot-image-classification',
                MODEL_ID,
                {
                    progress_callback: (event) => {
                        progressHandlers.forEach((h) => {
                            try { h(event); } catch (_) { /* ignore handler errors */ }
                        });
                    },
                }
            );
            cachedClassifier = classifier;
            return classifier;
        } catch (err) {
            inFlight = null;
            throw new Error('Could not load the mood model. ' + (err && err.message ? err.message : ''));
        }
    })();

    return inFlight;
}
