/*
 * upload.js
 *
 * Handles photo selection on index.html through four input surfaces:
 * file input change, drag and drop, clipboard paste, and keyboard activation
 * of the file picker (via the native label + input relationship).
 *
 * After a valid image is received, replaces the upload form with a preview
 * and an "Analyze this" button. Clicking Analyze loads the SigLIP model with
 * a progress bar; inference will land in the next module.
 */

import { loadModel, onModelProgress, isModelReady } from './model.js';
import { recommend, loadSongs, PAGE_SIZE } from './recommend.js';

// Seven moods matching the --mood-* CSS tokens, each with several prompts.
// SigLIP is trained on web alt-text, so it scores literal descriptions of
// what is visible in the frame far higher than abstract mood words. Each
// mood gets prompts for both people and scenes, because one mood has to
// cover a crying face and an empty rainy street alike.
// Each mood needs prompts for the three shapes a real upload takes: one face
// close up (a selfie), a group or scene with people in it, and a photo with
// no people at all. Missing any one of those makes the mood unreachable for
// a whole category of photo.
const MOOD_PROMPTS = {
    joyful:     ['a selfie of a smiling person', 'a photo of people smiling and laughing', 'a photo of a happy celebration', 'a bright cheerful sunny photo'],
    melancholy: ['a photo of a person looking sad', 'a photo of a sad crying person', 'a photo of a lonely person sitting alone', 'a grey rainy gloomy photo'],
    tense:      ['a photo of a person with an angry face', 'a photo of an angry shouting person', 'a photo of a fight or conflict', 'a harsh aggressive photo'],
    romantic:   ['a photo of a couple on a date', 'a romantic photo of a couple', 'a photo of a man and a woman embracing'],
    calm:       ['a photo of a quiet empty landscape', 'a photo of a peaceful forest or lake', 'a soft minimal photo of nature'],
    nostalgic:  ['a photo of an old historic building', 'a nostalgic photo of an old place', 'an old faded vintage photograph'],
    dark:       ['a dark photo taken at night', 'a photo of a city at night with lights', 'a moody dark shadowy photo'],
};

// Below this raw sigmoid score the model is not recognising any mood, it is
// just ranking noise. Without this the page always names a confident winner,
// because dividing by the sum turns any seven numbers into tidy percentages.
//
// Measured on the sample set: real matches run from 7.0e-4 (the weakest) up
// to 7.7e-1, while random noise tops out at 2.1e-4 and a page of text at
// 4.9e-5. The gap is narrow, and a flat grey image still clears it by
// matching the "grey rainy gloomy" prompt, so treat this as a guard against
// obvious nonsense rather than a real confidence measure. Worth re-tuning
// once there are more everyday photos to test against.
const CONFIDENCE_FLOOR = 3e-4;

// One flat list goes to the model; the Map takes each prompt back to its mood.
const MOOD_LABELS = Object.values(MOOD_PROMPTS).flat();
const PROMPT_TO_MOOD = new Map(
    Object.entries(MOOD_PROMPTS).flatMap(([mood, prompts]) => prompts.map((prompt) => [prompt, mood]))
);

// Preload the model on page open so the download runs while the user is
// still reading the hero. The call is idempotent; the Analyze click later
// awaits this same promise.
const topLoader = createTopLoader();
onModelProgress((event) => {
    if (!event) return;
    if (event.status === 'progress' && typeof event.progress === 'number') {
        topLoader.update(`Preparing the mood model... ${Math.round(event.progress)}%`, event.progress);
    } else if (event.status === 'ready' || event.status === 'done') {
        topLoader.done();
    }
});
loadModel().catch((err) => {
    console.warn('[vibecheck] model preload failed:', err);
    topLoader.fail();
});

// The song list is small next to the model, so fetch it up front too and let
// the analysis step find it already cached.
loadSongs().catch((err) => {
    console.warn('[vibecheck] song list preload failed:', err);
});

function createTopLoader() {
    const wrap = document.createElement('div');
    wrap.className = 'top-loader';
    wrap.innerHTML = `
        <div class="top-loader__bar"></div>
        <p class="top-loader__label caption">Preparing the mood model</p>
    `;
    document.body.prepend(wrap);
    const bar = wrap.querySelector('.top-loader__bar');
    const label = wrap.querySelector('.top-loader__label');
    let hideTimer = null;
    return {
        update(text, pct) {
            label.textContent = text;
            bar.style.width = pct + '%';
        },
        done() {
            label.textContent = 'Mood model ready';
            bar.style.width = '100%';
            clearTimeout(hideTimer);
            hideTimer = setTimeout(() => wrap.classList.add('is-hidden'), 1200);
        },
        fail() {
            label.textContent = 'Model could not load, Analyze will retry';
            wrap.classList.add('is-failed');
        }
    };
}

const fileInput = document.querySelector('#photo');
const uploadForm = document.querySelector('.upload');
const hero = document.querySelector('.hero');

let currentFile = null;
let currentPreviewUrl = null;
let dragDepth = 0;

function isImage(file) {
    return file && typeof file.type === 'string' && file.type.startsWith('image/');
}

function clearError() {
    const existing = hero.querySelector('.upload-error');
    if (existing) existing.remove();
}

function showError(message) {
    clearError();
    const p = document.createElement('p');
    p.className = 'upload-error';
    p.textContent = message;
    hero.appendChild(p);
}

function renderPreview(file, url) {
    const img = document.createElement('img');
    img.src = url;
    img.alt = 'Your photo';
    img.className = 'preview-image';

    const meta = document.createElement('p');
    meta.className = 'preview-meta caption';
    meta.textContent = `${file.name}, ${Math.round(file.size / 1024)} KB`;

    const analyze = document.createElement('button');
    analyze.type = 'button';
    analyze.className = 'upload-button';
    analyze.textContent = 'Analyze this';
    analyze.addEventListener('click', onAnalyzeClick);

    const change = document.createElement('button');
    change.type = 'button';
    change.className = 'preview-change';
    change.textContent = 'Choose a different photo';
    change.addEventListener('click', resetToUpload);

    const status = document.createElement('div');
    status.className = 'analyze-status';
    status.hidden = true;
    status.innerHTML = `
        <p class="analyze-status__label caption">Preparing model</p>
        <div class="analyze-status__track"><div class="analyze-status__bar"></div></div>
    `;

    const wrap = document.createElement('div');
    wrap.className = 'preview';
    wrap.append(img, meta, analyze, change, status);

    if (uploadForm) uploadForm.hidden = true;
    clearError();

    const existing = hero.querySelector('.preview');
    if (existing) existing.replaceWith(wrap);
    else hero.appendChild(wrap);
}

function resetToUpload() {
    const preview = hero.querySelector('.preview');
    if (preview) preview.remove();
    if (uploadForm) uploadForm.hidden = false;
    if (currentPreviewUrl) {
        URL.revokeObjectURL(currentPreviewUrl);
        currentPreviewUrl = null;
    }
    currentFile = null;
    fileInput.value = '';
}

function handleFile(file) {
    if (!file) return;
    if (!isImage(file)) {
        showError('That file is not an image. Try a JPG, PNG, or HEIC.');
        return;
    }
    if (currentPreviewUrl) URL.revokeObjectURL(currentPreviewUrl);
    currentFile = file;
    currentPreviewUrl = URL.createObjectURL(file);
    renderPreview(file, currentPreviewUrl);
}

async function onAnalyzeClick(event) {
    const button = event.currentTarget;
    const preview = hero.querySelector('.preview');
    if (!preview || !currentFile) return;

    const status = preview.querySelector('.analyze-status');
    const label = status.querySelector('.analyze-status__label');
    const bar = status.querySelector('.analyze-status__bar');

    button.disabled = true;
    button.textContent = 'Loading model';
    status.hidden = false;
    clearError();

    // If the preload already finished, this returns immediately. If still
    // downloading, the shared progress handler updates both the top bar
    // and this inline status.
    if (isModelReady()) {
        label.textContent = 'Model ready';
        bar.style.width = '100%';
    }

    const unsubscribe = onModelProgress((e) => {
        if (e && e.status === 'progress' && typeof e.progress === 'number') {
            label.textContent = `Downloading ${e.file} (${Math.round(e.progress)}%)`;
            bar.style.width = Math.round(e.progress) + '%';
        } else if (e && e.status === 'ready') {
            label.textContent = 'Model ready';
            bar.style.width = '100%';
        }
    });

    try {
        await loadModel();
        unsubscribe();
        label.textContent = 'Model ready';
        bar.style.width = '100%';
        button.disabled = false;
        button.textContent = 'Run analysis';
        button.removeEventListener('click', onAnalyzeClick);
        button.addEventListener('click', onRunAnalysisClick);
    } catch (err) {
        unsubscribe();
        console.error('[vibecheck] model load failed:', err);
        showError(err.message || 'The model could not load. Try again.');
        status.hidden = true;
        button.disabled = false;
        button.textContent = 'Analyze this';
    }
}

async function onRunAnalysisClick(event) {
    const button = event.currentTarget;
    if (!currentPreviewUrl) return;

    button.disabled = true;
    button.textContent = 'Reading the photo';
    clearError();

    try {
        const classifier = await loadModel();
        const raw = await classifier(currentPreviewUrl, MOOD_LABELS);

        // SigLIP scores every prompt on its own with a sigmoid, so the scores
        // do not add up to 1. A mood takes the score of its strongest prompt;
        // summing or averaging would punish a mood for having prompts that
        // do not apply to this particular photo.
        const best = new Map();
        for (const { label, score } of raw) {
            const mood = PROMPT_TO_MOOD.get(label);
            if (!mood) continue;
            if (!best.has(mood) || score > best.get(mood)) best.set(mood, score);
        }
        const scored = [...best]
            .map(([mood, score]) => ({ mood, score }))
            .sort((a, b) => b.score - a.score);

        console.log('[vibecheck] mood scores:', scored);
        renderMoodResults(scored);

        // Recommending from scores this low would be picking songs off noise,
        // which is the whole bug this floor exists to stop.
        if (scored[0].score < CONFIDENCE_FLOOR) {
            console.warn('[vibecheck] below confidence floor, no songs:', scored[0]);
            button.disabled = false;
            button.textContent = 'Analyze again';
            return;
        }

        button.textContent = 'Finding songs';
        try {
            const queue = await recommend(scored);
            console.log('[vibecheck] queue of', queue.length, 'songs:', queue);
            renderSongs(queue);
        } catch (songErr) {
            // A missing song list should not throw away a good mood read.
            console.error('[vibecheck] song lookup failed:', songErr);
            showError('Found the mood, but the song list could not load.');
        }

        button.disabled = false;
        button.textContent = 'Analyze again';
    } catch (err) {
        console.error('[vibecheck] inference failed:', err);
        showError(err.message || 'The analysis failed. Try again.');
        button.disabled = false;
        button.textContent = 'Run analysis';
    }
}

function renderMoodResults(scored) {
    const preview = hero.querySelector('.preview');
    if (!preview) return;

    const existing = preview.querySelector('.mood-results');
    if (existing) existing.remove();

    const top = scored[0];
    const totalPct = scored.reduce((acc, s) => acc + s.score, 0) || 1;
    const confident = top.score >= CONFIDENCE_FLOOR;

    const container = document.createElement('div');
    container.className = 'mood-results';
    if (!confident) container.classList.add('mood-results--unsure');

    const headline = document.createElement('h2');
    headline.className = 'mood-top';
    headline.textContent = confident ? formatMood(top.mood) + '.' : 'Not sure.';

    if (!confident) {
        const note = document.createElement('p');
        note.className = 'mood-note';
        note.textContent = 'This photo does not look strongly like any of the seven moods. The ranking below is weak, so no songs were picked.';
        container.appendChild(headline);
        container.appendChild(note);
    }

    const list = document.createElement('ol');
    list.className = 'mood-ranked';
    for (const item of scored) {
        const li = document.createElement('li');
        const swatch = document.createElement('span');
        swatch.className = `mood-swatch mood-swatch--${item.mood}`;
        const name = document.createElement('span');
        name.className = 'mood-name';
        name.textContent = formatMood(item.mood);
        const score = document.createElement('span');
        score.className = 'mood-score caption';
        score.textContent = Math.round((item.score / totalPct) * 100) + '%';
        li.append(swatch, name, score);
        list.appendChild(li);
    }

    // The headline is already in place when the read was weak.
    if (confident) container.appendChild(headline);
    container.appendChild(list);
    preview.appendChild(container);

    // The results land below the fold on most screens, so without this the
    // page looks unchanged and the click feels like it did nothing.
    const motionOk = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    container.scrollIntoView({ behavior: motionOk ? 'smooth' : 'auto', block: 'center' });
}

function renderSongs(queue) {
    const preview = hero.querySelector('.preview');
    if (!preview) return;

    const existing = preview.querySelector('.result-songs');
    if (existing) existing.remove();
    if (queue.length === 0) return;

    const section = document.createElement('section');
    section.className = 'result-songs';

    const heading = document.createElement('h3');
    heading.className = 'songs-heading';
    heading.textContent = 'Songs that match';

    const list = document.createElement('ol');
    list.className = 'song-list';

    const more = document.createElement('button');
    more.type = 'button';
    more.className = 'songs-more';

    // The reason line is written once per mood. Repeating the same percentage
    // on every card from the same mood reads like a glitch.
    const topMood = queue[0].mood;
    const explained = new Set();
    let shown = 0;

    function showMore() {
        const next = queue.slice(shown, shown + PAGE_SIZE);
        appendCards(list, next, topMood, explained);
        shown += next.length;

        const left = queue.length - shown;
        if (left <= 0) {
            more.hidden = true;
        } else {
            more.textContent = `Show ${Math.min(PAGE_SIZE, left)} more`;
        }
    }

    more.addEventListener('click', () => {
        const before = list.lastElementChild;
        showMore();
        // Put the first newly added card in view rather than jumping to the
        // bottom, so the page follows the click instead of outrunning it.
        const firstNew = before ? before.nextElementSibling : list.firstElementChild;
        if (firstNew) firstNew.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });

    showMore();

    section.append(heading, list, more);
    preview.appendChild(section);
}

function appendCards(list, picks, topMood, explained) {
    picks.forEach(({ song, mood, share }) => {
        const card = document.createElement('li');
        card.className = 'song-card';

        const cover = document.createElement('div');
        cover.className = `song-cover song-cover--${mood}`;
        if (song.cover) {
            const art = document.createElement('img');
            art.src = song.cover;
            art.alt = '';
            art.loading = 'lazy';
            art.className = 'song-cover__art';
            // A dead cover URL should leave the mood colour behind, not a
            // broken image icon.
            art.addEventListener('error', () => art.remove());
            cover.appendChild(art);
        }

        const info = document.createElement('div');
        info.className = 'song-info';

        const title = document.createElement('h4');
        title.className = 'song-title';
        title.textContent = song.title;

        const artist = document.createElement('p');
        artist.className = 'song-artist';
        artist.textContent = song.year ? `${song.artist}, ${song.year}` : song.artist;

        const why = document.createElement('p');
        why.className = 'song-why caption';
        const pct = Math.round(share * 100);
        if (!explained.has(mood)) {
            explained.add(mood);
            why.textContent = mood === topMood
                ? `Your photo read ${formatMood(mood)} (${pct}%).`
                : `It also picked up ${formatMood(mood)} (${pct}%).`;
        } else {
            why.textContent = `Another ${formatMood(mood).toLowerCase()} track.`;
        }

        info.append(title, artist, why);

        const links = document.createElement('div');
        links.className = 'song-links';
        const targets = [['Spotify', song.spotify], ['YouTube', song.youtube]];
        for (const [name, href] of targets) {
            if (!href) continue;
            const link = document.createElement('a');
            link.className = 'song-link song-link--text';
            link.href = href;
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
            link.textContent = name;
            link.setAttribute('aria-label', `Open ${song.title} on ${name}`);
            links.appendChild(link);
        }

        card.append(cover, info, links);
        list.appendChild(card);
    });
}

function formatMood(key) {
    const names = {
        calm: 'Calm',
        joyful: 'Joyful',
        tense: 'Tense',
        melancholy: 'Melancholy',
        dark: 'Dark',
        romantic: 'Romantic',
        nostalgic: 'Nostalgic',
    };
    return names[key] || key;
}

fileInput.addEventListener('change', (event) => {
    handleFile(event.target.files[0]);
});

window.addEventListener('dragenter', (event) => {
    if (!event.dataTransfer) return;
    event.preventDefault();
    dragDepth += 1;
    document.body.classList.add('is-dropping');
});

window.addEventListener('dragover', (event) => {
    event.preventDefault();
});

window.addEventListener('dragleave', (event) => {
    event.preventDefault();
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) document.body.classList.remove('is-dropping');
});

window.addEventListener('drop', (event) => {
    event.preventDefault();
    dragDepth = 0;
    document.body.classList.remove('is-dropping');
    const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
    handleFile(file);
});

window.addEventListener('paste', (event) => {
    if (!event.clipboardData) return;
    const items = Array.from(event.clipboardData.items || []);
    const item = items.find((entry) => entry.type && entry.type.startsWith('image/'));
    if (!item) return;
    const file = item.getAsFile();
    if (file) handleFile(file);
});
