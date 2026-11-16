/*
 * recommend.js
 *
 * Turns a ranked list of moods into song recommendations.
 *
 * A note on the ranking, because it is easy to assume more is happening here
 * than actually is. Every song in data/songs.json carries valence and energy,
 * but those two numbers are derived from the song's own mood label: every
 * track inside one mood sits within 0.025 of the same point, and danceability
 * and acousticness are the same constant on all 2011 rows. So ranking songs by
 * distance in valence/energy space would be sorting by rounding noise.
 *
 * The honest signals in the data are the mood label, the release year, and
 * the links. So the matching works like this:
 *
 *   1. The image's mood scores decide how many songs come from each mood,
 *      which uses the whole distribution rather than only the top mood.
 *   2. Within a mood, release year breaks the tie where it carries meaning.
 *   3. Anything still tied is drawn with a generator seeded from the mood
 *      scores, so one photo always gives the same songs while different
 *      photos give different ones.
 */

// The seven mood keys in upload.js, mapped onto the seven labels that
// actually occur in data/songs.json.
const MOOD_TO_SONG_LABEL = new Map([
    ['joyful',     'happy and excited'],
    ['melancholy', 'sad and melancholic'],
    ['tense',      'angry and aggressive'],
    ['romantic',   'romantic and gentle'],
    ['calm',       'calm and peaceful'],
    ['nostalgic',  'nostalgic and reflective'],
    ['dark',       'dark and mysterious'],
]);

// A mood needs at least this share of the image before it contributes a song.
const MIN_SHARE = 0.08;

// How many songs the page shows at once, and how many more each "show more"
// click reveals.
export const PAGE_SIZE = 3;

// Upper bound on one photo's queue. Well past what anyone scrolls, but it
// keeps the queue from being the whole mood group on every analysis.
const MAX_QUEUE = 60;

// Used when narrowing the nostalgic pool; see orderWithinMood.
const DRAW_WINDOW = 12;

let songsPromise = null;

export function loadSongs() {
    if (!songsPromise) {
        songsPromise = fetch('data/songs.json')
            .then((response) => {
                if (!response.ok) throw new Error('Could not load the song list.');
                return response.json();
            })
            .catch((err) => {
                songsPromise = null;
                throw err;
            });
    }
    return songsPromise;
}

// Small seeded generator so a given photo always produces the same picks.
function makeRandom(seed) {
    let a = seed >>> 0;
    return function random() {
        a = (a + 0x6D2B79F5) >>> 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

/*
 * A fresh seed per analysis, so running the same photo again draws a
 * different set. The order stays fixed for the length of one queue, which
 * is what lets "show more" walk down it without repeating.
 *
 * This does not make a posted song unstable: a post stores the song the
 * user actually chose, so the feed reads that stored value and never
 * re-runs the recommender.
 */
function freshSeed() {
    return (Math.floor(Math.random() * 0xFFFFFFFF) >>> 0) || 1;
}

/*
 * Walk the moods in proportion to their share, using the highest-average
 * rule: each turn goes to the mood with the best share-per-song-already-
 * taken. The top mood always goes first. Stops when every pool is spent,
 * so the queue length is bounded by the songs that actually exist.
 */
function buildQueue(shares, pools, limit) {
    const taken = new Map(shares.map((s) => [s.mood, 0]));
    const queue = [];

    while (queue.length < limit) {
        let bestMood = null;
        let bestValue = -Infinity;
        for (const { mood, share } of shares) {
            if (taken.get(mood) >= pools.get(mood).length) continue;
            const value = share / (taken.get(mood) + 1);
            if (value > bestValue) {
                bestValue = value;
                bestMood = mood;
            }
        }
        if (bestMood === null) break;

        const index = taken.get(bestMood);
        const share = shares.find((s) => s.mood === bestMood).share;
        queue.push({ song: pools.get(bestMood)[index], mood: bestMood, share });
        taken.set(bestMood, index + 1);
    }
    return queue;
}

/*
 * Order the candidates inside one mood. Year is a real column, so it is used
 * where it means something: nostalgia points at the past, so older records
 * come first. The other moods have no honest year preference, so they fall
 * through to the seeded draw.
 */
function orderWithinMood(mood, candidates, random) {
    let pool = candidates;

    if (mood === 'nostalgic') {
        // Nostalgia points at the past, so lean older. This narrows to the
        // older half rather than sorting by year outright: a hard year sort
        // is the same for every photo, so the draw window below would land
        // on the same dozen oldest songs every time and the other ~290
        // nostalgic tracks would never be reachable.
        const years = candidates.map((s) => s.year || 9999).sort((a, b) => a - b);
        const median = years[Math.floor(years.length / 2)];
        const older = candidates.filter((s) => (s.year || 9999) <= median);
        if (older.length >= DRAW_WINDOW) pool = older;
    }

    // Seeded shuffle, so the order is stable for a given photo but differs
    // between photos. The song data carries no quality or popularity signal
    // to rank on, so this is an honest draw rather than a fake ranking.
    const ordered = pool.slice();
    for (let i = ordered.length - 1; i > 0; i -= 1) {
        const j = Math.floor(random() * (i + 1));
        [ordered[i], ordered[j]] = [ordered[j], ordered[i]];
    }
    return ordered;
}

/*
 * Build the full ranked queue for one photo.
 *
 * scored: [{ mood, score }] sorted high to low, straight from the model.
 * Returns [{ song, mood, share }], already in the order to show them.
 * The page displays the first few and reveals the rest on demand, so this
 * runs once per analysis rather than once per "show more" click.
 */
export async function recommend(scored, limit = MAX_QUEUE) {
    const songs = await loadSongs();

    // Group once, so this does not re-scan all 2011 rows per mood.
    const byLabel = new Map();
    for (const song of songs) {
        const label = song.mood;
        if (!label) continue;
        if (!byLabel.has(label)) byLabel.set(label, []);
        byLabel.get(label).push(song);
    }

    const total = scored.reduce((sum, s) => sum + s.score, 0) || 1;
    const allShares = scored
        .filter(({ mood }) => byLabel.has(MOOD_TO_SONG_LABEL.get(mood)))
        .map(({ mood, score }) => ({ mood, share: score / total }));
    if (allShares.length === 0) return [];

    const eligible = allShares.filter((s) => s.share >= MIN_SHARE);
    const shares = eligible.length > 0 ? eligible : allShares.slice(0, 1);

    const random = makeRandom(freshSeed());
    const pools = new Map(
        shares.map(({ mood }) => [
            mood,
            orderWithinMood(mood, byLabel.get(MOOD_TO_SONG_LABEL.get(mood)) || [], random),
        ])
    );

    return buildQueue(shares, pools, limit);
}

export { MOOD_TO_SONG_LABEL };
