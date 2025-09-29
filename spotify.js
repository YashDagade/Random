/***** SPOTIFY PLAYLIST BUILDER — BROWSER CONSOLE VERSION -- Generated using GPT5 :D*****
 * Steps:
 * 1) Get a user token with scopes: playlist-modify-public and (optionally) playlist-modify-private.
 * 2) Paste the token into TOKEN below.
 * 3) Paste this entire script into the Console and hit Enter.
 ****************************************************************/

// ==== REQUIRED: paste your short-lived user token here ====
const TOKEN = 'PASTE_YOUR_USER_TOKEN_HERE'; // you can get the token from going to https://developer.spotify.com/dashboard logging in and you shall see your token there. You can than inspect element get to console and from there run this script to add all the songs to the playlist :D

// ==== Basic settings ====
const PLAYLIST_NAME = "Jacob's Playlist";
const PLAYLIST_DESC  = "Auto-built from my text list";
const PUBLIC         = true;       // set false if you want it private
const MARKET         = 'US';       // helps disambiguate searches

// ==== Your songs (title + helpful artist hints for accuracy) ====
const SONGS = [
  { title: "Never Tear Us Apart", artist: "INXS" },
  { title: "Angels", artist: "Robbie Williams" },
  { title: "21 Guns", artist: "Green Day" },
  { title: "The Scientist", artist: "Coldplay" },
  { title: "Demons", artist: "Imagine Dragons" },
  { title: "It's Time", artist: "Imagine Dragons" },
  { title: "Knockin' on Heaven's Door", artist: "Guns N' Roses" }, // or "Bob Dylan"
  { title: "Creep", artist: "Radiohead" },
  { title: "Teenagers", artist: "My Chemical Romance" },
  { title: "Disenchanted", artist: "My Chemical Romance" },
  { title: "The Sharpest Lives", artist: "My Chemical Romance" },
  { title: "The Light Behind Your Eyes", artist: "My Chemical Romance" },
  { title: "Flightless Bird, American Mouth", artist: "Iron & Wine" },
  { title: "Dance Monkey", artist: "Tones And I" },
  { title: "With or Without You", artist: "U2" },
  { title: "Numb", artist: "Linkin Park" },
  { title: "Whatever It Takes", artist: "Imagine Dragons" },
  { title: "Wonderwall", artist: "Oasis" },
  { title: "Centuries", artist: "Fall Out Boy" },
  { title: "Dragonhearted", artist: "TryHardNinja" }, // adjust if you prefer a specific version
  { title: "I Want to Hold Your Hand", artist: "The Beatles" },
  { title: "I'm Not Okay (I Promise)", artist: "My Chemical Romance" },
  { title: "Immortals", artist: "Fall Out Boy" },
  { title: "House of the Rising Sun", artist: "The Animals" },
  { title: "Riptide", artist: "Vance Joy" },
  { title: "Shape of You", artist: "Ed Sheeran" },
  { title: "Someone You Loved", artist: "Lewis Capaldi" },
  { title: "Wish You Were Here", artist: "Pink Floyd" },
  { title: "Not Today", artist: "twenty one pilots" }, // change to Imagine Dragons if you meant that
  { title: "Take Me Home, Country Roads", artist: "John Denver" },
  { title: "Shallow", artist: "Lady Gaga" }, // (with Bradley Cooper)
  { title: "Viva La Vida", artist: "Coldplay" },
  { title: "Wake Me Up When September Ends", artist: "Green Day" },
  { title: "Don't Look Back in Anger", artist: "Oasis" },
  { title: "Far Too Young to Die", artist: "Panic! At The Disco" },
  { title: "Zombie", artist: "The Cranberries" },
  { title: "Little Talks", artist: "Of Monsters and Men" },
  { title: "Stop and Stare", artist: "OneRepublic" },
  { title: "Let Her Go", artist: "Passenger" },
  { title: "Let It Be", artist: "The Beatles" },
  { title: "You're Beautiful", artist: "James Blunt" },
];

// ==== Utility: robust fetch with rate-limit handling & errors ====
async function fetchWebApi(endpoint, method = 'GET', body) {
  if (!TOKEN || TOKEN.includes('PASTE_YOUR_USER_TOKEN_HERE')) {
    throw new Error('TOKEN is missing. Paste your user token into TOKEN at the top.');
  }
  const res = await fetch(`https://api.spotify.com/${endpoint}`, {
    method,
    headers: {
      'Authorization': `Bearer ${TOKEN}`,
      'Content-Type': 'application/json'
    },
    body: body ? JSON.stringify(body) : undefined
  });

  // Handle rate limits
  if (res.status === 429) {
    const retry = parseInt(res.headers.get('Retry-After') || '5', 10);
    console.warn(`Rate limited (429). Retrying in ${retry}s…`);
    await new Promise(r => setTimeout(r, retry * 1000));
    return fetchWebApi(endpoint, method, body);
  }

  if (res.status === 401) {
    const text = await res.text().catch(() => '');
    throw new Error(`401 Unauthorized — your token likely expired. Get a fresh token and try again.\n${text}`);
  }

  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`HTTP ${res.status}: ${text}`);
  }

  return res.status === 204 ? null : res.json();
}

async function getCurrentUserId() {
  const me = await fetchWebApi('v1/me', 'GET');
  return me.id;
}

// Use field filters for precise search
async function searchTrackURI(title, artist) {
  // Prefer track:"..." artist:"..." to reduce bad matches
  const q = artist ? `track:"${title}" artist:"${artist}"` : `track:"${title}"`;
  const params = new URLSearchParams({ q, type: 'track', limit: '5', market: MARKET });
  const res = await fetchWebApi(`v1/search?${params.toString()}`, 'GET');
  return res?.tracks?.items?.[0]?.uri || null;
}

async function createPlaylist(userId, name, description, isPublic) {
  return fetchWebApi(`v1/users/${encodeURIComponent(userId)}/playlists`, 'POST', {
    name,
    description,
    public: isPublic
  });
}

async function addTracks(playlistId, uris) {
  // Add up to 100 URIs per request
  for (let i = 0; i < uris.length; i += 100) {
    const batch = uris.slice(i, i + 100);
    const qp = new URLSearchParams({ uris: batch.join(',') });
    await fetchWebApi(`v1/playlists/${playlistId}/tracks?${qp.toString()}`, 'POST');
  }
}

// ==== Main flow ====
(async () => {
  try {
    console.log('➡️ Starting: creating playlist and adding tracks…');

    // 1) Get user
    const userId = await getCurrentUserId();

    // 2) Create playlist
    const playlist = await createPlaylist(userId, PLAYLIST_NAME, PLAYLIST_DESC, PUBLIC);
    const playlistUrl = `https://open.spotify.com/playlist/${playlist.id}`;
    console.log('✅ Created playlist:', playlistUrl);

    // 3) Resolve track URIs
    const uris = [];
    const misses = [];
    for (const s of SONGS) {
      const uri = await searchTrackURI(s.title, s.artist);
      if (uri) uris.push(uri);
      else misses.push(`${s.title}${s.artist ? ' — ' + s.artist : ''}`);
    }

    // 4) De-duplicate while preserving order
    const seen = new Set();
    const uniqueUris = uris.filter(u => !seen.has(u) && seen.add(u));

    // 5) Add to playlist in batches
    if (uniqueUris.length > 0) {
      await addTracks(playlist.id, uniqueUris);
    }

    console.log(`🎉 Done! Added ${uniqueUris.length} tracks to "${PLAYLIST_NAME}".`);
    console.log('🔗 Open it here:', playlistUrl);

    if (misses.length) {
      console.warn('⚠️ Could not auto-match these (check spelling/version or change artist hint):');
      for (const m of misses) console.warn(' -', m);
    }
  } catch (err) {
    console.error('❌ Error:', err.message || err);
  }
})();
