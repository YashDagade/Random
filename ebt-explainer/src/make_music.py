"""Procedural ambient music bed for the EBT explainer video (no downloads, numpy only).

Output:
  video/music.wav               44.1 kHz, 16-bit stereo, video length + 30 s tail (630 s by default)
  video/music_preview_60s.wav   60 s excerpt (470-530 s: end of eval into the scaling build)
  video/music.m4a               AAC 128k copy of music.wav (the canvas player prefers it)
  video/music_meta.json         the beat-snapped section bounds, read by music_check/music_verify

Usage:
  python3 -I src/make_music.py                     # storyboard timing (9 scenes, video ends at 600 s)
  python3 -I src/make_music.py --bounds 0,53,130,205,265,345,415,490,563.5,608.5
        # re-time to the real scene starts + video end (10 numbers, seconds; snapped to beats)
  python3 -I src/make_music.py --timeline          # print the chord timeline only (fast)

Design (all deterministic, seeded):
  * 72 BPM, 4/4. One chord every 2 bars (8 beats). Key of D major, leaning lydian
    through the IV chord (Gmaj7#11).
  * Scene starts [0, 50, 125, 200, 260, 340, 410, 485, 555] s and the video end (600 s) fall on
    whole beats at 72 BPM (t * 1.2 = beat), so every scene starts with a fresh chord. Other
    bounds (--bounds) are snapped to the nearest beat (at most 0.42 s shift).
  * Ending cadence (Bm9 Gmaj9 Asus4 A, then Dmaj9 with falling bells) is placed so the Dmaj9
    lands RESOLVE_LEAD = 10 s before the video ends: the cadence plays under the last scene and
    the end card instead of after the video (render_video.js cuts the music with -shortest).
  * Instruments: detuned additive-sine pads (slow attack/release, harmonics open with the
    envelope for a low-pass feel), a soft marimba-like pluck arpeggio, a sparse bell
    melody in some sections, a very soft sine bass on chord roots.
  * Stereo: every pad sub-voice and every pluck has its own constant-power pan.
  * Reverb: convolution with a synthetic, decorrelated stereo impulse (band-split noise
    with frequency-dependent exponential decay).
  * Master: 35 Hz high-pass, 6.5 kHz low-pass, RMS normalised to about -24 dBFS, a
    transparent look-ahead peak limiter at -6 dBFS, 4 s fade-in, 6 s fade-out, TPDF dither.
"""
import os
import sys
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import minimum_filter1d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "video", "music.wav")
OUT_PREVIEW = os.path.join(ROOT, "video", "music_preview_60s.wav")
OUT_M4A = os.path.join(ROOT, "video", "music.m4a")   # player.html prefers this (smaller than the WAV)
OUT_META = os.path.join(ROOT, "video", "music_meta.json")

SR = 44100
BPM = 72.0
BEAT = 60.0 / BPM            # 0.8333 s
# 9 scene starts + video end, from notes/STORYBOARD.md. Override with --bounds (see Usage).
SCENE_BOUNDS = [0, 50, 125, 200, 260, 340, 410, 485, 555, 600]
RESOLVE_LEAD = 10.0          # the final Dmaj9 arrives this many seconds before the video ends
TAIL = 30.0                  # music keeps ringing past the video end (slack for ffmpeg -shortest)


def _scene_beats():
    vals = SCENE_BOUNDS
    if "--bounds" in sys.argv:
        vals = [float(v) for v in sys.argv[sys.argv.index("--bounds") + 1].split(",")]
    beats = [int(round(v / BEAT)) for v in vals]
    assert len(beats) == 10 and beats[0] == 0, "need 10 values: 9 scene starts (first 0) + video end"
    assert all(b - a >= 8 for a, b in zip(beats, beats[1:])), "every scene needs >= 8 beats (6.7 s)"
    return beats


SCENE_BEATS = _scene_beats()
VIDEO_END = SCENE_BEATS[-1] * BEAT
# the ending section (24 beats of cadence before the Dmaj9) borrows the end of the wrap scene
ENDING_BEAT = max(int(round((VIDEO_END - RESOLVE_LEAD) / BEAT)) - 24, SCENE_BEATS[8] + 8)
RESOLVE_BEAT = ENDING_BEAT + 24
DUR_BEAT = int(np.ceil((VIDEO_END + TAIL) / BEAT - 1e-9))
SEC_BEATS = SCENE_BEATS[:9] + [ENDING_BEAT, DUR_BEAT]     # section b0 for each of SECTIONS, + end
BOUNDS = [b * BEAT for b in SEC_BEATS]
DUR = DUR_BEAT * BEAT
N = int(round(SR * DUR))
TAU = 2 * np.pi
# instrument bus gains (balanced by measuring per-stem RMS, see main())
PAD_GAIN = 1.0
PLUCK_GAIN = 0.45
BELL_GAIN = 0.20
BASS_GAIN = 0.22


def mtof(m):
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


# ----------------------------------------------------------------------------- chords
# name: (bass midi, pad voicing, arp tones)
CHORDS = {
    "Dadd9":    (38, [50, 57, 62, 64, 66, 69], [62, 64, 66, 69, 74, 76]),
    "Dmaj9":    (38, [50, 57, 61, 64, 66],     [62, 64, 66, 69, 73, 74]),
    "Gmaj7#11": (43, [55, 59, 62, 66, 73],     [62, 66, 67, 71, 73, 78]),
    "Gmaj9":    (43, [55, 59, 62, 66, 69],     [62, 66, 67, 69, 71, 74]),
    "Bm9":      (35, [54, 59, 62, 66, 69, 73], [59, 62, 66, 69, 71, 73]),
    "Em9":      (40, [52, 59, 62, 66, 67],     [59, 62, 64, 66, 67, 71]),
    "Asus4":    (45, [52, 57, 62, 64, 69, 71], [57, 62, 64, 69, 71, 76]),
    "Aadd9":    (45, [52, 57, 61, 64, 69, 71], [57, 61, 64, 69, 71, 73]),
    "D/F#":     (42, [54, 57, 62, 64, 69],     [57, 62, 66, 69, 74, 76]),
    "F#m7":     (42, [54, 57, 61, 64, 69],     [61, 64, 66, 69, 73, 76]),
}

# ----------------------------------------------------------------------------- sections
# density = (start, end) linear ramp of arpeggio probability across the section.
# grid: 2 = eighth notes, 1 = quarter notes only.  span = slice of the arp tone list.
SECTIONS = [
    dict(name="intro", prog=["Dadd9", "Gmaj7#11"], pad=0.62, bright=0.40, top=True, sparkle=0.0,
         arp=(0.06, 0.30), grid=1, pattern="broken", span=(0, 6), shift=0, arp_gain=0.80,
         bass="hold", bass_gain=0.55, bass_from=12.0, bell=0.0, width=0.75),
    dict(name="families", prog=["Dadd9", "Bm9", "Gmaj9", "Asus4"], pad=0.75, bright=0.50, top=True,
         sparkle=0.0, arp=(0.30, 0.45), grid=2, pattern="up", span=(0, 6), shift=0, arp_gain=0.85,
         bass="hold", bass_gain=0.70, bell=0.0, width=0.85),
    dict(name="energy", prog=["Bm9", "Gmaj9", "D/F#", "Em9"], pad=0.72, bright=0.32, top=False,
         sparkle=0.0, arp=(0.40, 0.45), grid=1, pattern="broken", span=(0, 4), shift=0, arp_gain=0.85,
         bass="hold", bass_gain=0.80, bell=0.10, width=0.80),
    dict(name="data", prog=["Gmaj7#11", "D/F#", "Em9", "Aadd9"], pad=0.78, bright=0.55, top=True,
         sparkle=0.35, arp=(0.45, 0.55), grid=2, pattern="updown", span=(0, 4), shift=12,
         arp_gain=0.70, bass="pulse", bass_gain=0.65, bell=0.0, width=1.0),
    dict(name="thinking", prog=["Dmaj9", "Bm9", "Gmaj7#11", "Aadd9"], pad=0.78, bright=0.48, top=True,
         sparkle=0.0, arp=(0.50, 0.40), grid=2, pattern="down", span=(0, 6), shift=0, arp_gain=0.85,
         bass="hold", bass_gain=0.75, bell=0.18, width=0.90),
    dict(name="training", prog=["Bm9", "Gmaj9", "Dadd9", "Asus4"], pad=0.78, bright=0.55, top=True,
         sparkle=0.0, arp=(0.55, 0.70), grid=2, pattern="up", span=(0, 6), shift=0, arp_gain=0.85,
         bass="pulse", bass_gain=0.75, bell=0.25, width=0.95),
    dict(name="eval", prog=["Gmaj9", "F#m7", "Em9", "Aadd9"], pad=0.78, bright=0.48, top=True,
         sparkle=0.0, arp=(0.50, 0.50), grid=2, pattern="broken", span=(0, 6), shift=0, arp_gain=0.80,
         bass="walk", bass_gain=0.70, bell=0.0, width=0.90),
    dict(name="scaling", prog=["Gmaj9", "Aadd9", "Bm9", "D/F#"], pad=0.88, bright=0.62, top=True,
         sparkle=0.35, arp=(0.60, 0.80), grid=2, pattern="updown", span=(0, 6), shift=0, arp_gain=0.85,
         bass="pulse", bass_gain=0.80, bell=0.35, width=1.0),
    dict(name="wrap", prog=["Gmaj7#11", "Dadd9", "Em9", "Asus4"], pad=0.72, bright=0.42, top=True,
         sparkle=0.0, arp=(0.40, 0.30), grid=1, pattern="broken", span=(0, 6), shift=0, arp_gain=0.85,
         bass="hold", bass_gain=0.70, bell=0.15, width=0.85),
    dict(name="ending", prog=None, pad=0.75, bright=0.40, top=True, sparkle=0.0,
         arp=(0.35, 0.0), grid=1, pattern="broken", span=(0, 6), shift=0, arp_gain=0.80,
         bass="hold", bass_gain=0.65, bell=0.0, width=0.85),
]
# Ending: (chord, beats). The Dmaj9 starts at RESOLVE_BEAT and rings to the end of the file.
ENDING = [("Bm9", 8), ("Gmaj9", 8), ("Asus4", 4), ("Aadd9", 4), ("Dmaj9", DUR_BEAT - RESOLVE_BEAT)]


def build_timeline():
    """List of chord events {t0, t1, name, sec}."""
    events = []
    for si, sec in enumerate(SECTIONS):
        b0, b1 = SEC_BEATS[si], SEC_BEATS[si + 1]
        if sec["prog"] is None:
            lens = [b for _, b in ENDING]
            names = [c for c, _ in ENDING]
            assert sum(lens) == b1 - b0
        else:
            total = b1 - b0
            n8, r = divmod(total, 8)
            lens = [8] * n8
            if r >= 4:
                lens.append(r)        # a short one-bar turnaround chord
            elif r > 0:
                lens[-1] += r         # stretch the last chord
            names = [sec["prog"][i % len(sec["prog"])] for i in range(len(lens))]
        b = b0
        for name, L in zip(names, lens):
            events.append(dict(t0=b * BEAT, t1=(b + L) * BEAT, name=name, sec=si, beats=L, b0=b))
            b += L
    return events


# ----------------------------------------------------------------------------- buses
STEMS = {k: np.zeros((2, N), np.float32) for k in ("pad", "pluck", "bell", "bass")}
send = np.zeros((2, N), np.float32)


def pan_gains(p):
    th = (np.clip(p, -1, 1) + 1) * np.pi / 4
    return np.cos(th), np.sin(th)


def add(sig, t0, pan, gain, rev, inst):
    i0 = int(round(t0 * SR))
    if i0 >= N:
        return
    if i0 < 0:
        sig = sig[-i0:]
        i0 = 0
    n = min(len(sig), N - i0)
    s = sig[:n].astype(np.float32)
    gl, gr = pan_gains(pan)
    dry = STEMS[inst]
    dry[0, i0:i0 + n] += (gain * gl) * s
    dry[1, i0:i0 + n] += (gain * gr) * s
    if rev > 0:
        send[0, i0:i0 + n] += (gain * gl * rev) * s
        send[1, i0:i0 + n] += (gain * gr * rev) * s


def rc_ramp(n):
    """Raised-cosine 0 -> 1 ramp of n samples."""
    if n <= 0:
        return np.ones(0)
    return 0.5 - 0.5 * np.cos(np.pi * (np.arange(n) + 0.5) / n)


# ----------------------------------------------------------------------------- instruments
def pad_note(f, length, att, rel, bright, rng):
    """One pad note: 4 detuned sub-voices, each (signal, pan offset). Harmonics bloom with env."""
    n = int(length * SR)
    t = np.arange(n) / SR
    env = np.ones(n)
    na, nr = int(att * SR), int(rel * SR)
    na = min(na, n // 2)
    nr = min(nr, n - na)
    env[:na] = rc_ramp(na)
    env[n - nr:] = rc_ramp(nr)[::-1]
    # a gentle swell over the note's life
    env *= 0.88 + 0.12 * np.sin(np.pi * t / length)
    henv = env ** 1.6          # harmonics lag the fundamental: "filter opening" feel
    out = []
    for cents, pan in ((-7.0, -0.9), (-2.5, 0.35), (2.5, -0.35), (7.0, 0.9)):
        fd = f * 2 ** ((cents + rng.uniform(-0.8, 0.8)) / 1200)
        ph = TAU * fd * t + rng.uniform(0, TAU)
        lfo = 1.0 + 0.18 * np.sin(TAU * rng.uniform(0.05, 0.14) * t + rng.uniform(0, TAU))
        hiroll = (330.0 / max(f, 330.0)) ** 0.8   # higher notes get fewer harmonics
        s = np.sin(ph) * env + henv * bright * hiroll * (
            0.34 * np.sin(2 * ph + rng.uniform(0, TAU)) + 0.10 * np.sin(3 * ph + rng.uniform(0, TAU)))
        out.append((s * lfo * 0.25, pan))
    return out


def pluck(f, vel, rng, tau_scale=1.0):
    """Soft marimba-like pluck: exponentially decaying sines with a little harmonic content."""
    tau = 0.95 * tau_scale * (440.0 / f) ** 0.35
    length = min(5.0 * tau, 4.5)
    n = int(length * SR)
    t = np.arange(n) / SR
    hi = (440.0 / f) ** 1.0      # fewer upper partials on high notes
    s = (np.sin(TAU * f * t) * np.exp(-t / tau)
         + 0.10 * hi * np.sin(TAU * 2 * f * t + 0.3) * np.exp(-t / (0.45 * tau))
         + 0.035 * hi * np.sin(TAU * 3 * f * t + 0.7) * np.exp(-t / (0.25 * tau))
         + 0.05 * hi * np.sin(TAU * 4 * f * t + 1.1) * np.exp(-t / 0.09))
    a = int(0.006 * SR)
    s[:a] *= rc_ramp(a)
    r = int(0.35 * SR)
    s[-r:] *= rc_ramp(r)[::-1]
    return s * vel


def bell(f, vel, rng):
    tau = 2.0
    length = 7.0
    n = int(length * SR)
    t = np.arange(n) / SR
    s = (np.sin(TAU * f * t) * np.exp(-t / tau)
         + 0.07 * np.sin(TAU * 2 * f * t + 0.5) * np.exp(-t / 0.8)
         + 0.02 * np.sin(TAU * 3 * f * t + 1.3) * np.exp(-t / 0.4))
    a = int(0.012 * SR)
    s[:a] *= rc_ramp(a)
    r = int(0.6 * SR)
    s[-r:] *= rc_ramp(r)[::-1]
    return s * vel


def bass_note(f, length, att, rel, decay=None):
    n = int(length * SR)
    t = np.arange(n) / SR
    env = np.ones(n)
    na, nr = int(att * SR), int(rel * SR)
    na = min(na, n // 2)
    nr = min(nr, n - na)
    env[:na] = rc_ramp(na)
    env[n - nr:] *= rc_ramp(nr)[::-1]
    if decay is not None:
        env *= np.exp(-t / decay)
    # harmonics so the root is audible on small speakers; still very soft
    s = np.sin(TAU * f * t) + 0.28 * np.sin(TAU * 2 * f * t + 0.4) + 0.07 * np.sin(TAU * 3 * f * t + 0.9)
    return s * env


# ----------------------------------------------------------------------------- arp patterns
def pattern_index(kind, s, n):
    if kind == "up":
        return s % n
    if kind == "down":
        return (n - 1) - (s % n)
    if kind == "updown":
        period = 2 * n - 2
        k = s % period
        return k if k < n else period - k
    # "broken": 0 2 1 3 2 4 3 5 ...
    return (s // 2) % max(1, n - 2) + (2 if s % 2 else 0)


def melody_pool(name):
    _, pad, arp = CHORDS[name]
    pcs = {m % 12 for m in pad + arp}
    return [m for m in range(66, 82) if m % 12 in pcs]


# ----------------------------------------------------------------------------- render
def render():
    events = build_timeline()
    rng_pad = np.random.default_rng(11)
    note_count = dict(pad=0, pluck=0, bell=0, bass=0)
    mel_prev = 74

    for ei, ev in enumerate(events):
        sec = SECTIONS[ev["sec"]]
        si = ev["sec"]
        t0, t1 = ev["t0"], ev["t1"]
        dur = t1 - t0
        bass_m, pad_notes, arp_tones = CHORDS[ev["name"]]
        last = ei == len(events) - 1
        first = ei == 0

        # ---------------- pads
        notes = list(pad_notes)
        if not sec["top"]:
            notes = [m for m in notes if m <= 66]
        att = 3.0 if first else (1.3 if dur < 4.0 else 2.2)
        rel = 12.0 if last else 3.2
        length = dur + (0 if last else rel)
        if last:
            length = DUR - t0 + 0.5
        g = PAD_GAIN * sec["pad"] / np.sqrt(len(notes))
        for m in notes:
            f = mtof(m)
            reg = (220.0 / f) ** 0.15      # keep low voices from getting muddy
            for s, pan in pad_note(f, length, att, rel, sec["bright"], rng_pad):
                add(s, t0, pan * sec["width"] * 0.8, g * reg, 0.40, "pad")
            note_count["pad"] += 1
        if sec["sparkle"] > 0:   # soft octave-up doubling of the top chord tone
            f = mtof(max(pad_notes) + 12)
            for s, pan in pad_note(f, length, att + 0.8, rel, 0.15, rng_pad):
                add(s, t0, pan * sec["width"], sec["sparkle"] * g * 0.55, 0.65, "pad")

        # ---------------- bass
        if sec["bass"] and t0 + 1e-6 >= sec.get("bass_from", 0.0):
            fb = mtof(bass_m)
            bg = BASS_GAIN * sec["bass_gain"]
            if sec["bass"] == "hold" or dur < 4.0:
                blen = dur + (8.0 if last else 1.8)
                if last:
                    blen = DUR - t0 + 0.5
                add(bass_note(fb, blen, 0.9, 6.0 if last else 1.8), t0, 0.0, bg, 0.06, "bass")
                note_count["bass"] += 1
            else:
                second = bass_m + 7 if sec["bass"] == "walk" else bass_m
                if second > 47:
                    second -= 12
                half = dur / 2
                for k, m in enumerate((bass_m, second)):
                    add(bass_note(mtof(m), half + 2.2, 0.05, 2.2, decay=2.6), t0 + k * half, 0.0,
                        bg * (1.5 if k == 0 else 1.25), 0.06, "bass")
                    note_count["bass"] += 1

        # ---------------- arpeggio
        rng = np.random.default_rng(1000 + ei)
        lo, hi = sec["span"]
        tones = [m + sec["shift"] for m in arp_tones[lo:hi]]
        nslots = ev["beats"] * 2
        sec_t0, sec_t1 = BOUNDS[si], BOUNDS[si + 1]
        for s in range(nslots):
            if sec["grid"] == 1 and s % 2 == 1:
                continue
            ts = t0 + s * BEAT / 2
            frac = (ts - sec_t0) / (sec_t1 - sec_t0)
            d0, d1 = sec["arp"]
            dens = d0 + (d1 - d0) * frac
            if sec["name"] == "ending":
                # thin out from the ending start to nothing 10 s after the Dmaj9 arrives
                dens = max(0.0, 0.35 * (1 - (ts - BOUNDS[9]) / (RESOLVE_BEAT * BEAT + 10.0 - BOUNDS[9])))
            if sec["name"] == "intro" and ts < 6.0:
                continue
            w = 1.0 if s % 4 == 0 else (0.75 if s % 2 == 0 else 0.5)
            p = dens * w
            if s == 0 and dens >= 0.2:
                p = 1.0
            if rng.random() >= p:
                continue
            idx = pattern_index(sec["pattern"], s if sec["grid"] == 2 else s // 2, len(tones))
            m = tones[idx]
            if m > 88:
                m -= 12
            f = mtof(m)
            vel = (0.95 if s % 8 == 0 else 0.72) * rng.uniform(0.82, 1.0)
            jitter = rng.uniform(-0.008, 0.008)
            pan = sec["width"] * (np.clip((m - 68) / 14.0, -1, 1) * 0.45 + rng.uniform(-0.15, 0.15))
            add(pluck(f, vel, rng), ts + jitter, pan, PLUCK_GAIN * sec["arp_gain"], 0.55, "pluck")
            note_count["pluck"] += 1

        # ---------------- bell melody (sparse, smooth random walk on chord tones)
        if sec["bell"] > 0:
            pool = melody_pool(ev["name"])
            for b in range(0, ev["beats"], 2):
                if rng.random() >= sec["bell"] * (1.4 if b == 0 else 0.6):
                    continue
                near = sorted(pool, key=lambda m: abs(m - mel_prev) + rng.uniform(0, 2.5))
                m = near[0] if near[0] != mel_prev or len(near) < 2 else near[1]
                mel_prev = m
                ts = t0 + b * BEAT + rng.uniform(-0.01, 0.01)
                add(bell(mtof(m), rng.uniform(0.75, 0.95), rng), ts, rng.uniform(-0.35, 0.35), BELL_GAIN, 0.75, "bell")
                note_count["bell"] += 1

    # ---------------- ending: resolved, falling bells on the final Dmaj9 (D5+D4, A4, F#4; 4 beats apart)
    tr = RESOLVE_BEAT * BEAT
    for ts, m, v in ((tr, 74, 0.95), (tr, 62, 0.60), (tr + 4 * BEAT, 69, 0.75), (tr + 8 * BEAT, 66, 0.65)):
        add(bell(mtof(m), v, None), ts, 0.15 if m > 70 else -0.15, BELL_GAIN * 0.9, 0.8, "bell")
        note_count["bell"] += 1
    return events, note_count


def make_ir(seed, rt60_bands=((0, 700, 3.6), (700, 2500, 2.6), (2500, 9000, 1.2)), length=4.5):
    rng = np.random.default_rng(seed)
    n = int(length * SR)
    t = np.arange(n) / SR
    ir = np.zeros(n)
    for lo, hi, rt in rt60_bands:
        noise = rng.standard_normal(n)
        if lo <= 0:
            sos = signal.butter(4, hi, "lowpass", fs=SR, output="sos")
        else:
            sos = signal.butter(4, [lo, hi], "bandpass", fs=SR, output="sos")
        band = signal.sosfilt(sos, noise)
        ir += band * np.exp(-6.91 * t / rt)
    pre = int(0.022 * SR)               # pre-delay
    ir = np.concatenate([np.zeros(pre), ir])
    fi = int(0.035 * SR)                # soft onset: no transient in the tail
    ir[pre:pre + fi] *= rc_ramp(fi)
    ir[-int(0.3 * SR):] *= rc_ramp(int(0.3 * SR))[::-1]
    return ir / np.sqrt(np.sum(ir ** 2))


def limiter(x, ceiling, look=0.008, hold=0.12):
    """Look-ahead peak limiter: gain <= ceiling/|x| everywhere, smooth envelope."""
    peak = np.max(np.abs(x), axis=0)
    need = np.minimum(1.0, ceiling / np.maximum(peak, 1e-12))
    if need.min() >= 1.0:
        return x, 0.0, 0.0
    w = int(2 * hold * SR) | 1
    g = minimum_filter1d(need, size=w, mode="nearest")
    ws = int(2 * look * SR) | 1
    k = np.hanning(ws + 2)[1:-1]
    k /= k.sum()
    g = signal.oaconvolve(g, k, mode="same")  # smoothing window shorter than min window
    g = np.minimum(g, need)                 # guarantee (numerical safety)
    gr_db = -20 * np.log10(g.min())
    frac = float(np.mean(g < 10 ** (-1 / 20)))
    return x * g[None, :], gr_db, frac


def print_timeline(events):
    print(f"video end {VIDEO_END:.2f} s | file {DUR:.2f} s | Dmaj9 resolution {RESOLVE_BEAT * BEAT:.2f} s")
    for i, sec in enumerate(SECTIONS):
        evs = [e for e in events if e["sec"] == i]
        print(f"  {sec['name']:<9} {BOUNDS[i]:7.2f}-{BOUNDS[i + 1]:7.2f}  "
              + " ".join(f"{e['name']}({e['beats']})" for e in evs))


def write_meta():
    import json
    meta = dict(_meta=dict(source="src/make_music.py", note="beat-snapped music sections; seconds"),
                bpm=BPM, sr=SR, dur_s=round(DUR, 4), video_end_s=round(VIDEO_END, 4),
                resolve_s=round(RESOLVE_BEAT * BEAT, 4),
                scene_starts_s=[round(b * BEAT, 4) for b in SCENE_BEATS[:9]],
                sections=[dict(name=s["name"], t0=round(BOUNDS[i], 4), t1=round(BOUNDS[i + 1], 4))
                          for i, s in enumerate(SECTIONS)])
    with open(OUT_META, "w") as f:
        json.dump(meta, f, indent=1)
    print("wrote", OUT_META, flush=True)


def main():
    events, counts = render()
    print_timeline(events)
    print("chords:", len(events), "notes:", counts, flush=True)

    # reverb (decorrelated L/R impulses)
    irL, irR = make_ir(7), make_ir(8)
    wetL = signal.oaconvolve(send[0].astype(np.float64), irL)[:N]
    wetR = signal.oaconvolve(send[1].astype(np.float64), irR)[:N]
    dry = sum(STEMS.values()).astype(np.float64)
    mix = dry + 0.55 * np.stack([wetL, wetR])
    wet = 0.55 * np.stack([wetL, wetR])

    # tone shaping: remove sub-rumble and anything harsh
    hp = signal.butter(2, 35, "highpass", fs=SR, output="sos")
    lp = signal.butter(4, 6500, "lowpass", fs=SR, output="sos")
    mix = signal.sosfilt(lp, signal.sosfilt(hp, mix, axis=1), axis=1)

    # fades
    fin, fout = int(4.0 * SR), int(6.0 * SR)
    mix[:, :fin] *= rc_ramp(fin)
    mix[:, N - fout:] *= rc_ramp(fout)[::-1]

    # loudness: RMS of the body (10 s .. DUR-10 s) to -24 dBFS, then peak-limit at -6 dBFS
    body = mix[:, int(10 * SR):int((DUR - 10) * SR)]
    rms = np.sqrt(np.mean(body ** 2))
    gain = 10 ** (-24 / 20) / rms
    mix *= gain

    def sdb(a):
        return 20 * np.log10(max(np.sqrt(np.mean(np.asarray(a, np.float64) ** 2)), 1e-12))
    print("section     " + "".join(f"{k:>8}" for k in list(STEMS) + ["wet", "mix"]) + "   (RMS dBFS, final scale)")
    for i in range(len(SECTIONS)):
        a, b = int(BOUNDS[i] * SR), int(BOUNDS[i + 1] * SR)
        row = [sdb(STEMS[k][:, a:b] * gain) for k in STEMS] + [sdb(wet[:, a:b] * gain), sdb(mix[:, a:b])]
        print(f"{SECTIONS[i]['name']:<12}" + "".join(f"{v:8.1f}" for v in row), flush=True)
    del wet, dry
    pre_peak = 20 * np.log10(np.max(np.abs(mix)))
    mix, gr, frac = limiter(mix, 10 ** (-6.2 / 20))
    print(f"pre-limit peak {pre_peak:.2f} dBFS; limiter max GR {gr:.2f} dB, "
          f"time with GR>1dB {100 * frac:.2f}%", flush=True)

    # 16-bit with TPDF dither
    rng = np.random.default_rng(3)
    dith = (rng.random(mix.shape) - rng.random(mix.shape))
    pcm = np.clip(np.round(mix * 32767 + dith), -32768, 32767).astype(np.int16)
    wavfile.write(OUT, SR, pcm.T)
    print("wrote", OUT, pcm.shape, flush=True)

    # preview: 470..530 s with short fades
    a, b = int(470 * SR), int(530 * SR)
    pv = mix[:, a:b].copy()
    f = int(1.5 * SR)
    pv[:, :f] *= rc_ramp(f)
    pv[:, -f:] *= rc_ramp(f)[::-1]
    dith = (rng.random(pv.shape) - rng.random(pv.shape))
    pcm = np.clip(np.round(pv * 32767 + dith), -32768, 32767).astype(np.int16)
    wavfile.write(OUT_PREVIEW, SR, pcm.T)
    print("wrote", OUT_PREVIEW, pcm.shape, flush=True)


def encode_m4a():
    import subprocess
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", OUT, "-c:a", "aac", "-b:a", "128k",
                    "-movflags", "+faststart", OUT_M4A], check=True)
    print("wrote", OUT_M4A, f"{os.path.getsize(OUT_M4A) / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    if "--timeline" in sys.argv:
        print_timeline(build_timeline())
        sys.exit(0)
    if "--m4a-only" in sys.argv:
        encode_m4a()
        sys.exit(0)
    main()
    write_meta()
    encode_m4a()
