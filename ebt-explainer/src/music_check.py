"""Checks for video/music.wav: levels, clipping, discontinuities, harsh highs, spectrogram, ffmpeg.

Usage: python3 -I src/music_check.py [spectrogram_png]
"""
import os
import subprocess
import sys
import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAV = os.path.join(ROOT, "video", "music.wav")
PREVIEW = os.path.join(ROOT, "video", "music_preview_60s.wav")
SCRATCH = "/tmp/claude-0/-home-user-Random/9843a8a1-7b0a-5ed7-8867-6762f4a2ea11/scratchpad/ebt"
PNG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRATCH, "music_spec.png")
BOUNDS = [0, 50, 125, 200, 260, 340, 410, 485, 555, 570, 630]
NAMES = ["intro", "families", "energy", "data", "thinking", "training", "eval", "scaling", "wrap", "ending"]
META = os.path.join(ROOT, "video", "music_meta.json")   # written by make_music.py (re-timed bounds)
if os.path.exists(META):
    import json
    with open(META) as _f:
        _secs = json.load(_f)["sections"]
    BOUNDS = [s["t0"] for s in _secs] + [_secs[-1]["t1"]]
    NAMES = [s["name"] for s in _secs]


def db(x):
    return 20 * np.log10(max(float(x), 1e-12))


def main():
    sr, pcm = wavfile.read(WAV)
    assert pcm.dtype == np.int16 and pcm.ndim == 2 and pcm.shape[1] == 2, (pcm.dtype, pcm.shape)
    x = pcm.astype(np.float64) / 32768.0
    n = len(x)
    print(f"file: {WAV}\n  sr={sr} Hz, channels=2, 16-bit, duration={n / sr:.3f} s, "
          f"size={os.path.getsize(WAV) / 1e6:.1f} MB")

    peak = np.max(np.abs(x))
    rms_all = np.sqrt(np.mean(x ** 2))
    body = x[int(10 * sr):n - 10 * sr]
    rms_body = np.sqrt(np.mean(body ** 2))
    print(f"  peak {db(peak):.2f} dBFS | RMS whole {db(rms_all):.2f} dBFS | RMS 10 s..end-10 s {db(rms_body):.2f} dBFS"
          f" | crest {db(peak) - db(rms_body):.1f} dB")
    clip = int(np.sum(np.abs(pcm.astype(np.int32)) >= 32767))
    print(f"  samples at full scale: {clip}")

    # discontinuities: biggest first and second differences, and where
    d1 = np.abs(np.diff(x, axis=0))
    d2 = np.abs(np.diff(x, n=2, axis=0))
    i1 = np.unravel_index(np.argmax(d1), d1.shape)
    print(f"  max sample-to-sample jump {d1.max():.5f} FS ({d1.max() * 32768:.0f} LSB) at t={i1[0] / sr:.3f} s"
          f" | max 2nd difference {d2.max():.6f} FS")
    # a full-scale-peak sine at 1 kHz would jump 2*pi*1000*peak/sr per sample:
    print(f"  (reference: a {db(peak):.1f} dBFS 1 kHz sine jumps {2 * np.pi * 1000 * peak / sr:.5f} FS per sample)")

    # click detector: energy above 9 kHz, frame-wise, peak vs median
    sos = signal.butter(6, 9000, "highpass", fs=sr, output="sos")
    hpx = signal.sosfilt(sos, x.mean(axis=1))
    fr = int(0.01 * sr)
    m = (len(hpx) // fr) * fr
    fe = np.sqrt(np.mean(hpx[:m].reshape(-1, fr) ** 2, axis=1))
    print(f"  >9 kHz content: max abs {db(np.max(np.abs(hpx))):.1f} dBFS, frame RMS max {db(fe.max()):.1f} dBFS,"
          f" median {db(np.median(fe)):.1f} dBFS (dither floor about -101 dBFS)")

    # long-term spectrum
    f, P = signal.welch(x.mean(axis=1)[int(10 * sr):n - 10 * sr], fs=sr, nperseg=16384)
    tot = P.sum()
    for fc in (2000, 4000, 8000, 12000):
        print(f"  energy above {fc / 1000:g} kHz: {10 * np.log10(P[f > fc].sum() / tot):.1f} dB rel. total")

    # per-section levels, stereo correlation
    print("  section          t0-t1       RMS dBFS  peak dBFS  L/R corr")
    sec_rms = []
    for i, nm in enumerate(NAMES):
        seg = x[int(BOUNDS[i] * sr):int(BOUNDS[i + 1] * sr)]
        r = np.sqrt(np.mean(seg ** 2))
        c = np.corrcoef(seg[:, 0], seg[:, 1])[0, 1]
        sec_rms.append(db(r))
        print(f"  {nm:<12} {BOUNDS[i]:>4}-{BOUNDS[i + 1]:<4}     {db(r):7.2f}   {db(np.max(np.abs(seg))):7.2f}"
              f"    {c:.3f}")

    # ---------------- figure
    mono = x.mean(axis=1)
    fig = plt.figure(figsize=(16, 13), dpi=100)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 1.0, 0.9], hspace=0.35, wspace=0.18)
    ax = fig.add_subplot(gs[0, :])
    ff, tt, S = signal.spectrogram(mono, fs=sr, nperseg=4096, noverlap=2048, window="hann", mode="psd")
    keep = ff <= 16000
    im = ax.pcolormesh(tt, ff[keep] / 1000, 10 * np.log10(S[keep] + 1e-16), shading="auto",
                       cmap="magma", vmin=-150, vmax=-50, rasterized=True)
    for b in BOUNDS[1:-1]:
        ax.axvline(b, color="w", lw=0.6, alpha=0.6)
    ax.set_title("Full track spectrogram (0-16 kHz, linear frequency). White lines: scene boundaries")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("kHz")
    fig.colorbar(im, ax=ax, label="dB")

    ax = fig.add_subplot(gs[1, 0])
    a, b = int(485 * sr), int(515 * sr)
    ff2, tt2, S2 = signal.spectrogram(mono[a:b], fs=sr, nperseg=8192, noverlap=7168, mode="psd")
    k2 = ff2 <= 4000
    ax.pcolormesh(tt2 + 485, ff2[k2], 10 * np.log10(S2[k2] + 1e-16), shading="auto", cmap="magma",
                  vmin=-140, vmax=-50, rasterized=True)
    ax.set_yscale("symlog", linthresh=200)
    ax.set_ylim(30, 4000)
    ax.set_title("Zoom 485-515 s (scaling section), 0-4 kHz")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Hz")

    ax = fig.add_subplot(gs[1, 1])
    ax.semilogx(f[1:], 10 * np.log10(P[1:] + 1e-20), color="#ffc95c")
    ax.set_xlim(20, 22050)
    ax.set_ylim(-170, -40)
    ax.grid(alpha=0.3, which="both")
    ax.set_title("Long-term average spectrum (10 s to end-10 s)")
    ax.set_xlabel("Hz")
    ax.set_ylabel("dB/Hz")

    ax = fig.add_subplot(gs[2, :])
    hop = sr // 2
    mm = (len(mono) // hop) * hop
    blocks = x[:mm].reshape(-1, hop, 2)
    rms_t = 20 * np.log10(np.sqrt(np.mean(blocks ** 2, axis=(1, 2))) + 1e-9)
    pk_t = 20 * np.log10(np.max(np.abs(blocks), axis=(1, 2)) + 1e-9)
    tb = np.arange(len(rms_t)) * 0.5
    ax.plot(tb, pk_t, color="#ff8f7a", lw=0.8, label="peak (0.5 s blocks)")
    ax.plot(tb, rms_t, color="#74e0c0", lw=1.0, label="RMS (0.5 s blocks)")
    ax.axhline(-6, color="#ff8f7a", ls="--", lw=0.8)
    ax.axhline(-24, color="#74e0c0", ls="--", lw=0.8)
    for i, bnd in enumerate(BOUNDS[:-1]):
        ax.axvline(bnd, color="k", lw=0.5, alpha=0.4)
        ax.text(bnd + 2, -12, NAMES[i], fontsize=9)
    ax.set_ylim(-60, 0)
    ax.set_xlim(0, n / sr)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("dBFS")
    ax.legend(loc="lower center", ncol=2)
    ax.set_title("Level over time (dashed: -6 dBFS peak and -24 dBFS RMS targets)")
    fig.savefig(PNG)
    print("  spectrogram ->", PNG)

    # ---------------- ffmpeg acceptance test (30 s AAC)
    import imageio_ffmpeg
    ff_bin = imageio_ffmpeg.get_ffmpeg_exe()
    test = os.path.join(SCRATCH, "music_test_30s.m4a")
    r = subprocess.run([ff_bin, "-y", "-loglevel", "error", "-ss", "470", "-t", "30", "-i", WAV,
                        "-c:a", "aac", "-b:a", "160k", test], capture_output=True, text=True)
    print(f"  ffmpeg AAC encode rc={r.returncode} {r.stderr.strip()[:300]}")
    r2 = subprocess.run([ff_bin, "-hide_banner", "-i", test], capture_output=True, text=True)
    info = [ln.strip() for ln in r2.stderr.splitlines() if "Duration" in ln or "Audio:" in ln]
    print("  ", " | ".join(info), f"| {os.path.getsize(test) / 1e3:.0f} kB")
    r3 = subprocess.run([ff_bin, "-hide_banner", "-i", PREVIEW], capture_output=True, text=True)
    info = [ln.strip() for ln in r3.stderr.splitlines() if "Duration" in ln or "Audio:" in ln]
    print("  preview:", " | ".join(info))


if __name__ == "__main__":
    main()
