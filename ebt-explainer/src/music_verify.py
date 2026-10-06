"""Independent verifier for video/music.wav, music_preview_60s.wav and music.m4a.

Checks: format/length, peak/RMS/true-peak, clipping, DC, fades, silence gaps, mono
compatibility, and clicks with three detectors (raw sample jumps vs local slope,
LPC-residual spikes, >10 kHz transient frames). Also decodes the m4a and compares it
to the WAV. Numpy/scipy only, so it runs under `python3 -I`.

Usage: python3 -I src/music_verify.py
"""
import os
import json
import subprocess
import sys
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.linalg import solve_toeplitz
from scipy.ndimage import median_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAV = os.path.join(ROOT, "video", "music.wav")
PREVIEW = os.path.join(ROOT, "video", "music_preview_60s.wav")
M4A = os.path.join(ROOT, "video", "music.m4a")
META = os.path.join(ROOT, "video", "music_meta.json")
with open(META) as _f:
    _meta = json.load(_f)
BOUNDS = [sec["t0"] for sec in _meta["sections"]] + [_meta["sections"][-1]["t1"]]
DUR = _meta["dur_s"]
FAILS = []


def db(v):
    return 20 * np.log10(max(float(v), 1e-12))


def check(cond, msg):
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def lpc_residual_spikes(x, sr, order=24, block_s=0.5, thr=12.0, floor=16 / 32768):
    """Fit an LPC model per 0.5 s window (hop 0.25 s) and flag residual samples in the window's
    central half that exceed thr robust sigmas AND an absolute floor (16 LSB). Only the centre
    is scored because residuals at window edges are unreliable (edge-of-window false alarms)."""
    hits = []
    worst = (0.0, 0.0)
    B = int(block_s * sr)
    q0, q1 = B // 4, 3 * B // 4
    for c in range(x.shape[1]):
        ch = x[:, c]
        for b0 in range(0, len(ch) - B, B // 2):
            seg = ch[b0:b0 + B]
            if np.max(np.abs(seg)) < 1e-4:      # near-silent (fades): skip, dither dominates
                continue
            w = seg * np.hanning(B)
            spec = np.fft.rfft(w, 2 * B)
            r = np.fft.irfft(np.abs(spec) ** 2)[:order + 1]
            r[0] *= 1.0 + 1e-9
            a = solve_toeplitz(r[:order], r[1:order + 1])
            pred = signal.lfilter(np.concatenate([[0.0], a]), [1.0], seg)
            e = (seg - pred)[q0:q1]
            s = 1.4826 * np.median(np.abs(e - np.median(e))) + 1e-12
            z = np.abs(e) / s
            k = int(np.argmax(z))
            if z[k] > worst[0]:
                worst = (float(z[k]), (b0 + q0 + k) / sr)
            if z[k] > thr and abs(e[k]) > floor:
                hits.append(((b0 + q0 + k) / sr, c, float(z[k])))
    return hits, worst


def main():
    print("== music.wav")
    sr, pcm = wavfile.read(WAV)
    check(sr == 44100, f"sample rate {sr}")
    check(pcm.dtype == np.int16, f"dtype {pcm.dtype}")
    check(pcm.ndim == 2 and pcm.shape[1] == 2, f"channels {pcm.shape}")
    dur = len(pcm) / sr
    check(abs(dur - DUR) < 1e-3, f"duration {dur:.4f} s (meta {DUR} s, video end {_meta['video_end_s']} s)")
    x = pcm.astype(np.float64) / 32768.0

    peak = np.max(np.abs(x))
    clip = int(np.sum(np.abs(pcm.astype(np.int32)) >= 32767))
    check(clip == 0, f"samples at full scale: {clip}")
    check(db(peak) <= -6.0, f"sample peak {db(peak):.2f} dBFS (<= -6)")
    # true peak via 4x oversampling on the loudest 20 s
    hop = sr * 20
    loud = max(range(0, len(x) - hop, hop), key=lambda i: np.max(np.abs(x[i:i + hop])))
    up = signal.resample_poly(x[loud:loud + hop], 4, 1, axis=0)
    check(db(np.max(np.abs(up))) <= -5.0, f"true peak (4x) {db(np.max(np.abs(up))):.2f} dBFS")
    end10 = len(x) - 10 * sr
    rms_body = np.sqrt(np.mean(x[10 * sr:end10] ** 2))
    print(f"  info RMS 10 s..end-10 s {db(rms_body):.2f} dBFS, whole {db(np.sqrt(np.mean(x ** 2))):.2f} dBFS")
    check(-27 <= db(rms_body) <= -20, "body RMS within -27..-20 dBFS (quiet bed)")
    dc = np.mean(x, axis=0)
    check(np.all(np.abs(dc) < 1e-4), f"DC offset L {dc[0]:.2e} R {dc[1]:.2e}")

    # per-section levels and mono compatibility
    print("  section   RMS dBFS  mono-sum loss dB  L/R corr")
    for i in range(len(BOUNDS) - 1):
        seg = x[int(BOUNDS[i] * sr):int(BOUNDS[i + 1] * sr)]
        st = np.sqrt(np.mean(seg ** 2))
        mono = np.sqrt(np.mean(seg.mean(axis=1) ** 2))
        corr = np.corrcoef(seg[:, 0], seg[:, 1])[0, 1]
        print(f"  {BOUNDS[i]:6.1f}-{BOUNDS[i + 1]:<6.1f} {db(st):8.2f} {db(mono) - db(st):12.2f} {corr:12.3f}")
    mono_loss = db(np.sqrt(np.mean(x[10 * sr:end10].mean(axis=1) ** 2))) - db(rms_body)
    check(mono_loss > -6, f"mono fold-down loss {mono_loss:.2f} dB (> -6 dB, no big phase cancellation)")

    # fades
    first = np.max(np.abs(x[:int(0.05 * sr)]))
    last = np.max(np.abs(x[-int(0.05 * sr):]))
    check(first < 1e-3 and last < 1e-3, f"start/end 50 ms peak {db(first):.1f} / {db(last):.1f} dBFS")
    blk = int(0.5 * sr)
    rms_blocks = lambda a: np.array([np.sqrt(np.mean(a[i:i + blk] ** 2)) for i in range(0, len(a) - blk + 1, blk)])
    fin = rms_blocks(x[:5 * sr])
    fout = rms_blocks(x[-7 * sr:])
    print("  fade-in RMS (0.5 s blocks):", " ".join(f"{db(v):.0f}" for v in fin))
    print("  fade-out RMS (0.5 s blocks):", " ".join(f"{db(v):.0f}" for v in fout))
    check(db(fin[0]) < db(fin[-1]) - 15, "fade-in ramps up")
    check(db(fout[-1]) < db(fout[0]) - 20, "fade-out ramps down")

    # silence gaps in the body (1 s windows below -50 dBFS)
    w = sr
    r1 = np.array([np.sqrt(np.mean(x[i:i + w] ** 2)) for i in range(5 * sr, end10, w)])
    gaps = np.where(20 * np.log10(r1 + 1e-12) < -50)[0]
    check(len(gaps) == 0, f"no near-silent 1 s windows in 5 s..end-10 s (found {len(gaps)})")
    jumps = np.diff(20 * np.log10(r1 + 1e-12))
    j = int(np.argmax(np.abs(jumps)))
    check(np.max(np.abs(jumps)) < 6, f"largest 1 s RMS step {jumps[j]:+.2f} dB at ~{5 + j + 1} s")

    # ---- clicks 1: sample jump vs local slope (jump / rolling-max of local |d1|)
    d1 = np.abs(np.diff(x, axis=0)).max(axis=1)
    print(f"  info max sample-to-sample jump {d1.max():.5f} FS ({d1.max() * 32768:.0f} LSB) at {np.argmax(d1) / sr:.3f} s")
    # local reference: median of |d1| per 10 ms block, max with the two neighbouring blocks
    fb = int(0.01 * sr)
    nb = len(d1) // fb
    bm = np.median(d1[:nb * fb].reshape(nb, fb), axis=1)
    bm = np.maximum(bm, np.maximum(np.r_[bm[1:], bm[-1]], np.r_[bm[0], bm[:-1]]))
    ref = np.repeat(np.r_[bm, bm[-1]], fb)[:len(d1)]
    ratio = d1 / (ref + 2.0 / 32768)
    k = int(np.argmax(ratio))
    check(ratio[k] < 15, f"max jump/local-median ratio {ratio[k]:.1f} at {k / sr:.3f} s (< 15)")

    # ---- clicks 2: LPC residual spikes
    hits, worst = lpc_residual_spikes(x, sr)
    check(len(hits) == 0, f"LPC residual spikes > 12 sigma: {len(hits)}; worst {worst[0]:.1f} sigma at {worst[1]:.3f} s")
    for h in hits[:10]:
        print(f"     hit t={h[0]:.3f} s ch={h[1]} z={h[2]:.1f}")

    # ---- clicks 3: >10 kHz transient frames vs 1 s rolling median
    sos = signal.butter(8, 10000, "highpass", fs=sr, output="sos")
    hp = signal.sosfilt(sos, x, axis=0)
    fr = int(0.005 * sr)
    m = (len(hp) // fr) * fr
    fe = np.sqrt(np.mean(hp[:m].reshape(-1, fr, 2) ** 2, axis=(1, 2)))
    med = median_filter(fe, size=201, mode="nearest")
    rr = 20 * np.log10((fe + 1e-12) / (med + 1e-12))
    k = int(np.argmax(rr))
    check(rr[k] < 10, f"max >10 kHz frame vs local median {rr[k]:.1f} dB at {k * 0.005:.3f} s (< 10 dB)")
    print(f"  info >10 kHz level: median {db(np.median(fe)):.1f} dBFS, max {db(fe.max()):.1f} dBFS")

    # chord boundaries line up with scene boundaries (beats)
    beat = 60.0 / 72.0
    check(all(abs(b / beat - round(b / beat)) < 1e-3 for b in BOUNDS), "all section bounds on whole beats at 72 BPM")
    vend, res = _meta["video_end_s"], _meta["resolve_s"]
    check(0 < vend - res <= 15, f"final Dmaj9 at {res:.2f} s lands {vend - res:.1f} s before the video end ({vend:.2f} s)")
    check(dur >= vend + 20, f"music outlasts the video by {dur - vend:.1f} s (ffmpeg -shortest keeps the whole video)")

    # ---- preview
    print("== music_preview_60s.wav")
    sr2, pv = wavfile.read(PREVIEW)
    check(sr2 == 44100 and pv.dtype == np.int16 and pv.shape == (60 * sr, 2), f"preview format {sr2} {pv.dtype} {pv.shape}")
    pvf = pv.astype(np.float64) / 32768
    seg = x[470 * sr:530 * sr]
    mid = slice(5 * sr, 55 * sr)
    err = np.max(np.abs(pvf[mid] - seg[mid]))
    check(err < 4 / 32768, f"preview equals music.wav 470-530 s in the middle (max diff {err * 32768:.1f} LSB)")
    check(np.max(np.abs(pvf[:int(0.02 * sr)])) < 1e-3 and np.max(np.abs(pvf[-int(0.02 * sr):])) < 1e-3,
          "preview starts and ends silent (fades)")

    # ---- m4a
    print("== music.m4a")
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    r = subprocess.run([ff, "-v", "error", "-i", M4A, "-f", "s16le", "-ac", "2", "-ar", "44100", "-"],
                       capture_output=True)
    check(r.returncode == 0 and not r.stderr, f"m4a decodes cleanly (rc={r.returncode}, stderr={r.stderr[:200]!r})")
    a = np.frombuffer(r.stdout, np.int16).reshape(-1, 2).astype(np.float64) / 32768
    check(abs(len(a) / sr - dur) < 0.05, f"m4a decoded duration {len(a) / sr:.4f} s")
    # alignment: cross-correlate a 5 s chunk at 300 s
    c0 = 300 * sr
    ref_c = x[c0:c0 + 5 * sr, 0]
    cc =signal.correlate(a[c0 - 4096:c0 + 5 * sr + 4096, 0], ref_c, mode="valid", method="fft")
    lag = int(np.argmax(cc)) - 4096
    check(lag == 0, f"m4a sample-aligned with WAV (lag {lag} samples)")
    diff = a[10 * sr:end10] - x[10 * sr:end10]
    snr = db(np.sqrt(np.mean(x[10 * sr:end10] ** 2))) - db(np.sqrt(np.mean(diff ** 2)))
    print(f"  info m4a vs wav SNR {snr:.1f} dB, m4a peak {db(np.max(np.abs(a))):.2f} dBFS")
    check(np.max(np.abs(a)) < 0.99, "m4a decoded peak below full scale")

    print()
    print("RESULT:", "ALL PASS" if not FAILS else f"{len(FAILS)} FAIL(S): " + "; ".join(FAILS))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
