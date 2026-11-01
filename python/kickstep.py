"""Kickstep reference inference in NumPy: audio in, kick / clap-snare / hi-hat hits out.

    import kickstep
    model = kickstep.load("kickstep.thdm")
    hits = kickstep.detect(model, samples, 48000)   # mono float samples -> [(time_s, "kick", confidence), ...]

Runs a whole recording at once and gives exactly what the streaming version gives hop by hop (it is causal: nothing
at time t depends on audio after t). Only NumPy is needed.
"""
import struct

import numpy as np

DRUMS = ("kick", "snare", "hihat")
RATE, FFT, HOP, BANDS, LO, HI = 48000, 2048, 512, 64, 30.0, 16000.0


def load(path):
    """Read a .thdm file (version 1 or 2)."""
    with open(path, "rb") as f:
        b = f.read()
    assert b[:4] == b"THDM", "not a Kickstep model"
    version, delay, tau, refractory = struct.unpack_from("<iiff", b, 4)
    thresholds = struct.unpack_from("<3f", b, 20)
    (n,), o = struct.unpack_from("<i", b, 32), 36
    layers = []
    for _ in range(n):
        cin, cout, k, dil = struct.unpack_from("<4i", b, o)
        o += 16
        residual = False
        if version >= 2:
            residual = bool(struct.unpack_from("<i", b, o)[0] & 1)
            o += 4
        w = np.frombuffer(b, "<f4", cout * cin * k, o).reshape(cout, cin, k)
        o += w.nbytes
        bias = np.frombuffer(b, "<f4", cout, o)
        o += bias.nbytes
        layers.append((w, bias, dil, residual))
    return dict(layers=layers, delay=delay, tau=tau, refractory=refractory, thresholds=thresholds)


def features(samples, rate=RATE):
    """(hops, 64) log band magnitudes: Hann-windowed 2048-point FFT every 512 samples at 48 kHz, 64 bands evenly
    spaced in pitch from 30 Hz to 16 kHz. Resample to 48 kHz first. An overall gain doesn't matter (the model removes
    the running level), so mono mixdown and FFT scaling conventions don't either."""
    assert rate == RATE, "resample to 48 kHz first"
    # Hop i is the 2048 samples ending at sample (i + 1) * 512, silence before the start (as live).
    x = np.r_[np.zeros(FFT - HOP), np.asarray(samples, np.float64)]
    hops = (len(x) - FFT) // HOP + 1
    frames = np.lib.stride_tricks.sliding_window_view(x, FFT)[::HOP][:hops] * np.hanning(FFT)
    mag = np.abs(np.fft.rfft(frames, axis=1)) / FFT
    edges = LO * (HI / LO) ** (np.arange(BANDS + 1) / BANDS)
    lo = np.clip(np.round(edges[:-1] / (RATE / FFT)).astype(int), 1, FFT // 2 - 1)
    hi = np.clip(np.round(edges[1:] / (RATE / FFT)).astype(int) - 1, lo, FFT // 2 - 1)
    out = np.stack([mag[:, a:b + 1].mean(axis=1) for a, b in zip(lo, hi)], axis=1)
    return np.log(1e-7 + out).astype(np.float32)


def probabilities(model, feats, hop_s=HOP / RATE):
    """(hops, 3) probability that each drum started `delay` hops before each hop."""
    level = feats.mean(axis=1)
    a, m, run = 1 - np.exp(-hop_s / model["tau"]), level[0], np.empty_like(level)
    for i, v in enumerate(level):  # the same causal running level as live
        m += a * (v - m)
        run[i] = m
    x = (feats - run[:, None]).T.astype(np.float32)  # (bands, hops)
    for j, (w, bias, dil, residual) in enumerate(model["layers"]):
        k = w.shape[2]
        pad = np.pad(x, ((0, 0), ((k - 1) * dil, 0)))
        y = bias[:, None] + sum(w[:, :, t] @ pad[:, t * dil:t * dil + x.shape[1]] for t in range(k))
        if j < len(model["layers"]) - 1:
            y = np.maximum(y, 0) + (x if residual else 0)
        x = y
    return (1 / (1 + np.exp(-x))).T


def detect(model, samples, rate=RATE, sensitivity=(1, 1, 1)):
    """Hits as (time in s, drum, confidence): a rising edge over the drum's threshold, at most one per refractory
    period. Sensitivity > 1 catches more (each doubling lowers the threshold by 2 log-odds), < 1 fewer."""
    p = probabilities(model, features(samples, rate))
    hop_s, hits = HOP / RATE, []
    for d, name in enumerate(DRUMS):
        th = model["thresholds"][d]
        logit = np.log(th / (1 - th)) - 2 * np.log(max(0.05, sensitivity[d]))
        above = p[:, d] >= 1 / (1 + np.exp(-logit))
        last = -1.0
        for i in np.flatnonzero(above & ~np.r_[False, above[:-1]]):
            t = (i - model["delay"]) * hop_s  # back to when the hit started, as Thump does
            if t - last >= model["refractory"]:
                hits.append((round(float(t), 4), name, float(p[i, d])))
                last = t
    return sorted(hits)


if __name__ == "__main__":
    import sys
    import wave
    with wave.open(sys.argv[1]) as w:  # 16-bit PCM WAV at 48 kHz
        a = np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, w.getnchannels()).mean(axis=1) / 32768
    for hit in detect(load(sys.argv[2] if len(sys.argv) > 2 else "kickstep.thdm"), a, w.getframerate()):
        print(*hit, sep=",")
