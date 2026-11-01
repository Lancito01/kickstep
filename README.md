# Kickstep

**A tiny live drum detector for electronic music.** Every 10.7 ms Kickstep says whether a **kick**, a
**clap/snare** (rims included) or a **hi-hat** just started, about 50 ms after the hit, with no lookahead and well under
1 % of a CPU core. It hears the last ~4 s, so it judges each hit by where it sits in the groove as well as by its
sound, without any notion of songs, tempo or bars: DJ blends, crossfades, hard cuts and pitch-riding are fine.

It drives the lights in [Thump](https://github.com/Lancito01/thump), and it's free to use in your own projects
(MIT): visualisers, lighting, VJ tools, games, music analysis.

| | |
|---|---|
| Weights | [`kickstep.thdm`](kickstep.thdm), ~213k float32 weights (850 KB) |
| Input | Mono audio at 48 kHz → 64 log bands (30 Hz – 16 kHz) of a Hann-windowed 2048-point FFT every 512 samples |
| Network | 8 causal dilated 1-D convolutions (96 channels, kernel 3, dilations 1–64, residual), 1×1 to 3 sigmoids; ~4.1 s receptive field |
| Output | Per hop, the probability that each drum started 4 hops earlier; a hit is a rising edge over the drum's threshold (in the file) |
| Code | [`python/kickstep.py`](python/kickstep.py) (NumPy only) and [`csharp/`](csharp) (streaming, one hop at a time) |

## Use it

Python (NumPy only):

```python
import kickstep
model = kickstep.load("kickstep.thdm")
hits = kickstep.detect(model, samples, 48000)  # mono float samples -> [(time_s, "kick" | "snare" | "hihat", confidence)]
```

or from the command line on a 48 kHz 16-bit WAV:

```bash
python python/kickstep.py song.wav kickstep.thdm
```

`kickstep.probabilities(model, kickstep.features(samples))` gives the raw per-hop probabilities if you want your own
peak picking. A light that gets brighter with confidence (and stays dark below the threshold) looks much better than
on/off flashes.

C# (live, hop by hop): [`csharp/DrumModel.cs`](csharp/DrumModel.cs) runs the model one hop at a time;
[`csharp/ModelFeatures.cs`](csharp/ModelFeatures.cs) turns an FFT (NAudio's `Complex[]`) into its 64 bands. Feed
`DrumModel.Step` every 512 samples and read `Probabilities`. Both implementations are checked against the training
code: `python python/test_kickstep.py`.

For live use, any audio API works: keep a 2048-sample window, step it every 512 samples, and Kickstep never needs
audio from the future.

## How good it is

On 116 held-out songs (split by artist), against an offline drum transcriber's labels:

| | F1 | Precision | Recall |
|---|---|---|---|
| Kick | 0.86 | 0.96 | 0.78 |
| Clap / snare | 0.72 | 0.92 | 0.60 |
| Hi-hat | 0.82 | 0.89 | 0.75 |

It errs on the side of not firing: precision is high, and the transcriber itself misses many electronic claps, so
some of the "misses" and "fakes" are the reference's. Weak spots: unusual or heavily processed claps, muted or
filtered pre-drop snares, snare rolls now and then read as hats. It's tuned for electronic music (house, bassline,
trap house, riddim, DnB, pop); acoustic drum kits weren't in its training.

## Training

Distilled offline from three teachers: ~730 electronic tracks labelled by a large offline drum transcriber with full
hindsight; [WaivOps EDM-HSE](https://doi.org/10.5281/zenodo.13769544) house drum loops (CC BY 4.0, with labels
corrected from each loop's drum list); and synthesised 808/909-style grooves with exact labels. Loops and grooves
were laid over the songs, and training crops were built like a DJ set (blends with both tracks at full volume, hard
cuts, ±6 % speed, drums coming and going), so it never counts on a groove lasting. None of the teachers' weights,
code or data are included; the training code and music are private.

## File format (.thdm, little-endian)

`"THDM"`, int32 version (2), int32 delay (hops), float32 gain time constant (s), float32 refractory (s), float32
thresholds[3] (kick, snare, hihat), int32 layer count; then per layer int32 in, out, kernel, dilation, flags (bit 0:
residual, input added to the output after the ReLU), float32 weights[out][in][kernel], bias[out]. Hidden layers are
ReLU, the last one sigmoid; kernel tap k looks (kernel − 1 − k) × dilation hops back. The input to the first layer
is the 64 bands minus a running mean of their average (exponential, the file's time constant), so overall loudness
doesn't matter. Version 1 is the same without the flags field.

## Licence

[MIT](LICENSE): the weights and code. If you use Kickstep, a credit is appreciated; the EDM-HSE loops it learned from
are CC BY 4.0 by WaivOps.
