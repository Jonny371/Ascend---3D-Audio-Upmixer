# Ascend — 3D Height Upmixer (Windows / macOS)

An Audio Upmixer that converts Stereo or Surround Signals into Surround and Immersive Formats.

---

## Install

Ascend installs the same way on **Windows, macOS and Linux**. Unzip the folder,
then run the installer for your system — it detects what's present, installs
anything missing into a private virtual environment (`.venv`, so your system
Python is never touched), and writes a one-click launcher:

| System  | Do this                                             |
|---------|-----------------------------------------------------|
| Windows | double-click **`install_windows.bat`**              |
| macOS   | double-click **`install_macos.command`** (first time: right-click → Open) |
| Linux   | run **`./install_linux.sh`** in a terminal          |

Each installer makes sure Python 3.9+ is available (offering to fetch it via
winget / Homebrew / your distro's package manager if it isn't), installs numpy,
scipy, soundfile and PySide6, and on macOS/Linux also pulls in `libsndfile` and
`ffmpeg` where available. If you already have Python on your PATH you can skip
the bootstrap and just run `python install.py` directly.

`ffmpeg` is optional — if none is found on the system, a bundled build
(`imageio-ffmpeg`) is used automatically, so compressed input still works out of
the box.

## Run the GUI

After installing, launch Ascend with the launcher the installer created in the
folder:

| System  | Launch with        |
|---------|--------------------|
| Windows | **`Ascend.bat`**   |
| macOS   | **`Ascend.command`** |
| Linux   | **`./Ascend.sh`**  |

(Or run it manually with the environment's Python: `.venv/bin/python ascend_gui.py`,
or on Windows `.venv\Scripts\pythonw ascend_gui.py`.)

1. **Choose audio file.** WAV / FLAC / AIFF / W64 are read directly; **MP3,
   AAC, M4A/MP4, AC-3/E-AC-3, DTS / DTS-HD and Dolby TrueHD** (and most other
   media containers) are decoded to full-resolution multichannel PCM through
   **ffmpeg**. If no ffmpeg is on the system, a bundled copy is installed on
   first use (`pip install imageio-ffmpeg`), so it works out of the box.
2. Pick an **output layout** and a **preset**, set **strength** (0–16).
3. Optionally toggle the options below (centre/LFE generation, decorrelation,
   phase-difference height sources, rear-height front mix, Pro Logic decode).
4. **Upmix** → writes a float32 multichannel WAV next to the source.

### Low-memory mode (long files)

The toggle in the **top-right corner** — **"Low-memory mode (segment long files)"**,
on by default — lets Ascend process long material (roughly **8–120 minutes**)
without running out of memory. Instead of holding the whole upmix in RAM, it
processes the file in **6-minute segments** and streams each one straight to the
output file, so peak memory is set by a single segment rather than the entire
movie.

The segments are not simply butt-joined: each one is processed with a few seconds
of **overlap** that both pre-warms its reverb/decorrelation state and is used to
**cross-fade the seam** with its neighbour, so the joins are inaudible. The fade
is chosen per channel — **equal-power** for the synthesised/decorrelated channels
(so the spatial level doesn't dip through the blend) and **linear** for the
coherent bed (so the passthrough channels stay sample-accurate across the seam).
The strength-0 bed remains bit-exact end to end.

Output longer than the 4 GB WAV limit (long multichannel files easily exceed it)
is written automatically as **RF64**, which Audition, Reaper, ffmpeg and most
modern tools read transparently. Short files are processed in one pass exactly as
before. Turn the toggle **off** to force whole-file processing.

### Strength adds reverb to a dry direct signal (0–16) — *Small / Medium / Large / Speech*

For the room presets, every synthesised speaker (surround **and** height) carries
its own **dry direct signal plus room reverb** of the speakers around it. Strength
sets how much reverb is added on top of the always-present dry direct:

* **0** — the surround/height channels are a **pure dry field**: no reverb, no
  reflections, and **no tonal shaping of any kind** (the ceiling-darkening and
  room-treble shelves belong to the reverb, so they are applied to the wet
  signal only). Each channel is a flat, decorrelated copy of its source built
  from a cascade of **Schroeder all-pass sections** — a *true* all-pass, so its
  magnitude response is flat to **~0.01 dB**: it scrambles phase to widen the
  image but leaves the **tone completely uncoloured** (measured flat across
  80 Hz–16 kHz). The **phase-difference height
  sources are still applied** at 0 (see below); they're simply dry.
* **16** — the dry direct **plus the full proximity reverb + reflections**.
* In between, the reverb is added in proportion (e.g. strength 8 ≈ dry + half
  reverb). The dry direct is never removed — strength only adds room.

**Proximity reverb.** The reverb a speaker receives is sourced by position: it
gets **~60 % from the nearest speaker below/adjacent** (e.g. *Height Front Left*
← *Front Left*, a back height ← the surround beneath it) and the remaining
**~40 % shared among the rest, weighted by distance** — the further a speaker is,
the less reverb it contributes. Each output uses its own room kernel, so the
channels stay decorrelated. A speaker is never fed the reverb of its own signal.

### Movie mode — the 3D microphone-array capture engine

The **Movie** preset works differently from the room presets, in two ways.

**1. Strength is the surround/height *mix*, not the reverb.** In Movie mode the
slider sets **how much of the synthesised 3D field is blended in over the original
mix** — **0 = the original mix untouched** (stereo stays stereo; a 5.1/7.1 source
keeps its own surrounds, with no height layer added), **16 = the full 3D upmix**.
The fronts are always the original mix, bit-for-bit. Reverb is no longer tied to
the slider — it lives entirely in the **3D Reverb Environment** toggle.

**2. The surround + height field is REDISTRIBUTED from the source, not added on
top.** Following the energy-preserving of parametric time-frequency
upmixing, Movie mode does not pile a synthesised
ambience/reverb over the original. Instead:

* **Coherence-driven extraction.** The diffuse ambience is pulled out by
  per-band, short-frame **inter-channel coherence** (0 = decorrelated/ambient …
  1 = coherent/direct), so direct content is left alone and the bass (mono/room
  modes) is largely untouched.
* **Energy-preserving lift.** A fraction of that ambience energy (set by strength)
  is **moved** into the surround/height layer — decorrelated per channel by a
  random-phase all-pass (unique seed each, so heights never collapse to a phantom)
  — and **the same energy is removed from the fronts** (√-law matched). Total
  energy is conserved: the mix never gets louder, the direct image stays intact,
  the sound just opens up around it. Surround/height levels are therefore
  **proportional to how diffuse the source actually is**, not forced to a fixed
  level.
* **Auro-Matic height tilt.** Heights get a high-frequency tilt (ceiling-reflected
  sound reaches us with an HF emphasis).
* **5.1 / 7.1+ sources keep every discrete channel dry**; only a gentle height
  layer is added, derived from the front-minus-surround **phase subtraction** fed
  through the same coherence extraction and tied to the surround level. Strength
  scales the amount; **0 = original**.

### Natural — capture the untouched source

The **Natural** toggle switches off the decorrelation / matrix / phase-difference
upmix techniques entirely and feeds the **whole, untouched source** into the same
3D-capture engine — i.e. it renders your stereo (or surround) recording *as if it
had been re-recorded by a 3D microphone array and reproduced*, rather than
extracting an ambience to spread around. Pair it with **3D Reverb Environment**
for the room. The Movie preset always uses the 3D-capture engine; Natural simply
feeds it the full signal instead of the extracted diffuse field, and it can be
enabled with any preset.

### Strength — *Small / Medium / Large / Speech* (continued)

**Options**
* **Widening** — pushes the steered layer correlations to full independence
  for the widest, most diffuse field (Small/Medium/Large carry 50 %
  decorrelation by default; Movie is fully decorrelated).
* **Dolby Pro Logic decode** — auto-detected matrix-encoded sources.


The output is a `WAVE_FORMAT_EXTENSIBLE` file with a correct `dwChannelMask`,
so editors / receivers read the speaker assignment automatically.

## Command line (batch)

```bat
python ascend_cli.py mix.wav -o mix_auro.wav --layout "Auro 9.1 (5.1.4)" --preset Movie --strength 12
python ascend_cli.py *.flac --preset Large           :: batch, auto-named
```

---

## How it works

Ascend uses a single **reverb-send** engine for every preset (Small / Medium /
Large / **Movie** / Speech), mirroring the Auro-Matic philosophy — keep the
original channels intact and **add** a synthesised 3D environment.


### The reverb-send engine

Signal flow:

1. **Primary / ambient decomposition** — STFT-domain, coherence-based soft
   masking. Per time–frequency bin it estimates the L/R coherence γ; coherent
   bins (direct, pannable sources) are *primary*, decorrelated bins (room
   tone, reverb, applause) are *ambient*. Temporal smoothing avoids
   musical-noise artefacts. *(Avendano & Jot, ICASSP 2002.)*
2. **Centre generation** — a discrete centre is derived from the coherent
   (primary) mid signal, high-passed so sub energy stays in mains/LFE.
3. **Phase-difference height sources** (option, default on, applied at **all**
   strengths) — for discrete 5.1/7.1 input the height reverb sends are fed by
   *difference* signals that isolate the content unique to each layer:
   * **Front heights** ← `L − Ls`, `R − Rs` (front minus the primary/rear
     surround pair), so the front height carries only front-specific content.
   * **Rear/surround heights (7.1)** ← additionally `Ls − Lss`, `Rs − Rss`
     (back surround minus side surround), isolating the back-specific content.
   These subtractions are present even at strength 0 (the dry field is then a
   decorrelated copy of the difference signal).
4. **Surround + height synthesis** — every synthesised speaker is built the
   same way: its **dry direct signal plus proximity reverb**, the reverb added
   scaled by strength (silent at 0). **The reverb source bed is unified across all
   input types**: it is always the **front pair (+ centre)**, so the room
   character is identical whether the source is mono, stereo, 5.1 or 7.1 — the
   discrete surrounds still play in the dry/direct path but no longer add their own
   (channel-count-dependent) reverb excitation. The reverb a speaker receives is
   **~60 % from the front speaker nearest it and ~40 % from the rest, weighted by
   distance**. Each output convolves its proximity-weighted front mix with its own
   room kernel, so the channels stay decorrelated, and **no speaker is fed the
   reverb of its own signal**.
   * **Front heights** ← dry from the front (phase-diff `L − Ls`, `R − Rs`),
     reverb from *Front L/R* below them.
   * **Back/surround heights** ← dry from the surround beneath them (7.1 also
     subtracts the side surround), reverb from the front by proximity.
   * **Dry/direct still differs by source** (this is intentional, not reverb): a
     stereo upmix synthesises its surrounds/heights from the front, while a 5.1/7.1
     upmix passes its discrete **bed through intact at unity** (centre, LFE and the
     surround bed stay discrete) and a **5.1 → 7.1.4** keeps the input surrounds at
     the **side** and **generates the rear/back** from the decorrelated input
     surrounds. Only the *reverb* is now source-independent.
   * **Front / centre reverb send** → the screen channels also excite the room,
     so L/R and C get a **reduced** reverb send (a preset `front_reverb_db` below
     their dry level, the centre kept ~3 dB lower to protect dialogue) **added on
     top of the intact direct signal**, scaling with strength.
   Each generated layer is RMS-normalised to a fixed ratio of the front level,
   so the level balance matches the reference regardless of programme spectrum.
   The **height layer is then levelled per row**: the **front heights match the
   front level**, and the **rear / side / overhead heights sit midway between the
   front and surround levels** — so the overhead image is anchored to the front
   stage up top and tapers back toward the surround level behind you, at any
   strength.
   * **The bed is never touched** → clip protection scales only the synthesised
     channels, so the passthrough bed (L/R, and C/LFE/surround bed when present)
     is bit-exact at strength 0 and carries only its intended reverb send above
     it. If your source itself is hot, the bed passes through at that level
     rather than being pulled down.
5. **Height & surround coloration** — height channels get an **RBJ high-shelf
   cut** to emulate ceiling-reflection darkening and soften localisation; the
   Top "Voice of God" channel is a darker decorrelated mono diffuse feed. The
   synthesised **surrounds** get a gentle high-shelf **lift** so they stay as
   bright as the fronts (a reverb send otherwise loses a little HF and reads as
   "veiled"). The reverb keeps a high air band (~20 kHz) and a clear early
   window so it is bright and defined over the low-frequency bloom.
6. **Strength / presets** — **strength (0–16) sets how much proximity reverb is
   added** to the always-present dry direct; the reverb
   parameters are fixed by the preset and not user-editable. The room presets
   (Small / Medium / Large) are matched to **measured Auro-Matic T30** data
   (Integra DRX-8.4): a long low-frequency bloom below ~180 Hz over a tight,
   clear midrange/HF tail; low-frequency RT60 ≈ **Small 0.6 / Medium 0.65 /
   Large 1.0 s**. **Movie** instead follows the **measured-cinema octave-band
   RT60 curve** described above (~1.0 s mid, longer bass, faster highs).
   **Small / Medium / Large** apply **50 % decorrelation** by default for a wider
   field; **Movie** is fully decorrelated.

## Output formats

Every preset can target any layout: plain **5.1** and **7.1** (bed + surrounds,
no heights), **Auro 9.1 (5.1.4)**, **Auro 10.1**
(adds the Top "Voice of God" channel), **Auro 11.1** (adds Top + Centre
Height), **5.1.4 (side surrounds)**, **7.1.4**, and **9.1.6** (16-channel: nine
ear-level + LFE + six heights). Stereo, 5.1, 7.1, and **12- and 16-channel
immersive** inputs are all accepted.

When the surround/height layer is **synthesised from a mono/stereo source**, its
summed energy is held below the front pair (a front-dominance ceiling, ~9 dB) so
the original front never drowns as the speaker count grows. The same protection
applies to **pure surround→immersive** conversions (~6 dB margin), where a
discrete surround pair fanned across many positions plus the overhead layer would
otherwise overpower the front. Discrete multichannel content in normal mode is
never touched, and the faithful pure-ambience level (already well below the
front) is left alone.

On top of that, the synthesised **height** layer is levelled to match the bed:
**all height channels** — front and rear/"surround" alike — are set to the
**front L/R** level, so every overhead speaker is equally loud. The match is by
loudness and is peak-capped, so a height channel is never *taller* than its
reference even when reverb/decorrelation make it peakier. The reference excludes
the centre, so a loud centre's dialogue/score no longer inflates the heights — and
the centre is also kept out of the height reverb bed, so its reverb no longer
bleeds overhead. The **centre** itself is lifted **+1.5 dB** in every mode for a
little more dialogue presence.

Both normal and **pure** (no-reverb) upmixes build the heights with the same
**front-minus-surround phase subtraction**, done in the **time domain**
(`height = front − pd·surround`, decorrelated) — never an STFT primary/ambient
extraction, whose spectral smearing reads as room reflections. Pure mode is simply
this dry source with the reverb send switched off, so the heights stay
reflection-free yet still follow the height principle and the bed's own character
(matching the floor channels). A continuous level band then keeps the field gliding
with the bed instead of jumping. A 7.1 source keeps its side and rear
surround pairs discrete for 7.1.4, or folds them at equal power for the
single-surround Auro layouts; a 5.1 source expanding to 7.1.4 keeps its
surrounds at the sides and generates the rear/back pair from the decorrelated
surrounds. For a **stereo (or mono) source**, where the surround field is fully
synthesised, the ear-level surrounds are set to **half the front level (−6 dB)**
for an enveloping upmix (multichannel sources, which have discrete surrounds, are
unaffected). The synthesised **height** layer is always trimmed a further **−3 dB**
below its matched reference so the overhead sits just under the ear-level layer.

**Immersive (12–16 ch) inputs.** A 7.1.4 (12 ch) or 9.1.6 (16 ch) source carries
its own discrete height channels; those are **passed through untouched** to the
matching output positions (bit-exact at strength 0). Expanding a 7.1.4 source to
**9.1.6** generates the two additional pairs — front **wide** and top **middle**
— from the adjacent existing pairs using the phase subtraction plus a *slight*
inter-pair decorrelation, so each new pair widens a little without losing its
direct content. When a 12–16-channel mix is folded into an **Auro** layout with a
Voice-of-God, the **in-phase (coherent) content common to the height channels is
extracted up to the Top/VoG** — exactly the way the centre is pulled from a
stereo pair.

The **Decorrelate** option pushes Small/Medium/Large from their default 50 %
toward full decorrelation for the widest, most diffuse field; it affects only the
**synthesised** channels — a discrete bed (and discrete input heights) always
passes through untouched. **Movie** is voiced as a longer, lusher,
fully-decorrelated hall on the same engine.

## Channel order

Channels are written in ascending `WAVEFORMATEXTENSIBLE` speaker-bit order,
e.g. **Auro 9.1**: `L R C LFE Ls Rs Height-L Height-R Height-Ls Height-Rs`.
Each layout's exact order and mask are printed by the CLI and shown in the GUI.

## Notes / limits

- Whole file is processed in memory (fine for normal tracks; very long files
  use proportional RAM).
- Best results from real stereo with genuine decorrelated ambience; heavily
  mono or dry material yields a subtler height field (as with any upmixer).
- Feed the output to your Auro-3D / Atmos-capable encoder/decoder or DAW for monitoring.
