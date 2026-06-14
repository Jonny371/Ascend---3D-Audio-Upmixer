# Ascend — 3D Height Upmixer (Windows / macOS)

An Upmixer that converts mono / stereo / 5.1 / 7.1 audio into a layered 3D format 
with synthesised **height** channels (Auro 9.1 / 10.1 / 11.1, 5.1.4, 7.1.4).
-
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

### Strength adds reverb to a dry direct signal (0–16)

Every synthesised speaker (surround **and** height) carries its own **dry direct
signal plus room reverb** of the speakers around it. Strength sets how much
reverb is added on top of the always-present dry direct:

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
* The recommended strength is 9 for Small/Medium/Large, and 4 for Movie Mode.

**Proximity reverb.** The reverb a speaker receives is sourced by position: it
gets **~60 % from the nearest speaker below/adjacent** (e.g. *Height Front Left*
← *Front Left*, a back height ← the surround beneath it) and the remaining
**~40 % shared among the rest, weighted by distance** — the further a speaker is,
the less reverb it contributes. Each output uses its own room kernel, so the
channels stay decorrelated. A speaker is never fed the reverb of its own signal.

**Options**
* **Decorrelate** — pushes the steered layer correlations to full independence
  for the widest, most diffuse field (Small/Medium/Large carry 50 %
  decorrelation by default; Movie is fully decorrelated).
* **Spread reverb across nearby speakers** — when on (default), reverb is the
  60 % adjacent / 40 %-by-distance mix described above; off feeds each speaker
  reverb from its single nearest neighbour only.
* **Pure upmix — no reverb / reflections** *(default off)* — outputs **just the
  dry spatial redistribution**: the intact bed plus the surround and height layers,
  with **no proximity reverb, no front/centre reflection send and no LFE bloom**.
  It is the strength-0 field forced on regardless of the strength slider (the
  slider is greyed out while it's active). In this mode the surround/height field
  also drops **all synthetic decorrelation** (the short all-pass that, while
  flat, is technically a set of early reflections), so the heights carry **no
  reflections of any kind**. For **mono / stereo** sources the surround + height
  layer is built from the **recording's own extracted ambience** (the diffuse
  component of the stereo image), placed at its **natural level a fair bit below
  the front** (rather than normalised up to the front level, which made the
  separated content too loud and harsh) — so a dry mono source keeps its
  surrounds/heights silent, as it has no ambience to extract. Multichannel sources
  keep their discrete surrounds/heights, just without the decorrelation — and the
  overhead layer is built from the **direct (primary) component** of the front and
  surround channels rather than the raw channels, so the heights don't inherit the
  mix's baked-in surround reverb. A generated rear zone (e.g. stereo→7.1, or the
  back pair of a 5.1→7.1.4) is given a static, reflection-free rotation so it is
  **not identical** to the side it is derived from.
* **Phase-difference height source** — see *How it works* step 3.
* **Dynamics follow** *(default on)* — the synthesised field (reverb, ambience,
  generated surrounds and heights) is reshaped so its loudness contour tracks the
  **original mix**: where the source dips, the reverb tail is pulled down with it;
  where it swells, the field rises. The gain is recentred to a synth-energy-
  weighted unit mean, so the field's overall level is preserved and only its
  *contour over time* changes — the upmix breathes with the source instead of
  smearing it with a constant wash. The **bed stays bit-exact**; only non-bed
  channels are touched, and only at strength > 0. `dyn_amount` (0–1) sets how far
  it follows — the default (0.4) is a gentle nudge rather than a hard envelope
  match.
* **Steer atmosphere / objects to heights** *(default off)* — an internal,
  heuristic content analyzer (no trained model) that detects **rain, wind, storm,
  helicopters and isolated transient objects** and lifts them into the height
  layer. It uses purely spectral / spatial cues: diffuse (low inter-channel
  coherence) **and** noise-like (high spectral-flatness) energy for rain / wind /
  storm; a periodic low-frequency rotor modulation (6–45 Hz, by envelope
  modulation-spectrum) for helicopters; and for isolated objects, **brief,
  spectrally-compact, off-centre transients** — found with an adaptive onset
  detector (a rise above the recent level, so steady ambience never counts) gated
  by spectral compactness (a high crest factor, so broadband wash like rain/wind
  is excluded). Tonal, centred, correlated content (dialogue, music)
  scores low and is left in place. The detected content drives a **time-varying,
  level-neutral crossfade**: when overhead content is present, each height channel
  morphs toward it; when nothing is detected the heights are untouched. Because it
  crossfades rather than adds, the height layer still obeys the front/surround
  average level and the bed is never touched. `steer_amount` (0–1) sets how
  strongly the heights morph toward the detected content. *This is a creative,
  heuristic effect — detection is approximate, so it is opt-in.*
* **3D Immersive** *(default off)* — a separate pathway built on the analyzer
  above. It steers detected overhead content into the height layer **and**, as it
  rises, **ducks the ear-level bed** — front L/R and every surround — by up to
  **`max bed duck` (default 11 dB)**. The duck is driven by a deliberately
  **slow, sustained** measure of the **diffuse / ambient** energy actually present
  — wind, storm, the recording's **natural reflections**, an **ambient-music**
  wash, a helicopter's rotor wash — with a gentle multi-second attack and release,
  so it only swells in and recedes (never fast or pumping). It **explicitly
  ignores brief transients and dry, coherent content**: isolated object hits,
  dialogue and dry music produce **no duck** at all. The **centre (dialogue) and
  LFE are left at full level**, so voices and low end stay anchored while the
  atmosphere lifts overhead. *This intentionally moves energy off the bed, so
  unlike the other modes the bed is no longer bit-exact while it is engaged.*
* **Dolby Pro Logic decode** — auto-detected; see below.

**Dolby Pro Logic / Surround:** every stereo source is analysed for matrix
encoding (a surround signal hidden in the out-of-phase L−R component). If found,
the **Dolby Pro Logic decode** box auto-ticks — the stereo is decoded with the
classic passive matrix (centre = L+R, a band-limited, Dolby-delayed, 90°
phase-shifted R−L surround) before upmixing, and decorrelation is disabled
(the recovered surround is a real mono signal). You can override the checkbox.

The output is a `WAVE_FORMAT_EXTENSIBLE` file with a correct `dwChannelMask`,
so editors / receivers read the speaker assignment automatically.

## Command line (batch)

```bat
python ascend_cli.py mix.wav -o mix_auro.wav --layout "Auro 9.1 (5.1.4)" --preset Movie --strength 12
python ascend_cli.py *.flac --preset Large           :: batch, auto-named
```

---

## How it works (the DSP, honestly)

Ascend uses a single **reverb-send** engine for every preset (Small / Medium /
Large / **Movie** / Speech), mirroring the Auro-Matic philosophy — keep the
original channels intact and **add** a synthesised 3D environment.

**Movie is a cinema model built on measured theatre acoustics.** Its reverb
tail follows a published octave-band RT60 curve (`CINEMA_RT60`) rather than a
single decay: a **mid-frequency RT60 ≈ 1.0 s** (a ~100k ft³ auditorium target),
the **bass running ~25–35 % longer** (real rooms measure 20–40 % longer in the
bass), and the **highs absorbed faster** (~0.85 s at 2 kHz down to ~0.55 s at
8 kHz, from speech-optimised cinema treatment and air absorption). The tail is
built from a flat-summing octave-band filterbank that decays each band by its
measured RT60.

It is voiced as a **THX-style acoustically-treated auditorium**:

* **Slightly lowered treble** — a gentle high-shelf rolloff on the reverb
  (`reverb_hf_db`), modelling the absorptive wall treatment and air absorption
  that keep a THX room from sounding bright or harsh.
* **Absorptive rear wall** — the screen channels' reflected energy is largely
  absorbed rather than bounced back into the seating, so the front speakers
  contribute **less reverb to the rear/back** layer (`front_rear_absorb`).
* **Absorptive sidewalls (to ear level)** — sidewall reflections are damped, so
  the **side surrounds carry less reverb** (`side_absorb_db`).
* **Room geometry → reverb timing** — the speakers sit at real cinema distances
  (front ≈ 12 m, sides ≈ 13 m, rears ≈ 6 m, heights ≈ 11 m above), so each
  layer's room energy arrives with a **distance-based pre-delay** (`geo_predelay`,
  using the speed of sound): the near rears first, then the heights, fronts and
  the far sidewalls last.
* Early reflections are pushed past ~20 ms (cinemas suppress them; the field is
  diffuse), and the wet field is trimmed ~3 dB at full strength. (A stricter
  ISO 2969 / SMPTE 202M "X-curve" tail is also available in the engine.)

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
the original front never drowns as the speaker count grows — discrete
multichannel content is never touched, and the faithful pure-ambience level
(already well below the front) is left alone. A 7.1 source keeps its side and rear
surround pairs discrete for 7.1.4, or folds them at equal power for the
single-surround Auro layouts; a 5.1 source expanding to 7.1.4 keeps its
surrounds at the sides and generates the rear/back pair from the decorrelated
surrounds.

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
- Feed the output to your Auro-3D / Atmos-capable decoder or DAW for monitoring.
