"""
Ascend — 3D Height Upmixer  ::  DSP engine  (v2)
================================================
Independent, original implementation of an Auro-Matic-style upmixer.
NOT Auro Technologies' code; built from published techniques (see README).

v2 changes
----------
* Zone-aware surround routing. 7.1 input now keeps its SIDE and REAR surround
  pairs DISCRETE instead of collapsing them into one synthesised feed, so
  side<->rear pans transition correctly. Folding to fewer output zones is
  equal-power; expanding fills the empty zone with decorrelated diffuse.
* Selectable INPUT channel order (WAV/Microsoft, SMPTE/Atmos, Film/Pro Tools)
  to stop WAV-vs-film 7.1 ordering from smearing the surround image.
* Frequency-dependent "hall" reverb: a long velvet-noise tail whose highs
  decay faster than its lows (two-band), plus predelay and correlation-
  controlled L/R sends. The new "Auro Hall" preset is fitted to a real
  Auro-Matic Pro render (RT60 ~2.9 s, HF tilt ~-6 dB, surr/front ~0.54,
  front L/R corr ~0.93, surround corr ~0.48).
"""
from __future__ import annotations
import struct
import numpy as np
from scipy import signal

# --------------------------------------------------------------------------
SPK = {
    "FL": 0x1, "FR": 0x2, "FC": 0x4, "LFE": 0x8, "BL": 0x10, "BR": 0x20,
    "FLC": 0x40, "FRC": 0x80, "BC": 0x100, "SL": 0x200, "SR": 0x400,
    "TC": 0x800, "TFL": 0x1000, "TFC": 0x2000, "TFR": 0x4000,
    "TBL": 0x8000, "TBC": 0x10000, "TBR": 0x20000,
    # extended keys for 9.1.6: front "wide" pair and top-middle (top-side) pair
    "FLW": 0x40000, "FRW": 0x80000, "TSL": 0x100000, "TSR": 0x200000,
}

LAYOUTS = {
    "5.1": [
        ("L", "FL"), ("R", "FR"), ("C", "FC"), ("LFE", "LFE"),
        ("Ls", "BL"), ("Rs", "BR")],
    "7.1": [
        ("L", "FL"), ("R", "FR"), ("C", "FC"), ("LFE", "LFE"),
        ("Lrs", "BL"), ("Rrs", "BR"), ("Lss", "SL"), ("Rss", "SR")],
    "Auro 9.1 (5.1.4)": [
        ("L", "FL"), ("R", "FR"), ("C", "FC"), ("LFE", "LFE"),
        ("Ls", "BL"), ("Rs", "BR"),
        ("Height L", "TFL"), ("Height R", "TFR"),
        ("Height Ls", "TBL"), ("Height Rs", "TBR")],
    "Auro 10.1 (5.1.4 + Top)": [
        ("L", "FL"), ("R", "FR"), ("C", "FC"), ("LFE", "LFE"),
        ("Ls", "BL"), ("Rs", "BR"), ("Top (VoG)", "TC"),
        ("Height L", "TFL"), ("Height R", "TFR"),
        ("Height Ls", "TBL"), ("Height Rs", "TBR")],
    "Auro 11.1 (5.1.4 + Top + CH)": [
        ("L", "FL"), ("R", "FR"), ("C", "FC"), ("LFE", "LFE"),
        ("Ls", "BL"), ("Rs", "BR"), ("Top (VoG)", "TC"),
        ("Height L", "TFL"), ("Center Height", "TFC"), ("Height R", "TFR"),
        ("Height Ls", "TBL"), ("Height Rs", "TBR")],
    "5.1.4 (side surrounds)": [
        ("L", "FL"), ("R", "FR"), ("C", "FC"), ("LFE", "LFE"),
        ("Ls", "SL"), ("Rs", "SR"),
        ("Top FL", "TFL"), ("Top FR", "TFR"),
        ("Top BL", "TBL"), ("Top BR", "TBR")],
    "7.1.4": [
        ("L", "FL"), ("R", "FR"), ("C", "FC"), ("LFE", "LFE"),
        ("Lrs", "BL"), ("Rrs", "BR"), ("Lss", "SL"), ("Rss", "SR"),
        ("Top FL", "TFL"), ("Top FR", "TFR"),
        ("Top BL", "TBL"), ("Top BR", "TBR")],
    "9.1.6": [
        ("L", "FL"), ("R", "FR"), ("C", "FC"), ("LFE", "LFE"),
        ("Lrs", "BL"), ("Rrs", "BR"), ("Lss", "SL"), ("Rss", "SR"),
        ("Lw", "FLW"), ("Rw", "FRW"),
        ("Top FL", "TFL"), ("Top FR", "TFR"),
        ("Top ML", "TSL"), ("Top MR", "TSR"),
        ("Top BL", "TBL"), ("Top BR", "TBR")],
}

# Input channel-order maps. SSL/SSR = side surround L/R, RSL/RSR = rear surr L/R.
ORDERS_8 = {
    "WAV / Microsoft (L R C LFE BL BR SL SR)":
        dict(L=0, R=1, C=2, LFE=3, RSL=4, RSR=5, SSL=6, SSR=7),
    "SMPTE / Atmos (L R C LFE Lss Rss Lrs Rrs)":
        dict(L=0, R=1, C=2, LFE=3, SSL=4, SSR=5, RSL=6, RSR=7),
    "Film / Pro Tools (L C R Lss Rss Lrs Rrs LFE)":
        dict(L=0, C=1, R=2, SSL=3, SSR=4, RSL=5, RSR=6, LFE=7),
}
ORDERS_6 = {
    "WAV / Microsoft (L R C LFE Ls Rs)":
        dict(L=0, R=1, C=2, LFE=3, RSL=4, RSR=5),
    "Film / Pro Tools (L C R Ls Rs LFE)":
        dict(L=0, C=1, R=2, RSL=3, RSR=4, LFE=5),
}
# 12-channel inputs (7.1.4): adds the four input HEIGHT roles
#   HFL/HFR = top front L/R, HBL/HBR = top back L/R
ORDERS_12 = {
    "7.1.4 WAV (L R C LFE BL BR SL SR Ltf Rtf Ltb Rtb)":
        dict(L=0, R=1, C=2, LFE=3, RSL=4, RSR=5, SSL=6, SSR=7,
             HFL=8, HFR=9, HBL=10, HBR=11),
    "7.1.4 Atmos (L R C LFE Lss Rss Lrs Rrs Ltf Rtf Ltb Rtb)":
        dict(L=0, R=1, C=2, LFE=3, SSL=4, SSR=5, RSL=6, RSR=7,
             HFL=8, HFR=9, HBL=10, HBR=11),
}
# 16-channel inputs (9.1.6): adds front WIDE (WL/WR) and a top-MIDDLE height
# pair (HML/HMR) on top of the 7.1.4 height set
ORDERS_16 = {
    "9.1.6 WAV (… BL BR SL SR Lw Rw Ltf Rtf Ltm Rtm Ltb Rtb)":
        dict(L=0, R=1, C=2, LFE=3, RSL=4, RSR=5, SSL=6, SSR=7,
             WL=8, WR=9, HFL=10, HFR=11, HML=12, HMR=13, HBL=14, HBR=15),
}

# --------------------------------------------------------------------------
# Presets.  Small/Medium/Large are fitted to the Auro-Matic 5.1.4 room samples
# (measured RT60 ~1.6 / 2.7 / 3.3 s, dry fronts kept intact, a dark tail, and a
# rear/surround height layer that runs hotter than the front heights).
#
# Reverb keys: length_ms, density, decay_ms, predelay_ms, hf_decay_ratio,
# xover_hz, surr_gain, height_gain (front heights), rear_height_gain (back
# heights), center_amt, surr_corr.  front_reverb=0 keeps the front L/R as the
# untouched original (the room samples measured front L/R correlation = 1.00).
# rear_height_gain=None falls back to height_gain.
# --------------------------------------------------------------------------

# Measured-style cinema reverberation curve (octave-band RT60, seconds).
# Anchored to published theatre data: a ~1.0 s mid-frequency RT60 (a ~100k ft³
# auditorium target), the bass running ~25-35 % longer (real rooms measure
# 20-40 % longer in the bass — e.g. opera-house data rising from ~1.1 s at
# 4 kHz to ~1.8 s at 125 Hz), and the highs absorbed faster in a speech-
# optimised cinema (and by air absorption up top).  Tmid (500/1k avg) ≈ 1.0 s.
CINEMA_RT60 = {63: 1.35, 125: 1.25, 250: 1.12, 500: 1.05, 1000: 1.00,
               2000: 0.85, 4000: 0.70, 8000: 0.55, 16000: 0.42}


def _p(**kw):
    base = dict(predelay_ms=8, hf_decay_ratio=0.7, xover_hz=2500,
                front_reverb=0.0, front_corr=0.9, surr_corr=0.5,
                rear_height_gain=None, height_corr=0.35, rear_height_corr=None,
                air_hz=9000.0, clarity=0.0, decorr=0.0, wet_scale=1.0,
                lfe_gain_db=0.0, height_scale=1.0, decorr_bed=0.0,
                decorr_height=None, front_center_removal=False, surr_dry_wet=None,
                rear_height_dry=0.6, rear_height_front_mix=0.35,
                surr_hf_db=3.0, wet_trim_db=0.0, xcurve=False, front_reverb_db=None,
                xover_order=2, rt60_curve=None, prox_near=0.6, prox_power=1.5,
                reverb_hf_db=0.0, geo_predelay=False, front_rear_absorb=1.0,
                side_absorb_db=0.0)
    base.update(kw)
    return base

PRESETS = {
    # --- fitted to the measured Auro-Matic T30 reverb times (Integra DRX-8.4,
    #     strength 8): a long low-frequency bloom below ~180 Hz (the headline
    #     numbers) over a tight, flat ~0.4 s midrange/HF tail.  decay_ms sets the
    #     sub-180 Hz RT60; hf_decay_ratio brings the band above the crossover to
    #     ~0.4 s.  Small/Medium/Large carry 50% decorrelation by default.
    #     low/mid RT60 targets:  Small 0.61/0.42  Medium 0.65/0.42  Large 1.02/0.45
    #     air_hz is kept high (≈20 kHz) and clarity > 0 so the synthesised
    #     surrounds stay as bright as the fronts (un-muffled) with a clear early
    #     window over the LF-bloom tail, matching the reference render.
    "Small":  _p(diffuse=1.05, length_ms=750, density=1600, decay_ms=92,
                 er_ms=(6, 40),  surr_gain=0.55, height_gain=0.45,
                 rear_height_gain=1.06, tilt_db=3.0, center_amt=0.60,
                 predelay_ms=4, hf_decay_ratio=0.70, xover_hz=180, surr_corr=0.55,
                 height_corr=0.20, rear_height_corr=0.70, air_hz=20000.0,
                 clarity=0.30, decorr=0.50),
    "Medium": _p(diffuse=1.05, length_ms=800, density=1600, decay_ms=94,
                 er_ms=(8, 45),  surr_gain=0.52, height_gain=0.47,
                 rear_height_gain=1.10, tilt_db=2.5, center_amt=0.58,
                 predelay_ms=5, hf_decay_ratio=0.66, xover_hz=180, surr_corr=0.66,
                 height_corr=0.65, rear_height_corr=0.86, air_hz=20000.0,
                 clarity=0.30, decorr=0.50),
    "Large":  _p(diffuse=1.05, length_ms=1200, density=1600, decay_ms=148,
                 er_ms=(10, 55), surr_gain=0.44, height_gain=0.41,
                 rear_height_gain=0.89, tilt_db=2.5, center_amt=0.62,
                 predelay_ms=6, hf_decay_ratio=0.45, xover_hz=180, surr_corr=0.60,
                 height_corr=0.58, rear_height_corr=0.64, air_hz=20000.0,
                 clarity=0.30, decorr=0.50),
    # --- Movie = a CINEMA model built on MEASURED theatre acoustics.  The reverb
    #     tail follows a published octave-band RT60 curve (CINEMA_RT60): ~1.0 s at
    #     mid frequencies, the bass running ~25-35 % longer, and the highs
    #     absorbed faster (2 kHz ~0.85 s down to ~0.55 s at 8 kHz) — built from a
    #     flat-summing octave-band filterbank rather than a single decay.  Early
    #     reflections are pushed past ~20 ms (cinemas suppress them; the field is
    #     diffuse).  The screen channels get a reduced reverb send, the surrounds
    #     carry the FRONTS' room reverb over their dry direct signal, and the wet
    #     field is trimmed ~3 dB at full strength.
    "Movie":  _p(diffuse=1.05, length_ms=1700, density=2200, decay_ms=145,
                 er_ms=(8, 70),  surr_gain=0.85, height_gain=1.00, tilt_db=2.5,
                 center_amt=0.65, predelay_ms=15, hf_decay_ratio=0.60, xover_hz=1100,
                 xover_order=4, air_hz=19000.0, clarity=0.0, decorr=1.00,
                 wet_scale=0.85, wet_trim_db=-3.0, lfe_gain_db=3.0, xcurve=False,
                 surr_hf_db=0.0, front_reverb_db=-13.0, rt60_curve=CINEMA_RT60,
                 reverb_hf_db=-3.5, geo_predelay=True, front_rear_absorb=0.5,
                 side_absorb_db=-2.5),
    # --- Speech: short, dry, dialogue-focused (no forced decorrelation) --------
    "Speech": _p(diffuse=0.7, length_ms=750, density=1800, decay_ms=86,
                 er_ms=(5, 35),  surr_gain=0.60, height_gain=0.45, tilt_db=4.0,
                 center_amt=0.85, predelay_ms=3, hf_decay_ratio=0.64, xover_hz=180,
                 air_hz=19000.0, clarity=0.20),
}


# ==========================================================================
#  Building blocks
# ==========================================================================
def rbj_high_shelf(x, sr, f0, gain_db, q=0.707):
    if abs(gain_db) < 1e-3:
        return x
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / sr
    cw, sw = np.cos(w0), np.sin(w0)
    al = sw / (2 * q); sa = 2 * np.sqrt(A) * al
    b0 = A * ((A + 1) + (A - 1) * cw + sa)
    b1 = -2 * A * ((A - 1) + (A + 1) * cw)
    b2 = A * ((A + 1) + (A - 1) * cw - sa)
    a0 = (A + 1) - (A - 1) * cw + sa
    a1 = 2 * ((A - 1) - (A + 1) * cw)
    a2 = (A + 1) - (A - 1) * cw - sa
    return signal.lfilter(np.array([b0, b1, b2]) / a0,
                          np.array([1.0, a1 / a0, a2 / a0]), x)


def rbj_low_shelf(x, sr, f0, gain_db, q=0.707):
    if abs(gain_db) < 1e-3:
        return x
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / sr
    cw, sw = np.cos(w0), np.sin(w0)
    al = sw / (2 * q); sa = 2 * np.sqrt(A) * al
    b0 = A * ((A + 1) - (A - 1) * cw + sa)
    b1 = 2 * A * ((A - 1) - (A + 1) * cw)
    b2 = A * ((A + 1) - (A - 1) * cw - sa)
    a0 = (A + 1) + (A - 1) * cw + sa
    a1 = -2 * ((A - 1) + (A + 1) * cw)
    a2 = (A + 1) + (A - 1) * cw - sa
    return signal.lfilter(np.array([b0, b1, b2]) / a0,
                          np.array([1.0, a1 / a0, a2 / a0]), x)


def _xcurve_mag(freqs):
    """ISO 2969 / SMPTE 202M cinema 'X-curve' magnitude (linear): flat to 2 kHz,
    -3 dB/oct from 2-10 kHz (~-7 dB at 10 kHz), -6 dB/oct above 10 kHz.  Models
    the darker, balanced high-frequency response heard two-thirds back in a
    movie theatre."""
    f = np.maximum(freqs, 1.0)
    db = np.zeros_like(f)
    m1 = (f > 2000.0) & (f <= 10000.0)
    db[m1] = -3.0 * np.log2(f[m1] / 2000.0)
    m2 = f > 10000.0
    db[m2] = -3.0 * np.log2(5.0) - 6.0 * np.log2(f[m2] / 10000.0)
    return 10.0 ** (db / 20.0)


def reverb_kernel(sr, length_ms, density, decay_ms, er_ms, seed,
                  predelay_ms=8.0, hf_decay_ratio=0.7, xover_hz=2500.0,
                  air_hz=15000.0, clarity=0.0, xcurve=False, xover_order=2,
                  rt60_curve=None):
    """
    Sparse FIR = predelay + early reflections + frequency-dependent velvet tail.

    If `rt60_curve` (a {centre_Hz: RT60_seconds} dict) is given, the tail is
    built from a flat-summing octave-band filterbank with a PER-BAND decay set
    by the measured RT60 at each band — i.e. a real measured room's reverberation
    curve (longer at the bass, shorter at the top).  Otherwise the legacy 2-band
    model is used: the tail is split at xover_hz, the low band decaying with
    time-constant `decay_ms` and the high band faster (decay_ms*hf_decay_ratio);
    `xover_order` sets the crossover steepness.

    `air_hz` sets the final HF roll-off; `clarity` (0..1) boosts the early window
    relative to the late tail (raising the early/late ratio) without altering the
    decay slope; `xcurve` shapes the kernel to the ISO cinema X-curve.
    """
    rng = np.random.default_rng(seed)
    pre = int(predelay_ms * 1e-3 * sr)
    n = max(8, int(length_ms * 1e-3 * sr))
    total = pre + n

    m = max(1, int(density * length_ms * 1e-3))
    grid = n / m
    imp = np.zeros(total)
    for i in range(m):
        p = pre + int(i * grid + rng.random() * grid)
        if p < total:
            imp[p] += np.sign(rng.random() - 0.5)

    t = np.arange(total) / sr
    base = (t - pre / sr).clip(0)
    if rt60_curve:
        # Measured per-octave RT60.  Split the velvet train into octave bands
        # with cumulative-lowpass differences (these telescope, so the bands sum
        # back to the original train with no crossover ripple), then decay each
        # band by its measured RT60 (tau = RT60 / ln(1e6) = RT60 / 13.8 for a
        # 60 dB decay over e-folds -> tau = RT60/6.908).
        freqs = sorted(rt60_curve)
        xos = [np.sqrt(freqs[i] * freqs[i + 1]) for i in range(len(freqs) - 1)]
        lps = []
        for xo in xos:
            blo, alo = signal.butter(4, min(xo, 0.45 * sr) / (sr / 2.0), btype="low")
            lps.append(signal.lfilter(blo, alo, imp))
        k = np.zeros(total)
        for i, fc in enumerate(freqs):
            if i == 0:
                band = lps[0] if xos else imp
            elif i < len(freqs) - 1:
                band = lps[i] - lps[i - 1]
            else:
                band = imp - lps[-1]
            tau = max(rt60_curve[fc], 0.05) / 6.908
            k = k + band * np.exp(-base / tau)
    else:
        env_lo = np.exp(-base / (decay_ms * 1e-3))
        env_hi = np.exp(-base / (decay_ms * hf_decay_ratio * 1e-3))
        # Complementary split: low band = LP(slow-decay tail); high band = the
        # *complement* of the LP applied to the fast-decay tail (high = x - LP(x)).
        # When the two decays are equal this reconstructs the broadband tail exactly,
        # so there is no magnitude notch at the crossover (the old LP+HP butter sum
        # dipped ~5 dB around xover_hz).
        blo, alo = signal.butter(xover_order, xover_hz / (sr / 2.0), btype="low")
        tail_hi = imp * env_hi
        k = signal.lfilter(blo, alo, imp * env_lo) + (tail_hi - signal.lfilter(blo, alo, tail_hi))

    er0 = pre + int(er_ms[0] * 1e-3 * sr)
    er1 = pre + max(er0 + 1, int(er_ms[1] * 1e-3 * sr))
    for i in range(6):
        ppos = int(rng.integers(er0, min(er1, total - 1) + 1))
        g = (1.0 - i / 6) * (0.6 + 0.4 * rng.random())
        k[ppos] += np.sign(rng.random() - 0.5) * g

    # clarity: lift the early window (predelay .. ~130 ms) so the early/late
    # energy ratio rises while the late decay slope (RT60) is unchanged.
    if clarity > 0:
        ew = pre + int(0.130 * sr)
        k[:ew] *= (1.0 + 5.0 * clarity)

    bh, ah = signal.butter(2, min(air_hz, 0.45 * sr) / (sr / 2.0), btype="low")
    k = signal.lfilter(bh, ah, k)
    if xcurve:
        fk = np.fft.rfftfreq(len(k), 1.0 / sr)
        k = np.fft.irfft(np.fft.rfft(k) * _xcurve_mag(fk), len(k))
    e = np.sqrt(np.sum(k ** 2))
    if e > 1e-9:
        k /= e
    return k


# Speaker directions as (azimuth degrees, elevation degrees); azimuth 0 = front,
# positive = right.  Used by the proximity reverb model so a generated speaker is
# fed mostly the room reverb of the speaker directly below / nearest to it.
SPK_POS = {
    "FL": (-30.0, 0.0), "FR": (30.0, 0.0), "FC": (0.0, 0.0),
    "BL": (-135.0, 0.0), "BR": (135.0, 0.0), "SL": (-90.0, 0.0), "SR": (90.0, 0.0),
    "FLW": (-60.0, 0.0), "FRW": (60.0, 0.0),
    "TFL": (-45.0, 35.0), "TFR": (45.0, 35.0), "TFC": (0.0, 35.0),
    "TBL": (-135.0, 35.0), "TBR": (135.0, 35.0), "TC": (0.0, 90.0),
    "TSL": (-90.0, 45.0), "TSR": (90.0, 45.0),
}


def _spk_angle(p, q):
    """Great-circle angle (degrees) between two (azimuth, elevation) directions."""
    a1, e1, a2, e2 = np.radians([p[0], p[1], q[0], q[1]])
    u = np.array([np.cos(e1) * np.sin(a1), np.cos(e1) * np.cos(a1), np.sin(e1)])
    v = np.array([np.cos(e2) * np.sin(a2), np.cos(e2) * np.cos(a2), np.sin(e2)])
    return float(np.degrees(np.arccos(np.clip(u @ v, -1.0, 1.0))))


def proximity_weights(out_label, src_labels, near=0.6, power=1.5, tol=8.0):
    """Distance-weighted reverb-send weights from each bed source to one output
    speaker.  The nearest source(s) below/adjacent take `near` of the total
    (split on near-ties); the remaining (1-near) is shared among the rest by
    inverse angular-distance**power, so the further a speaker is the less reverb
    it contributes.  A source co-located with the output (same speaker label) is
    excluded — a speaker is never fed the reverb of its own signal."""
    po = SPK_POS.get(out_label)
    idx = [i for i, s in enumerate(src_labels) if s != out_label and s in SPK_POS]
    w = [0.0] * len(src_labels)
    if po is None or not idx:
        return w
    d = {i: max(_spk_angle(po, SPK_POS[src_labels[i]]), 1e-3) for i in idx}
    dmin = min(d.values())
    nearest = [i for i in idx if d[i] <= dmin + tol]
    rest = [i for i in idx if i not in nearest]
    for i in nearest:
        w[i] = near / len(nearest)
    if rest:
        inv = {i: 1.0 / (d[i] ** power) for i in rest}
        s = sum(inv.values())
        for i in rest:
            w[i] = (1.0 - near) * inv[i] / s
    else:
        for i in nearest:
            w[i] = 1.0 / len(nearest)
    return w


# Cinema geometry for the THX-style room model (Movie preset).  Distances are
# the listener-to-speaker spacing in metres; with the speed of sound this sets a
# per-zone reverb pre-delay (a farther speaker's room energy arrives later).
SPEED_OF_SOUND = 343.0
SPK_DIST = {"front": 12.0, "side": 13.0, "rear": 6.0, "height": 11.0}  # metres
SPK_ZONE = {
    "FL": "front", "FR": "front", "FC": "front",
    "BL": "rear", "BR": "rear", "SL": "side", "SR": "side",
    "FLW": "side", "FRW": "side",
    "TFL": "height", "TFR": "height", "TFC": "height",
    "TBL": "height", "TBR": "height", "TC": "height",
    "TSL": "height", "TSR": "height",
}


def primary_ambient_decompose(L, R, sr, diffuse_exp=1.0, smooth_ms=50.0):
    nper = 4096; hop = nper // 4; win = "hann"
    f, t, ZL = signal.stft(L, fs=sr, window=win, nperseg=nper, noverlap=nper - hop)
    _, _, ZR = signal.stft(R, fs=sr, window=win, nperseg=nper, noverlap=nper - hop)
    a = np.exp(-hop / (sr * smooth_ms * 1e-3))
    bsm, asm = [1 - a], [1, -a]
    Sll = signal.lfilter(bsm, asm, np.abs(ZL) ** 2, axis=1)
    Srr = signal.lfilter(bsm, asm, np.abs(ZR) ** 2, axis=1)
    Slr = signal.lfilter(bsm, asm, ZL * np.conj(ZR), axis=1)
    g = np.clip(np.abs(Slr) / (np.sqrt(Sll * Srr) + 1e-9), 0, 1)
    amb = np.clip((1 - g) ** diffuse_exp, 0, 1)
    pri = 1 - amb

    def back(Z):
        _, x = signal.istft(Z, fs=sr, window=win, nperseg=nper, noverlap=nper - hop)
        return x
    pL, pR, aL, aR = back(ZL * pri), back(ZR * pri), back(ZL * amb), back(ZR * amb)
    n = min(len(L), len(pL), len(aL))
    return pL[:n], pR[:n], aL[:n], aR[:n]


def antiphase_common(L, R, sr, smooth_ms=40.0):
    """Extract the component that is present in BOTH channels but phase-inverted
    (L carries +x where R carries -x) — the genuine out-of-phase 'surround' signal
    of a Dolby Surround / matrix encode.  Content that is in-phase (centre) or
    only in one channel (hard-panned) is rejected: only anti-correlated common
    energy is returned.  Done per time/frequency bin so it tracks the mix."""
    nper = 4096; hop = nper // 4; win = "hann"
    f, t, ZL = signal.stft(L, fs=sr, window=win, nperseg=nper, noverlap=nper - hop)
    _, _, ZR = signal.stft(R, fs=sr, window=win, nperseg=nper, noverlap=nper - hop)
    a = np.exp(-hop / (sr * smooth_ms * 1e-3))
    bsm, asm = [1 - a], [1, -a]
    Sll = signal.lfilter(bsm, asm, np.abs(ZL) ** 2, axis=1)
    Srr = signal.lfilter(bsm, asm, np.abs(ZR) ** 2, axis=1)
    Slr = signal.lfilter(bsm, asm, ZL * np.conj(ZR), axis=1)
    # weight = how anti-phase the bin is: 1 when L = -R, 0 when in-phase or when
    # one side is silent (panned).  -Re(cross)/(|L||R|) is the negated correlation.
    w = np.clip(-np.real(Slr) / (np.sqrt(Sll * Srr) + 1e-9), 0.0, 1.0)
    A = w * 0.5 * (ZL - ZR)
    _, x = signal.istft(A, fs=sr, window=win, nperseg=nper, noverlap=nper - hop)
    n = min(len(L), len(x))
    out = np.zeros(len(L))
    out[:n] = x[:n]
    return out


def conv_same(x, k):
    return signal.fftconvolve(x, k, mode="full")[:len(x)]


def allpass_decorr_params(sr, seed, n_stages=4, max_ms=8.0, g=0.7):
    """Delay/gain set for a Schroeder all-pass decorrelation chain (seeded)."""
    rng = np.random.default_rng(seed)
    delays = [max(1, int(rng.uniform(0.4, max_ms) * 1e-3 * sr)) for _ in range(n_stages)]
    gains = [g] * n_stages
    return delays, gains


def schroeder_allpass(x, delays, gains):
    """Cascade of Schroeder all-pass sections  y = -g x + x[n-M] + g y[n-M].
    Each section is a TRUE all-pass (unit magnitude at every frequency), so the
    chain decorrelates a signal (scrambles phase / disperses it in time) while
    leaving its magnitude spectrum — its tone — completely UNCOLOURED.  This is
    what feeds the dry height/surround field."""
    y = x.astype(np.float64)
    for M, g in zip(delays, gains):
        b = np.zeros(M + 1); b[0] = -g; b[M] = 1.0
        a = np.zeros(M + 1); a[0] = 1.0; a[M] = -g
        y = signal.lfilter(b, a, y)
    return y


def decorr_kernel(sr, length_ms, seed, decay_ms=None):
    """Short all-pass-ish decorrelator: randomised phase with a FRONT-LOADED
    (fast-decaying) envelope, so most of the energy sits in the first ~1 ms.
    Convolving two channels with two different such kernels makes them mutually
    decorrelated, but because the response is concentrated at the onset a
    transient stays sharp instead of smearing into an audible reflection — the
    'pure decorrelated, dry' field used at strength 0.  `decay_ms` is the onset
    time-constant (defaults to length_ms/6)."""
    n = max(8, int(sr * length_ms * 1e-3))
    rng = np.random.default_rng(seed)
    half = n // 2 + 1
    phase = rng.uniform(-np.pi, np.pi, half)
    phase[0] = 0.0                       # DC real
    if n % 2 == 0:
        phase[-1] = 0.0                  # Nyquist real
    ir = np.fft.irfft(np.exp(1j * phase), n)
    # front-loaded exponential envelope (energy concentrated at the onset)
    if decay_ms is None:
        decay_ms = length_ms / 6.0
    t = np.arange(n) / sr
    ir *= np.exp(-t / (max(decay_ms, 0.05) * 1e-3))
    e = np.sqrt(np.sum(ir ** 2))
    if e > 1e-9:
        ir /= e
    return ir


def _corr_pair(srcL, srcR, k_shared, k_iL, k_iR, rho):
    """Two reverb feeds with target L/R correlation rho (shared+independent mix)."""
    a, b = np.sqrt(np.clip(rho, 0, 1)), np.sqrt(1 - np.clip(rho, 0, 1))
    sh = conv_same(0.5 * (srcL + srcR), k_shared)
    return a * sh + b * conv_same(srcL, k_iL), a * sh + b * conv_same(srcR, k_iR)


# ==========================================================================
#  STFT panning-decomposition upmix  (after Kraft 2022)
# ==========================================================================
# A principled stereo->3D upmix that decomposes the stereo image per
# time-frequency bin into a *direct* signal (with an estimated pan position)
# and a *decorrelated ambient* signal, then re-pans the direct part onto the
# front L/C/R arc with VBAP and spreads the ambient part into the surround and
# height channels with magnitude-complementary random decorrelation filters.
# Reference: S. Kraft, "Stereo Signal Decomposition and Upmixing to Surround
# and 3D Audio", PhD thesis, Helmut-Schmidt-Universitaet Hamburg, 2022
# (Ch. 4 decomposition, Ch. 5 upmix).  Original re-implementation.

def _shelf_mag(freqs, fc, gain_db):
    """First-order high-shelf magnitude: 1 below fc, 10^(gain_db/20) above."""
    g = 10.0 ** (gain_db / 20.0)
    x = (freqs / max(fc, 1.0)) ** 2
    return np.sqrt((1.0 + (g * g) * x) / (1.0 + x))


def _decorr_mag(freqs, gamma, seed, lo=300.0, hi=10000.0):
    """Magnitude-complementary random response in [0,1] (Fink/Kraft):
    H = atan(gamma*R)/pi + 0.5, with gamma rolled off below `lo` and above `hi`
    so the lowest/highest bands stay correlated (avoids artefacts)."""
    rng = np.random.default_rng(seed)
    R = rng.standard_normal(len(freqs))
    f = np.maximum(freqs, 1e-3)
    roll = (1.0 / (1.0 + (lo / f) ** 2)) * (1.0 / (1.0 + (f / hi) ** 2))
    g = gamma * roll
    return np.arctan(g * R) / np.pi + 0.5


def _proc_band(freqs, lo=110.0, hi=13000.0):
    """Weight 1 inside [lo,hi], smoothly 0 outside.  Bins outside this band
    bypass the decomposition straight to L/R (bass + extreme highs stay put)."""
    f = np.maximum(freqs, 1e-3)
    hp = 1.0 / (1.0 + (lo / f) ** 4)
    lp = 1.0 / (1.0 + (f / hi) ** 4)
    return hp * lp


def kraft_upmix(xL, xR, sr, layout_name, strength=12, *, p=None, progress=None):
    """STFT panning-decomposition upmix of a stereo pair to an Auro/3D layout."""
    if p is None:
        p = {}
    def report(pp, m=""):
        if progress:
            progress(int(pp), m)

    layout = LAYOUTS[layout_name]
    labels = [l for l, _ in layout]
    pos = {key: l for l, key in layout}            # SPK key -> output label
    mask = 0
    for _, key in layout:
        mask |= SPK[key]
    N = len(xL)

    # ---- parameters ----------------------------------------------------
    NFFT = 2048
    HOP = 512
    win = signal.windows.hann(NFFT, sym=False)
    theta0 = np.deg2rad(p.get("width_deg", 30.0))  # stereo width (>=stereo=30)
    s0 = np.sin(theta0)
    gamma = float(p.get("decorr_gamma", 8.0))
    rear_db = float(p.get("rear_hf_db", -12.0))    # HF damping into the rears
    pba_hz = float(p.get("pba_hz", 7000.0))        # height crossover (PBA)
    pba_db = float(p.get("pba_db", -12.0))
    amb_base = float(p.get("amb_gain", 1.0))
    rear_bias = float(p.get("rear_bias", 1.15))    # push energy to rear/height
    amb_gain = amb_base * (0.35 + 0.65 * np.clip(strength / 15.0, 0, 1))

    report(6, "STFT analysis")
    f = np.fft.rfftfreq(NFFT, 1.0 / sr)
    def stft(x):
        return signal.stft(x, sr, window=win, nperseg=NFFT, noverlap=NFFT - HOP,
                           boundary="zeros", padded=True)[2]
    XL = stft(xL.astype(np.float64))
    XR = stft(xR.astype(np.float64))
    nb, nf = XL.shape

    # ---- per-bin panning + direct/ambient decomposition ---------------
    report(22, "Panning + direct/ambient decomposition")
    magL = np.abs(XL); magR = np.abs(XR)
    pw = np.sqrt(magL ** 2 + magR ** 2) + 1e-12
    gL = magL / pw                                  # constant-power coeffs
    gR = magR / pw
    Psi = (magR - magL) / (magL + magR + 1e-12)     # position index in [-1,1]
    # direct/ambient with HAL=1, HAR=exp(j*pi/2)=j  (ambient ICC -> 0)
    det = 1j * gL - gR                              # |det| = 1
    act = (magL + magR) > 1e-7
    S = np.where(act, (1j * XL - XR) / det, 0.0)    # mono direct
    A = np.where(act, (gL * XR - gR * XL) / det, 0.0)  # mono ambient

    # ---- VBAP re-pan of the direct signal onto L / C / R --------------
    report(45, "Re-panning direct signal")
    theta = np.arcsin(np.clip(s0 * Psi, -1.0, 1.0))  # source angle, +=right
    s = np.sin(theta); c = np.cos(theta)
    k = 1.0 / np.tan(theta0)                          # cot(theta0)
    right = theta >= 0
    gpL = np.where(right, 0.0, -s / s0)
    gpR = np.where(right, s / s0, 0.0)
    gpC = np.where(right, c - k * s, c + k * s)
    gpC = np.maximum(gpC, 0.0)
    nrm = np.sqrt(gpL ** 2 + gpC ** 2 + gpR ** 2) + 1e-12
    gpL /= nrm; gpC /= nrm; gpR /= nrm
    DL = gpL * S; DC = gpC * S; DR = gpR * S

    # ---- ambient decorrelation tree -----------------------------------
    report(62, "Decorrelating ambient field")
    fb = f[:, None]                                  # column for broadcasting
    # front L/R: magnitude-complementary + 90 deg phase  -> ICC ~ 0
    H1 = _decorr_mag(f, gamma, 101)[:, None]
    HAL = H1
    HAR = (1.0 - H1) * 1j
    # front/rear split (rear gets an HF shelf so the front keeps presence)
    HRr = (_decorr_mag(f, gamma, 202) * _shelf_mag(f, 8000.0, rear_db))[:, None]
    HFr = 1.0 - HRr
    # height split (Perceptual Band Allocation: highs go up)
    HLo = _shelf_mag(f, pba_hz, pba_db)[:, None]
    HHi = 1.0 - HLo
    # extra decorrelators for VoG / centre-height
    Htc = _decorr_mag(f, gamma, 303)[:, None]
    Htfc = _decorr_mag(f, gamma, 404)[:, None]
    # side-surround decorrelator (for layouts with both side and rear)
    Hsd = _decorr_mag(f, gamma, 505)[:, None]

    AL = A * HAL; AR = A * HAR
    A_Lf = AL * HFr; A_Lr = AL * HRr
    A_Rf = AR * HFr; A_Rr = AR * HRr

    # ---- assemble output spectra --------------------------------------
    report(78, "Assembling output channels")
    w = _proc_band(f)[:, None]
    out = {lbl: np.zeros((nb, nf), dtype=complex) for lbl in labels}

    def put(key, spec):
        if key in pos:
            out[pos[key]] = out[pos[key]] + spec

    # fronts: bypass band keeps bass/extreme-highs; processed band = direct + front-floor ambient
    put("FL", (1.0 - w) * XL + w * (DL + amb_gain * A_Lf * HLo))
    put("FR", (1.0 - w) * XR + w * (DR + amb_gain * A_Rf * HLo))
    put("FC", w * DC)
    # surrounds (rear-floor ambient); choose rear keys if present, else side keys
    rg = amb_gain * rear_bias
    rear_keys = [("BL", "BR")] if ("BL" in pos and "BR" in pos) else \
                ([("SL", "SR")] if ("SL" in pos and "SR" in pos) else [])
    for kl, kr in rear_keys:
        put(kl, w * rg * A_Lr * HLo)
        put(kr, w * rg * A_Rr * HLo)
    # if BOTH side and rear exist (7.1.4): feed side surrounds a decorrelated copy
    if ("BL" in pos and "BR" in pos) and ("SL" in pos and "SR" in pos):
        put("SL", w * rg * (AL * Hsd) * HFr * HLo)
        put("SR", w * rg * (AR * Hsd) * HFr * HLo)
    # height layer (PBA high band)
    put("TFL", w * amb_gain * A_Lf * HHi)
    put("TFR", w * amb_gain * A_Rf * HHi)
    put("TBL", w * rg * A_Lr * HHi)
    put("TBR", w * rg * A_Rr * HHi)
    # voice-of-god + centre height
    put("TC", w * rg * 0.7 * (A * Htc) * HHi)
    put("TFC", w * amb_gain * 0.6 * (A * Htfc) * HHi)

    # ---- inverse STFT --------------------------------------------------
    report(88, "Inverse STFT")
    M = np.zeros((N, len(labels)), dtype=np.float64)
    for i, lbl in enumerate(labels):
        if lbl == pos.get("LFE"):
            continue
        _, y = signal.istft(out[lbl], sr, window=win, nperseg=NFFT,
                            noverlap=NFFT - HOP, boundary=True)
        M[:, i] = y[:N] if len(y) >= N else np.pad(y, (0, N - len(y)))

    # LFE = low-passed sum (not part of the upmix core)
    if "LFE" in pos and p.get("gen_lfe", True):
        sos = signal.butter(4, 120.0 / (sr / 2), btype="low", output="sos")
        lfe = signal.sosfilt(sos, 0.5 * (xL + xR))
        M[:, labels.index(pos["LFE"])] = lfe[:N]

    # peak safety
    pk = np.max(np.abs(M))
    if pk > 0.999:
        M *= 0.999 / pk
    report(100, "Done")
    return M.astype(np.float32), mask, labels


def _pba_split_pair(xL, xR, sr, pba_hz, pba_db):
    """True Perceptual Band Allocation on a floor pair: split the decorrelated
    ambient into a low band (stays on the floor) and a high band (moves up to
    the height).  Direct sources stay full-band on the floor.  Returns
    (floor_L, floor_R, height_L, height_R) with floor+height == input (the
    high-band ambient is MOVED, not copied -> downmix-complementary)."""
    NFFT, HOP = 2048, 512
    win = signal.windows.hann(NFFT, sym=False)
    N = len(xL)

    def stft(x):
        return signal.stft(x, sr, window=win, nperseg=NFFT, noverlap=NFFT - HOP,
                           boundary="zeros", padded=True)[2]
    XL = stft(xL.astype(np.float64)); XR = stft(xR.astype(np.float64))
    f = np.fft.rfftfreq(NFFT, 1.0 / sr)
    magL = np.abs(XL); magR = np.abs(XR)
    pw = np.sqrt(magL ** 2 + magR ** 2) + 1e-12
    gL = magL / pw; gR = magR / pw
    det = 1j * gL - gR
    act = (magL + magR) > 1e-7
    A = np.where(act, (gL * XR - gR * XL) / det, 0.0)     # mono ambient (= ambient in L)
    HHi = (1.0 - _shelf_mag(f, pba_hz, pba_db))[:, None]   # high band to move up
    hLs = (A) * HHi                                        # ambient in L, high band
    hRs = (1j * A) * HHi                                   # ambient in R (90deg), high band
    fLs = XL - hLs                                         # floor keeps direct + low ambient
    fRs = XR - hRs

    def istft(Z):
        return signal.istft(Z, sr, window=win, nperseg=NFFT,
                            noverlap=NFFT - HOP, boundary=True)[1]
    outs = [istft(z) for z in (fLs, fRs, hLs, hRs)]
    return [o[:N] if len(o) >= N else np.pad(o, (0, N - len(o))) for o in outs]


def kraft_upmix_mc(audio, sr, layout_name, strength=12, *,
                   input_order=None, p=None, progress=None):
    """Apply the STFT method to EXISTING 5.1 / 7.1 content: the discrete bed is
    kept intact and a 3D height layer is generated from the decorrelated
    ambient high-band of the floor pairs (Perceptual Band Allocation)."""
    if p is None:
        p = {}
    def report(pp, m=""):
        if progress:
            progress(int(pp), m)

    layout = LAYOUTS[layout_name]
    labels = [l for l, _ in layout]
    pos = {key: l for l, key in layout}
    mask = 0
    for _, key in layout:
        mask |= SPK[key]

    audio = np.atleast_2d(audio.T).T if audio.ndim == 1 else audio
    audio = audio.astype(np.float64)
    N, in_ch = audio.shape

    # ---- map input channels to roles -----------------------------------
    if in_ch >= 8:
        order = ORDERS_8.get(input_order, list(ORDERS_8.values())[0])
    else:
        order = ORDERS_6.get(input_order, list(ORDERS_6.values())[0])
    def ch(role):
        i = order.get(role)
        return audio[:, i] if i is not None and i < in_ch else None
    L, R, C, LFE = ch("L"), ch("R"), ch("C"), ch("LFE")
    RSL, RSR = ch("RSL"), ch("RSR")          # rear surround (present in 5.1/7.1)
    SSL, SSR = ch("SSL"), ch("SSR")          # side surround (7.1 only)
    if L is None or R is None:
        L = audio[:, 0]; R = audio[:, 1] if in_ch > 1 else audio[:, 0]

    report(10, "5.1/7.1 bed + true-PBA height split")
    out = {lbl: np.zeros(N) for lbl in labels}
    def put(key, sig):
        if key in pos and sig is not None:
            out[pos[key]] = out[pos[key]] + sig

    pba_db = float(p.get("pba_db", -8.0))
    base_hz = float(p.get("pba_hz", 3500.0))
    # strength tilts the PBA crossover: more strength -> lower crossover -> more
    # ambient energy lifted to the heights (the split stays complementary, so a
    # height->floor downmix still reconstructs the original bed).
    pba_hz = float(np.clip(base_hz * (1.5 - np.clip(strength, 0, 15) / 15.0),
                           800.0, 16000.0))
    hf = {}                                        # cache of moved high-band ambient

    def pba(xl, xr, kl, kr, fkl, fkr, tag):
        """Move the high-band ambient of floor pair -> height pair (kl,kr)."""
        fL, fR, hL, hR = _pba_split_pair(xl, xr, sr, pba_hz, pba_db)
        if kl in pos or kr in pos:                 # height speakers exist: split
            put(fkl, fL); put(fkr, fR)
            put(kl, hL); put(kr, hR)
            hf[tag] = (hL, hR)
        else:                                      # no height above: keep floor whole
            put(fkl, fL + hL); put(fkr, fR + hR)

    # fronts -> front heights
    report(35, "Front PBA split")
    pba(L, R, "TFL", "TFR", "FL", "FR", "front")
    # centre + LFE stay intact (never lift dialogue or sub overhead)
    if C is not None:
        put("FC", C)
    if LFE is not None:
        put("LFE", LFE)
    elif "LFE" in pos and p.get("gen_lfe", True):
        sos = signal.butter(4, 120.0 / (sr / 2), btype="low", output="sos")
        put("LFE", signal.sosfilt(sos, 0.5 * (L + R)))

    # surrounds -> rear heights
    report(65, "Surround PBA split")
    has_BL = "BL" in pos and "BR" in pos
    has_SL = "SL" in pos and "SR" in pos
    if has_BL and has_SL:                          # 7.1.4: rear surr -> rear heights
        pba(RSL if RSL is not None else L, RSR if RSR is not None else R,
            "TBL", "TBR", "BL", "BR", "rear")
        put("SL", SSL if SSL is not None else RSL)   # side surrounds stay intact
        put("SR", SSR if SSR is not None else RSR)
    else:                                          # single surround zone
        sl = RSL.copy() if RSL is not None else L.copy()
        sr_ = RSR.copy() if RSR is not None else R.copy()
        if SSL is not None:                        # fold side into rear (equal power)
            sl = (sl + SSL) / np.sqrt(2.0); sr_ = (sr_ + SSR) / np.sqrt(2.0)
        kl, kr = ("BL", "BR") if has_BL else ("SL", "SR")
        pba(sl, sr_, "TBL", "TBR", kl, kr, "rear")

    # voice-of-god / centre height: low-level decorrelated blends of the moved
    # high-band ambient (supplementary top fill; the 5.1.4 core split is exact)
    fr_h = hf.get("front"); rr_h = hf.get("rear")
    if "TC" in pos and fr_h and rr_h:
        put("TC", 0.5 * (fr_h[0] + rr_h[1]) * 0.6)
    if "TFC" in pos and fr_h:
        put("TFC", 0.5 * (fr_h[0] + fr_h[1]) * 0.5)

    report(90, "Assembling output")
    M = np.zeros((N, len(labels)))
    for i, lbl in enumerate(labels):
        M[:, i] = out[lbl]
    pk = np.max(np.abs(M))
    if pk > 0.999:
        M *= 0.999 / pk
    report(100, "Done")
    return M.astype(np.float32), mask, labels


# ==========================================================================
#  Dolby Surround / Pro Logic  (matrix detection + passive decode)
# ==========================================================================
# Dolby Surround encodes a mono surround into a stereo (Lt/Rt) pair OUT OF
# PHASE: the surround lives in the L-R difference, the fronts/centre in the
# L+R sum.  Normal stereo keeps its energy mostly in-phase (positive L/R
# correlation); matrix-encoded surround pushes energy into anti-phase
# (correlation toward/below zero).  We detect that, and decode with the
# classic passive matrix (after Jose "Dogway" Linares,
# https://github.com/Dogway): C = 0.3536*(L+R), S = 0.3536*(R-L) band-limited
# 100-7000 Hz, Dolby-delayed and 90deg phase-shifted, fed to both surrounds.

PROLOGIC_ORDER = "WAV / Microsoft (L R C LFE Ls Rs)"


def _delay(x, ms, sr):
    d = int(round(ms * 1e-3 * sr))
    if d <= 0:
        return x
    return np.concatenate([np.zeros(d), x])[:len(x)]


# ==========================================================================
#  Dynamics following
# ==========================================================================
def _smooth_abs(x, sr, tau_ms):
    """One-pole smoothed magnitude envelope (fast, symmetric)."""
    a = np.exp(-1.0 / (sr * tau_ms * 1e-3))
    return signal.lfilter([1 - a], [1.0, -a], np.abs(x).astype(np.float64))


def dynamics_follow_gain(dry, synth, sr, amount=0.8,
                         tau_ms=60.0, floor_db=-9.0, ceil_db=3.0):
    """Time-varying gain that makes the SYNTHESISED field track the macro-dynamics
    of the original (dry) mix: where the original dips, reverb/ambience is pulled
    down with it; where it swells, the field rises.  The gain is recentred to a
    synth-energy-weighted unit mean, so the field's overall loudness is preserved
    and only its contour over time is reshaped.  `amount` (0..1) blends from no
    effect to full following."""
    ed = _smooth_abs(dry, sr, tau_ms) + 1e-9
    es = _smooth_abs(synth, sr, tau_ms) + 1e-9
    n = min(len(ed), len(es))
    ed, es = ed[:n], es[:n]
    ratio = ed / es
    w = es ** 2                                    # weight by synth energy
    gm = np.exp(np.sum(w * np.log(ratio)) / (np.sum(w) + 1e-12))  # weighted geomean
    ratio = ratio / (gm + 1e-12)
    lo, hi = 10.0 ** (floor_db / 20.0), 10.0 ** (ceil_db / 20.0)
    ratio = np.clip(ratio, lo, hi)
    g = np.exp(float(np.clip(amount, 0.0, 1.0)) * np.log(ratio))
    a = np.exp(-1.0 / (sr * 20.0 * 1e-3))          # de-zipper
    g = signal.lfilter([1 - a], [1.0, -a], g)
    return g


# ==========================================================================
#  Overhead object / atmosphere analyzer  (steer to the height layer)
# ==========================================================================
def _helicopter_band(x, sr):
    """Detect periodic low-frequency amplitude modulation (a rotor's blade-pass,
    ~6-45 Hz) via envelope AUTOCORRELATION — a true rotor shows a strong periodic
    peak; broadband noise (rain/wind) does not.  Returns a per-sample 0..1
    presence gain and the low-mid band content to lift overhead."""
    bl, al = signal.butter(2, 80.0 / (sr / 2), btype="low")
    env = signal.lfilter(bl, al, np.abs(x))
    ds = max(1, int(round(sr / 1000.0)))           # ~1 kHz envelope for autocorr
    e = env[::ds]; fs_e = sr / ds
    blk = max(int(fs_e * 0.5), 64); hopb = blk // 2
    g_ds = np.zeros(len(e))
    han = np.hanning(blk)
    for s in range(0, max(1, len(e) - blk), hopb):
        seg = e[s:s + blk]
        m = seg.mean()
        depth = seg.std() / (m + 1e-9)             # modulation depth
        if seg.std() < 1e-6 or depth < 0.12:        # shallow ripple -> not a rotor
            continue
        sp = np.abs(np.fft.rfft((seg - m) * han))
        ff = np.fft.rfftfreq(blk, 1.0 / fs_e)
        band = (ff >= 6.0) & (ff <= 45.0)
        if not band.any() or sp[1:].size == 0:
            continue
        glob = 1 + int(np.argmax(sp[1:]))          # dominant modulation bin (no DC)
        if not band[glob]:                          # rotor band must DOMINATE
            continue
        prom = sp[glob] / (np.median(sp[1:]) + 1e-9)
        g_ds[s:s + blk] = (np.clip((prom - 6.0) / 12.0, 0.0, 1.0)
                           * np.clip((depth - 0.12) / 0.25, 0.0, 1.0))
    gain = np.interp(np.arange(len(x)), np.arange(len(e)) * ds, g_ds)
    a = np.exp(-1.0 / (sr * 120.0 * 1e-3))
    gain = signal.lfilter([1 - a], [1.0, -a], gain)
    bb, ab = signal.butter(2, [40.0 / (sr / 2), 1500.0 / (sr / 2)], btype="band")
    lm = signal.lfilter(bb, ab, x)
    return gain, lm * gain


def analyze_overhead(L, R, sr):
    """Heuristic detector that isolates content best placed overhead and returns
    the (hL, hR) signal to add to the height layer, plus mean diagnostic scores.

    Targets, all using only spectral/spatial cues (no trained model):
      * rain / wind / storm  -> diffuse (low inter-channel coherence) AND
        noise-like (high spectral flatness) energy, across the band
      * helicopters          -> periodic low-frequency rotor modulation
      * isolated objects      -> brief, narrow-band, transient peaks sitting in
        the diffuse field
    Tonal, centred, correlated content (dialogue, music) scores low and is left
    in place."""
    nper = 4096; hop = nper // 4; win = "hann"
    f, t, ZL = signal.stft(L, fs=sr, window=win, nperseg=nper, noverlap=nper - hop)
    _, _, ZR = signal.stft(R, fs=sr, window=win, nperseg=nper, noverlap=nper - hop)
    PL = np.abs(ZL) ** 2; PR = np.abs(ZR) ** 2; P = 0.5 * (PL + PR) + 1e-12
    # inter-channel coherence -> diffuse = 1 - coherence
    a = np.exp(-hop / (sr * 50e-3)); bsm, asm = [1 - a], [1.0, -a]
    Sll = signal.lfilter(bsm, asm, PL, axis=1)
    Srr = signal.lfilter(bsm, asm, PR, axis=1)
    Slr = signal.lfilter(bsm, asm, ZL * np.conj(ZR), axis=1)
    coh = np.clip(np.abs(Slr) / (np.sqrt(Sll * Srr) + 1e-9), 0, 1)
    diffuse = 1.0 - coh
    # per-frame spectral flatness (geo/arith mean), noise-like -> ~1
    flat = np.exp(np.mean(np.log(P), axis=0)) / (np.mean(P, axis=0) + 1e-12)
    flat = np.clip(flat, 0, 1)
    # --- object detection: brief, spectrally-compact, off-centre transients ---
    rootP = np.sqrt(P)
    flux = np.maximum(0.0, np.diff(rootP, axis=1, prepend=rootP[:, :1]))
    flux_fr = flux.sum(axis=0)
    # adaptive onset: subtract a slow baseline of the flux, so only RISES above the
    # recent level count — sustained noise/ambience (steady flux) no longer reads
    # as onsets, only genuine transient hits do.
    ab = 0.92
    base = signal.lfilter([1 - ab], [1.0, -ab], flux_fr)
    onset = np.maximum(0.0, flux_fr - 1.6 * base)
    onset = np.clip(onset / (np.percentile(onset, 95) + 1e-9), 0, 1)
    # spectral compactness (crest): an isolated object concentrates its energy in a
    # few bins; broadband wash (rain/wind) does not.
    crest = np.max(P, axis=0) / (np.mean(P, axis=0) + 1e-12)
    spars = np.clip((crest - 8.0) / 40.0, 0, 1)
    # narrow-band peakiness (a bin well above its frame's median)
    med = np.median(rootP, axis=0, keepdims=True)
    peaky = np.clip(rootP / (3.0 * med + 1e-9) - 1.0, 0, 1)
    # masks
    atmos = diffuse * (flat[None, :] ** 0.5)
    objects = peaky * diffuse * (onset * spars)[None, :]
    G = np.clip(0.9 * atmos + 0.9 * objects, 0, 1)

    def back(Z):
        _, x = signal.istft(Z * G, fs=sr, window=win, nperseg=nper, noverlap=nper - hop)
        return x
    hL, hR = back(ZL), back(ZR)
    n = min(len(L), len(R), len(hL), len(hR))
    hL, hR = hL[:n], hR[:n]
    hg, hb = _helicopter_band(0.5 * (L[:n] + R[:n]), sr)
    hL = hL + hb[:n]; hR = hR + hb[:n]
    # per-sample presence: the fraction of the input that is overhead-able now
    a = np.exp(-1.0 / (sr * 40.0 * 1e-3))
    ov = signal.lfilter([1 - a], [1.0, -a], np.abs(0.5 * (hL + hR)))
    ip = signal.lfilter([1 - a], [1.0, -a], np.abs(0.5 * (L[:n] + R[:n]))) + 1e-9
    pres = np.clip(ov / ip, 0.0, 1.0)

    # SLOW, SUSTAINED duck presence — drives the 3D-Immersive bed duck.  This is a
    # deliberately different, much slower signal than `pres`: it follows ONLY the
    # sustained DIFFUSE/AMBIENT energy (wind, storm, the recording's natural
    # reflections, ambient-music wash) plus the rotor band, and explicitly
    # EXCLUDES the transient object detector — so brief sounds never cause fast,
    # random ducking.  Energy-weighted diffuse fraction per frame, de-weighted on
    # transient frames, then smoothed with a ~1.5 s time constant so the duck only
    # swells and recedes slowly (imperceptible as pumping).
    diff_e = np.sum(diffuse * P, axis=0)
    tot_e = np.sum(P, axis=0) + 1e-12
    e_rms = np.sqrt(tot_e)
    floor = 0.05 * np.percentile(e_rms, 90) + 1e-12
    gate = np.clip(e_rms / floor - 1.0, 0.0, 1.0)          # ~0 in silence, 1 with signal
    dfrac = (diff_e / tot_e) * gate     # diffuse fraction of the sound actually present
    # causal, asymmetric slow smoothing (≈1.5 s attack, ≈3 s release) at frame rate
    # so the duck only swells in and recedes gently — never fast or pumping.
    a_atk = np.exp(-hop / (sr * 2500.0 * 1e-3))
    a_rel = np.exp(-hop / (sr * 4000.0 * 1e-3))
    duck_fr = np.empty_like(dfrac)
    g_ = 0.0
    for ii in range(len(dfrac)):
        v = dfrac[ii]
        a_ = a_atk if v > g_ else a_rel
        g_ = a_ * g_ + (1.0 - a_) * v
        duck_fr[ii] = g_
    fi = np.clip((t * sr).astype(int), 0, n - 1)
    duck_pres = np.interp(np.arange(n), fi, duck_fr)

    scores = dict(atmosphere=float(atmos.mean()),
                  objects=float(objects.mean()),
                  helicopter=float(np.mean(hg)))
    return hL, hR, pres, duck_pres, scores


def _bw(x, fc, btype, sr, order=4):
    sos = signal.butter(order, min(fc, 0.45 * sr) / (sr / 2.0), btype=btype, output="sos")
    return signal.sosfilt(sos, x)


def detect_prologic(audio, sr):
    """Return (is_encoded, info).  Keys: rho, oop, score, active_db."""
    if audio.ndim == 1 or audio.shape[1] < 2:
        return False, dict(rho=1.0, oop=0.0, score=0.0, active_db=-120.0)
    L = audio[:, 0].astype(np.float64)
    R = audio[:, 1].astype(np.float64)
    mid = 0.5 * (L + R)
    side = 0.5 * (L - R)
    me = float(np.mean(mid ** 2))
    se = float(np.mean(side ** 2))
    tot = me + se
    active_db = 10.0 * np.log10(tot + 1e-12)
    if tot < 1e-7:                      # essentially silent
        return False, dict(rho=1.0, oop=0.0, score=0.0, active_db=active_db)
    oop = se / tot                      # out-of-phase energy fraction (0..1)
    sd = np.std(L) * np.std(R)
    rho = float(np.mean((L - L.mean()) * (R - R.mean())) / (sd + 1e-12))
    # encoded when the channels are not positively correlated and a real
    # anti-phase (surround) component is present.
    encoded = (rho < 0.2) and (oop > 0.30)
    score = max(0.0, min(1.0, 0.5 * (1.0 - rho)))   # 0 = mono, 1 = fully anti-phase
    return bool(encoded), dict(rho=rho, oop=oop, score=score, active_db=active_db)


def prologic_decode(audio, sr):
    """Passive Dolby Surround / Pro Logic decode of an Lt/Rt stereo pair to a
    6-channel 5.1 array in WAV order (L R C LFE Ls Rs).  The surround carries ONLY
    the content that is on both channels but phase-inverted (the genuine matrix
    surround); in-phase and hard-panned material stays in the fronts/centre.
    Surrounds are mono (one matrixed S feeds both)."""
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    L = audio[:, 0].astype(np.float64)
    R = audio[:, 1].astype(np.float64)
    g = 0.353553                                   # -9 dB matrix coefficient

    # The surround = ONLY the anti-phase common component (present in both
    # channels, inverted).  Keep deep bass out of the surround feed.
    A = antiphase_common(L, R, sr)
    Sa = _bw(A, 80.0, "high", sr)
    S = g * Sa

    # In-phase content -> centre; deep in-phase sum -> LFE.
    C = g * (L + R)
    C = _bw(C, 70.0, "high", sr)
    C = _bw(C, 20000.0, "low", sr)
    LFE = _bw(g * (L + R), 100.0, "low", sr)

    # Fronts keep everything EXCEPT the anti-phase common part (so the out-of-phase
    # surround material does not also play from the fronts): L carries +A, R -A.
    Lo = 0.5 * (L - A)
    Ro = 0.5 * (R + A)
    out = np.stack([Lo, Ro, C, LFE, S, S], axis=1)   # L R C LFE Ls Rs
    return out.astype(np.float32)



def upmix(audio, sr, layout_name, preset_name, strength=12, *,
          center_gen=True, gen_lfe=True, input_order=None,
          decorrelate=False, prologic=False, surr_dry_lift=True,
          height_phase_diff=True, dynamics_follow=True, dyn_amount=0.4,
          steer_to_heights=False, steer_amount=0.6,
          immersive_3d=False, immersive_duck_db=11.0, no_reverb=False,
          overrides=None, progress=None):
    def report(pp, m=""):
        if progress:
            progress(int(pp), m)

    if audio.ndim == 1:
        audio = audio[:, None]

    # Dolby Surround / Pro Logic: decode the stereo matrix to a 5.1 bed first,
    # then upmix that.  The recovered surround is a real mono signal, so we do
    # NOT decorrelate in this mode.
    if prologic and audio.shape[1] == 2:
        report(4, "Dolby Pro Logic decode")
        audio = prologic_decode(audio, sr)
        input_order = PROLOGIC_ORDER
        decorrelate = False

    p = dict(PRESETS[preset_name])
    if overrides:
        p.update({k: v for k, v in overrides.items() if v is not None})

    # Decorrelation widens the field by pulling the steered pair correlations
    # toward independence.  Each layer can decorrelate by a different amount:
    #   * decorr        - synthesised surrounds (stereo input)
    #   * decorr_bed    - pass-through surround bed (5.1/7.1 or Pro Logic input)
    #   * decorr_height - height layers (falls back to decorr if unset)
    # 1.0 = fully decorrelated; the explicit `decorrelate` flag forces all to
    # full.  dec/dec_bed/dec_h are the surviving-correlation multipliers
    # (1.0 = as measured, 0.0 = independent).
    decorr_frac = 1.0 if decorrelate else float(p.get("decorr", 0.0))
    dec = 1.0 - decorr_frac
    bed_frac = 1.0 if decorrelate else float(p.get("decorr_bed", 0.0))
    dec_bed = 1.0 - bed_frac
    _dh = p.get("decorr_height", None)
    h_frac = (1.0 if decorrelate
              else (float(_dh) if _dh is not None else float(p.get("decorr", 0.0))))
    dec_h = 1.0 - h_frac

    layout = LAYOUTS[layout_name]
    labels = [l for l, _ in layout]
    pos = {key: l for l, key in layout}
    mask = 0
    for _, key in layout:
        mask |= SPK[key]

    audio = audio.astype(np.float64)
    N, in_ch = audio.shape

    if in_ch == 1:
        omap = {"L": 0, "R": 0}
    elif in_ch == 2:
        omap = {"L": 0, "R": 1}
    elif in_ch <= 6:
        omap = ORDERS_6.get(input_order, list(ORDERS_6.values())[0])
    elif in_ch <= 8:
        omap = ORDERS_8.get(input_order, list(ORDERS_8.values())[0])
    elif in_ch <= 12:
        omap = ORDERS_12.get(input_order, list(ORDERS_12.values())[0])
    else:
        omap = ORDERS_16.get(input_order, list(ORDERS_16.values())[0])

    def ch(role):
        return audio[:, omap[role]] if role in omap else None

    L, R = ch("L"), ch("R")
    in_C, in_LFE = ch("C"), ch("LFE")
    in_zones = {}
    if ch("SSL") is not None:
        in_zones["side"] = (ch("SSL"), ch("SSR"))
    if ch("RSL") is not None:
        in_zones["rear"] = (ch("RSL"), ch("RSR"))
    # discrete INPUT height / wide channels (12-16 ch inputs), keyed by output
    # speaker key so they can be passed through to matching output positions.
    in_heights = {}
    for role, key in [("HFL", "TFL"), ("HFR", "TFR"), ("HBL", "TBL"),
                      ("HBR", "TBR"), ("HML", "TSL"), ("HMR", "TSR")]:
        s = ch(role)
        if s is not None:
            in_heights[key] = s
    in_wide = {}
    for role, key in [("WL", "FLW"), ("WR", "FRW")]:
        s = ch(role)
        if s is not None:
            in_wide[key] = s

    report(8, "Primary/ambient decomposition")
    pL, pR, aL, aR = primary_ambient_decompose(L, R, sr, diffuse_exp=p["diffuse"])
    n = len(pL)
    L, R = L[:n], R[:n]
    in_heights = {k: v[:n] for k, v in in_heights.items()}
    in_wide = {k: v[:n] for k, v in in_wide.items()}

    # Optional content analyzer: isolate rain / wind / storm / helicopter / lone
    # transient objects to lift into the height layer (added later, post-synth).
    steer_hL = steer_hR = steer_pres = steer_duck = None
    if steer_to_heights or immersive_3d:
        report(12, "Analyzing for overhead objects")
        steer_hL, steer_hR, steer_pres, steer_duck, _steer_scores = analyze_overhead(L, R, sr)
        steer_hL = steer_hL[:n]; steer_hR = steer_hR[:n]
        steer_pres = steer_pres[:n]; steer_duck = steer_duck[:n]
    # `w` is the immersion MORPH amount in [0,1] (strength 0..16), not a level:
    #   w = 0  -> pure decorrelated dry signal (no reverb/reflections); the
    #            immersive channels are still present, just dry.
    #   w = 1  -> full reverb + reflections.
    # The per-layer LEVEL is set by scale_pair (mostly strength-independent;
    # presets may trim the wet end via wet_trim_db).  Strength crossfades
    # dry<->wet and scales the LFE-lift amount.
    w = float(np.clip(strength, 0, 16)) / 16.0
    if no_reverb:
        w = 0.0          # PURE UPMIX: no reverb, no reflections, no LFE bloom —
        #                  just the dry spatial redistribution (the strength-0
        #                  field) regardless of the strength slider.
    # Phase-difference subtraction is FULL whenever enabled, INDEPENDENT of
    # strength (so even strength 0 carries the L-Ls / Ls-Lss subtraction).
    pd = 1.0 if height_phase_diff else 0.0
    out = {}

    def K(seed):
        return reverb_kernel(sr, p["length_ms"], p["density"], p["decay_ms"],
                             p["er_ms"], seed, p["predelay_ms"],
                             p["hf_decay_ratio"], p["xover_hz"],
                             p["air_hz"], p["clarity"], p.get("xcurve", False),
                             p.get("xover_order", 2), p.get("rt60_curve", None))
    k_sh = K(11)
    k_sh_h = K(12)        # shared component for front-height pair correlation
    k_sh_rh = K(13)       # shared component for rear-height pair correlation
    k = {s: K(s) for s in (101, 202, 303, 404, 505, 606, 707, 808, 909, 1010,
                            1212, 1313, 1414, 1515, 1616)}
    # all-pass decorrelators: flat magnitude, scrambled phase.  Used to widen a
    # pass-through bed AND to build the "dry decorrelated" end of the morph.
    # Kept SHORT (~6 ms) so the strength-0 dry field fuses with the direct sound
    # (under the ~5-10 ms echo-fusion threshold) instead of reading as a
    # reflection; still fully decorrelates the mids/highs (bass stays coherent).
    #   9001/9002 bed widening · 9003/9004 front heights · 9005/9006 rear heights
    #   9007/9008 stereo surround synth · 9011/9012 generated rear-from-surround
    kd = {s: decorr_kernel(sr, 6.0, s) for s in range(9001, 9023)}
    # Dry decorrelation uses TRUE all-pass chains (flat magnitude), so the dry
    # height/surround field stays tonally uncoloured — only its phase is
    # scrambled to widen the image.  (The FIR kernels above remain for the
    # reverb/diffuse paths only.)
    apd = {s: allpass_decorr_params(sr, s) for s in range(9001, 9023)}

    def dap(x, s):
        return schroeder_allpass(x, *apd[s])

    def dry_wet(wetL, wetR, dryL, dryR, dryfrac):
        # normalise dry and wet to equal RMS, then amplitude-blend dryfrac/(1-dryfrac)
        dn = np.sqrt(0.5 * (np.mean(dryL ** 2) + np.mean(dryR ** 2))) + 1e-12
        wn = np.sqrt(0.5 * (np.mean(wetL ** 2) + np.mean(wetR ** 2))) + 1e-12
        return (dryfrac * dryL / dn + (1 - dryfrac) * wetL / wn,
                dryfrac * dryR / dn + (1 - dryfrac) * wetR / wn)

    def morph(dryL, dryR, wetL, wetR):
        """Crossfade the decorrelated-dry pair (strength 0) into the reverb-wet
        pair (strength 16) using the global immersion amount `w`.  Both ends are
        RMS-normalised first so the blend keeps a steady level across strength;
        the final per-layer level is then set by scale_pair."""
        return dry_wet(wetL, wetR, dryL, dryR, 1.0 - w)

    def morph1(dry, wet):
        """Mono version of `morph` (single channel)."""
        dn = np.sqrt(np.mean(dry ** 2)) + 1e-12
        wn = np.sqrt(np.mean(wet ** 2)) + 1e-12
        return (1.0 - w) * dry / dn + w * wet / wn

    def decorr2(xl, xr, sa, sb):
        """A fully decorrelated stereo pair from a source (two all-pass chains,
        flat magnitude => uncoloured)."""
        return dap(xl, sa), dap(xr, sb)

    def slight_decorr2(xl, xr, sa, sb, amount=0.18):
        """A SLIGHTLY decorrelated pair: mostly the dry source with a small
        decorrelated blend (used for the extra 9.1.6 channels so each new pair
        widens a little without losing its direct content)."""
        a_, b_ = np.sqrt(1.0 - amount), np.sqrt(amount)
        return (a_ * xl + b_ * dap(xl, sa),
                a_ * xr + b_ * dap(xr, sb))

    # Reference front level + a helper that scales a generated stereo pair so its
    # per-channel RMS sits at `ratio * front * wet_scale` (strength-INDEPENDENT,
    # so the dry decorrelated field at strength 0 is at the same level as the
    # full reverb at strength 16 — strength morphs character, not loudness).
    # wet_scale lowers a preset's whole synthesized field (e.g. Movie = 0.85).
    frontref = 0.5 * (np.sqrt(np.mean(L ** 2)) + np.sqrt(np.mean(R ** 2))) + 1e-12
    ws = float(p.get("wet_scale", 1.0))
    wt_db = float(p.get("wet_trim_db", 0.0))      # wet-end level trim, scaled by w
    wt = 10.0 ** (wt_db * w / 20.0)               # 1.0 at strength 0, full trim at 16

    def scale_pair(xl, xr, ratio):
        cur = np.sqrt(0.5 * (np.mean(xl ** 2) + np.mean(xr ** 2))) + 1e-12
        g = (ratio * frontref * ws * wt) / cur
        return xl * g, xr * g

    def scale_mono(x, ratio):
        cur = np.sqrt(np.mean(x ** 2)) + 1e-12
        return x * ((ratio * frontref * ws * wt) / cur)

    def unit(x):
        return x / (np.sqrt(np.mean(x ** 2)) + 1e-12)

    # ---- front bed (intact) + reduced reverb send ----------------------
    # In a cinema the screen channels excite the room too, so L/R/C get a
    # REDUCED reverb send (front_reverb_db below their dry level) on top of the
    # intact direct signal.  Scaled by strength, so strength 0 stays pure dry.
    out["L"], out["R"] = L.copy(), R.copy()
    fr_db = p.get("front_reverb_db", None)
    if fr_db is None and p.get("front_reverb", 0.0) > 0:        # legacy fallback
        fr_db = 20.0 * np.log10(max(p["front_reverb"], 1e-6))
    # cinema geometry / treble: the screen speakers are SPK_DIST["front"] away,
    # so their room reverb arrives later, with the same gentle treble rolloff.
    _geo = bool(p.get("geo_predelay", False))
    _revhf = float(p.get("reverb_hf_db", 0.0))
    _fpre = ((SPK_DIST["front"] - min(SPK_DIST.values())) / SPEED_OF_SOUND * 1000.0
             if _geo else 0.0)

    def _front_send(x):
        if _revhf:
            x = rbj_high_shelf(x, sr, 5500, _revhf)
        if _fpre:
            x = _delay(x, _fpre, sr)
        return x
    if fr_db is not None and w > 0:
        fr_lvl = (10.0 ** (fr_db / 20.0)) * w * frontref
        rl, rr = _corr_pair(L, R, k_sh, k[101], k[202], p["front_corr"])
        out["L"] = out["L"] + _front_send(unit(rl)) * fr_lvl
        out["R"] = out["R"] + _front_send(unit(rr)) * fr_lvl

    # ---- centre + optional dialogue removal from the fronts ------------
    if in_C is not None:
        out["C"] = in_C[:n].copy()
    elif center_gen and "FC" in pos:
        cc = 0.5 * (pL + pR)                         # coherent mono centre
        bc, ac = signal.butter(2, 90 / (sr / 2), btype="high")
        # zero-phase high-pass so the band subtracted from the fronts lines up
        # phase-exactly with the fronts and cancels cleanly (a causal filter's
        # phase shift would leave the dialogue only partly removed).
        c_hp = signal.filtfilt(bc, ac, cc)           # dialogue band of the centre
        if p.get("front_center_removal", False) and decorr_frac > 0:
            # Pull the coherent phantom centre (dialogue) OUT of L/R and route it
            # to the centre channel, so no dialogue plays from the fronts.  Only
            # the high-passed centre is subtracted, so bass stays in the fronts
            # and the high-frequency centre energy is conserved (it moves to C).
            out["C"] = c_hp
            out["L"] = out["L"] - c_hp
            out["R"] = out["R"] - c_hp
        else:
            out["C"] = c_hp * p["center_amt"]
    else:
        out["C"] = np.zeros(n)

    # reduced reverb send on the centre (same room as the fronts), kept ~3 dB
    # lower than the L/R send to protect dialogue intelligibility.
    if fr_db is not None and w > 0:
        c = out.get("C")
        if isinstance(c, np.ndarray) and np.any(c):
            cref = np.sqrt(np.mean(c ** 2)) + 1e-12
            out["C"] = c + _front_send(unit(conv_same(c, k[1616]))) * (10.0 ** (fr_db / 20.0)) * w * cref * 0.7

    # ---- LFE -----------------------------------------------------------
    if in_LFE is not None:
        out["LFE"] = in_LFE[:n].copy()
    elif gen_lfe and "LFE" in pos:
        bl, al = signal.butter(4, 120 / (sr / 2), btype="low")
        out["LFE"] = signal.lfilter(bl, al, 0.5 * (L + R)) * 0.7
    else:
        out["LFE"] = np.zeros(n)
    # optional LFE bass lift, scaled by strength (full at max strength)
    lfe_db = float(p.get("lfe_gain_db", 0.0))
    if lfe_db and "LFE" in pos:
        out["LFE"] = out["LFE"] * (10.0 ** (lfe_db * w / 20.0))

    report(35, "Routing surround zones")
    out_zones = {}
    if "BL" in pos:
        out_zones["rear"] = ("BL", "BR")
    if "SL" in pos:
        out_zones["side"] = ("SL", "SR")

    primary = "rear" if "rear" in out_zones else ("side" if "side" in out_zones else None)

    # Assign input surround zones to output zones with correct gains:
    #   * equal counts (1->1, 2->2)  : direct unity map (same-name preferred)
    #   * fewer outputs than inputs  : fold all inputs into the primary at equal
    #                                  power (1/sqrt(N))
    #   * more outputs than inputs   : fill matched zones at unity, synthesise rest
    feed = {z: [] for z in out_zones}
    in_keys, out_keys = list(in_zones.keys()), list(out_zones.keys())
    if in_keys and out_keys:
        if len(in_keys) <= len(out_keys):
            used = set()
            two_out = ("side" in out_zones and "rear" in out_zones)
            for iz in in_keys:
                if len(in_keys) == 1 and two_out:
                    # CHANGE 2: a lone 5.1 surround expanding to a layout with
                    # BOTH side and rear zones (7.1.4) keeps its physical SIDE
                    # position; the REAR/back zone is then GENERATED from the
                    # decorrelated input surrounds (see the synthesis below).
                    tz = "side"
                else:
                    tz = iz if (iz in out_zones and iz not in used) else \
                         next((o for o in out_keys if o not in used), out_keys[0])
                used.add(tz)
                feed[tz].append((in_zones[iz][0], in_zones[iz][1], 1.0))
        else:
            g = 1.0 / np.sqrt(len(in_keys))
            for iz in in_keys:
                feed[primary].append((in_zones[iz][0], in_zones[iz][1], g))

    # ---- proximity reverb model -----------------------------------------
    # Every GENERATED speaker (surrounds + heights) carries its own DRY direct
    # signal PLUS room reverb sourced from the BED speakers by proximity: ~60%
    # from the nearest speaker below/adjacent, the remaining ~40% shared among
    # the rest weighted by inverse distance (farther = less).  A speaker is never
    # fed the reverb of its own signal.  The reverb is added scaled by strength,
    # so strength 0 is pure dry and strength 16 is dry + full reverb.
    shf = float(p.get("surr_hf_db", 0.0))
    prox_near = float(p.get("prox_near", 0.6))
    prox_power = float(p.get("prox_power", 1.5))
    if not surr_dry_lift:            # only the single nearest speaker (no 40% spread)
        prox_near = 1.0

    # Bed sources for the REVERB — UNIFIED across all input types so the room
    # character is identical whether the source is mono, stereo, 5.1 or 7.1.  It is
    # always just the front pair (+ centre): the synthesised reverb is sourced and
    # proximity-weighted the same way regardless of how many channels the input
    # had.  The discrete input surrounds still play in the DRY/direct path; they
    # simply no longer add their own (channel-count-dependent) reverb excitation,
    # so a stereo upmix and a 5.1/7.1 upmix get the same reverb.
    src_lbl, src_sig = ["FL", "FR"], [L, R]
    if in_C is not None:
        src_lbl.append("FC"); src_sig.append(in_C[:n])
    if prologic:
        # Pro Logic: the surround/height REVERB must also come only from the
        # anti-phase matrix surround, not the (in-phase/panned) fronts — otherwise
        # front content leaks up into the surrounds/heights via the room.
        _bz = in_zones.get("rear", in_zones.get("side"))
        if _bz is not None:
            _a = 0.5 * (_bz[0][:n] + _bz[1][:n])
            src_lbl, src_sig = ["FL", "FR"], [_a, _a]

    PROX_SEED = {"FL": 101, "FR": 202, "FC": 1616, "BL": 303, "BR": 404,
                 "SL": 505, "SR": 606, "TFL": 707, "TFR": 808, "TFC": 1313,
                 "TBL": 909, "TBR": 1010, "TC": 1212,
                 "FLW": 1717, "FRW": 1818, "TSL": 1919, "TSR": 2020}

    # PURE-AMBIENCE upmix: in no-reverb mode with NO discrete surround input
    # (mono/stereo), the surround + height field is built from the recording's
    # OWN extracted ambience (the diffuse aL/aR from the primary/ambient split)
    # instead of synthetic phase-decorrelated copies — so there are no synthetic
    # reflections at all, just the real diffuse content relocated overhead/around.
    pure_amb = bool(no_reverb) and not in_zones
    ambL, ambR = aL[:n], aR[:n]

    # In pure-ambience (mono/stereo, no reverb) the surround/height layer carries
    # the recording's OWN diffuse energy.  That ambience must sit at its NATURAL
    # level — a fraction below the front — not be normalised up to the front level
    # the way synthetic decorrelation is (doing so over-amplifies it and makes the
    # separated content loud and harsh).  So in pure-ambience mode these use a
    # direct gain instead of the RMS-normalising scale_pair, and the height
    # levelling is skipped.
    AMB_MAKEUP = 1.365   # stereo/mono pure-ambience makeup (5% stronger than 1.30)
    # The extracted ambience is spread across every surround + height speaker, so
    # the SUMMED diffuse energy would otherwise grow with the channel count and
    # start to drown the front (e.g. 9.1.6 put ~12 diffuse channels nearly level
    # with the front pair).  Normalise by the diffuse-channel count (power) so the
    # total ambient field sits at a consistent, front-dominant level whatever the
    # layout, while still tracking how much ambience the recording actually has.
    _ndiff = max(2, sum(1 for _k in pos if _k not in ("FL", "FR", "FC", "LFE")))
    amb_norm = float(np.sqrt(2.0 / _ndiff))

    def place(xl, xr, ratio):
        if pure_amb:
            g = ratio * AMB_MAKEUP * amb_norm
            return xl * g, xr * g
        return scale_pair(xl, xr, ratio)

    def place_mono(x, ratio):
        if pure_amb:
            return x * ratio * AMB_MAKEUP * amb_norm
        return scale_mono(x, ratio)

    def _rot(xl, xr, deg):
        """Static (delay-free, reflection-free) rotation in the L/R plane — used to
        differentiate a GENERATED rear zone from the side it is derived from, so
        they are not identical, without adding any synthetic reflection."""
        th = np.deg2rad(deg); c, s = np.cos(th), np.sin(th)
        return c * xl - s * xr, s * xl + c * xr

    # In PURE mode with discrete surround input, the height field should carry the
    # DIRECT (primary) content, not the source's baked-in reverb.  Pre-extract the
    # primary of the front and of the surround so the heights can be built dry.
    spri_L = spri_R = None
    if no_reverb and in_zones:
        _sz = in_zones.get("rear", in_zones.get("side"))
        # stronger diffuse rejection (diffuse_exp>1) so the surround's DIRECT part
        # used for the overhead-rear heights carries as little reverb as possible
        _spL, _spR, _saL, _saR = primary_ambient_decompose(_sz[0][:n], _sz[1][:n], sr,
                                                            diffuse_exp=2.2)
        spri_L, spri_R = _spL[:n], _spR[:n]

    def kk(seed):
        return k[seed] if seed in k else K(seed)

    # THX / cinema room controls (neutral for the room presets):
    rev_hf = float(p.get("reverb_hf_db", 0.0))         # slightly lowered treble
    fr_absorb = float(p.get("front_rear_absorb", 1.0)) # rear-wall absorbs fronts
    side_abs = float(p.get("side_absorb_db", 0.0))     # sidewall absorption
    geo = bool(p.get("geo_predelay", False))           # distance -> pre-delay
    _mind = min(SPK_DIST.values())

    def prox_reverb(out_label):
        """Distance-weighted reverb for one output speaker: the proximity-mixed
        bed sources convolved with this speaker's own room kernel (so each output
        is decorrelated from the others).  The reverb is renormalised to its
        SOURCE level, which keeps a left/right pair perfectly balanced for centred
        content (removing per-kernel energy variance) while still following the
        source pan, then shaped by the room (THX absorption / treble / geometry)."""
        wts = proximity_weights(out_label, src_lbl, prox_near, prox_power)
        zone = SPK_ZONE.get(out_label)
        if SPK_POS.get(out_label, (0.0, 0.0))[1] > 0:
            # Height speaker: keep the CENTRE channel's reverb out of the overhead
            # layer (its dialogue/score reverb was bleeding up into the heights).
            wts = [0.0 if src_lbl[i] == "FC" else wi for i, wi in enumerate(wts)]
        if fr_absorb < 1.0 and zone == "rear":
            # THX rear wall: the screen channels' energy is absorbed, not
            # reflected back into the seating, so down-weight the front sources.
            wts = [wi * (fr_absorb if src_lbl[i] in ("FL", "FR", "FC") else 1.0)
                   for i, wi in enumerate(wts)]
        mixed = np.zeros(n)
        for wi, sig in zip(wts, src_sig):
            if wi:
                mixed = mixed + wi * sig[:n]
        rv = conv_same(mixed, kk(PROX_SEED.get(out_label, 707)))
        mrms = np.sqrt(np.mean(mixed ** 2)) + 1e-12
        rrms = np.sqrt(np.mean(rv ** 2)) + 1e-12
        rv = rv * (mrms / rrms)                        # carry the source level
        if rev_hf:
            rv = rbj_high_shelf(rv, sr, 5500, rev_hf)  # gentle treble rolloff
        if side_abs and zone == "side":
            rv = rv * (10.0 ** (side_abs / 20.0))      # absorptive sidewalls
        if geo and zone is not None:
            rv = _delay(rv, (SPK_DIST[zone] - _mind) / SPEED_OF_SOUND * 1000.0, sr)
        return rv

    def surr_reverb(out_label):
        rv = prox_reverb(out_label)
        if shf:
            rv = rbj_high_shelf(rv, sr, 5000, shf)
        return rv

    for zname, (kl, kr) in out_zones.items():
        rvL, rvR = scale_pair(surr_reverb(kl), surr_reverb(kr), p["surr_gain"])
        if feed[zname]:
            # Discrete bed surround(s): the direct content passes through
            # COMPLETELY UNTOUCHED (no widening, no decorrelation — bit-exact at
            # strength 0); only the proximity reverb of the OTHER speakers is
            # added on top, scaled by strength.
            fl = np.zeros(n); fr = np.zeros(n)
            for zl, zr, g in feed[zname]:
                fl = fl + zl[:n] * g
                fr = fr + zr[:n] * g
            out[pos[kl]] = fl + w * rvL
            out[pos[kr]] = fr + w * rvR
        else:
            # No discrete bed -> the DRY DIRECT is a decorrelated presence (the
            # front for stereo, or the input surrounds for a generated 5.1->7.1.4
            # rear zone); the proximity reverb is sourced from the bed speakers.
            if in_zones:
                bz = in_zones.get("rear", in_zones.get("side"))
                gsL, gsR = bz[0][:n], bz[1][:n]
                dryl, dryr = (gsL, gsR) if no_reverb else decorr2(gsL, gsR, 9011, 9012)
            elif pure_amb:
                dryl, dryr = ambL, ambR          # recording's own ambience
            else:
                dryl, dryr = decorr2(L, R, 9007, 9008)
            if zname == "rear":
                # A GENERATED rear zone shares its source with the side it is
                # derived from; differentiate them so they are NOT identical.  In
                # pure mode use a static (reflection-free) rotation; otherwise an
                # extra decorrelation pass.
                if no_reverb:
                    dryl, dryr = _rot(dryl, dryr, 60.0)
                else:
                    dryl, dryr = decorr2(dryl, dryr, 9021, 9022)
            dl, dr = place(dryl, dryr, p["surr_gain"])
            out[pos[kl]] = dl + w * rvL
            out[pos[kr]] = dr + w * rvR

    report(60, "Synthesising height field")
    tilt = p["tilt_db"]
    hs = float(p.get("height_scale", 1.0))      # extra level trim for heights only
    rear_ratio = (p["rear_height_gain"] if p["rear_height_gain"] is not None
                  else p["height_gain"])
    hc = p["height_corr"] * dec_h
    rhc = (p["rear_height_corr"] if p["rear_height_corr"] is not None else p["height_corr"]) * dec_h

    def steer(xl, xr, sh, corr):
        a_, b_ = np.sqrt(corr), np.sqrt(1 - corr)
        return a_ * sh + b_ * xl, a_ * sh + b_ * xr

    # Track which height positions are DISCRETE PASSTHROUGH (from a 7.1.4/9.1.6
    # input) — those stay untouched (bed-like) and are excluded from the height
    # levelling, so an existing immersive mix is preserved.
    passthru_keys = set()

    def _hshelf(rv):
        return rbj_high_shelf(rv, sr, 6500, -tilt)

    # front heights
    if "TFL" in pos and "TFR" in pos:
        if "TFL" in in_heights and "TFR" in in_heights:
            rvL, rvR = scale_pair(prox_reverb("TFL"), prox_reverb("TFR"),
                                  p["height_gain"] * hs)
            out[pos["TFL"]] = in_heights["TFL"] + w * _hshelf(rvL)
            out[pos["TFR"]] = in_heights["TFR"] + w * _hshelf(rvR)
            passthru_keys |= {"TFL", "TFR"}
        else:
            if prologic and in_zones:
                # Pro Logic: the heights carry ONLY the matrixed anti-phase
                # surround (the same out-of-phase content as the surrounds), not
                # the in-phase / hard-panned front material.
                bz = in_zones.get("rear", in_zones.get("side"))
                a_src = 0.5 * (bz[0][:n] + bz[1][:n])
                dryl, dryr = (a_src, a_src) if no_reverb else decorr2(a_src, a_src, 9003, 9004)
            else:
                # FRONT-minus-SURROUND phase subtraction (time domain) — the height
                # principle.  The SAME dry source in normal and PURE mode; pure
                # simply omits the reverb send below (w = 0), so the heights stay
                # reflection-free while still built on the phase-subtraction
                # principle.  (Time-domain subtraction, not an STFT primary/ambient
                # extraction, so there is no spectral smearing that reads as room
                # reflections.)  Mono/stereo, no surround zone: the decorrelated bed.
                if pd > 0 and in_zones:
                    sub = in_zones.get("rear", in_zones.get("side"))   # Ls/Rs
                    hsrcL, hsrcR = L - pd * sub[0][:n], R - pd * sub[1][:n]
                else:
                    hsrcL, hsrcR = L, R
                dryl, dryr = decorr2(hsrcL, hsrcR, 9003, 9004)
            dl, dr = place(dryl, dryr, p["height_gain"] * hs)
            rvL, rvR = scale_pair(prox_reverb("TFL"), prox_reverb("TFR"),
                                  p["height_gain"] * hs)
            # ceiling-darkening shelf is a property of the ROOM reflections, so it
            # is applied to the reverb ONLY — the dry direct stays unfiltered, and
            # at strength 0 the height is a clean dry decorrelated signal.
            out[pos["TFL"]] = dl + w * _hshelf(rvL)
            out[pos["TFR"]] = dr + w * _hshelf(rvR)

    # rear/surround heights
    if "TBL" in pos and "TBR" in pos:
        if "TBL" in in_heights and "TBR" in in_heights:
            rvL, rvR = scale_pair(prox_reverb("TBL"), prox_reverb("TBR"),
                                  rear_ratio * hs)
            out[pos["TBL"]] = in_heights["TBL"] + w * _hshelf(rvL)
            out[pos["TBR"]] = in_heights["TBR"] + w * _hshelf(rvR)
            passthru_keys |= {"TBL", "TBR"}
        else:
            # Surround(-minus-side) phase subtraction, time domain — the same dry
            # source in normal and PURE mode (pure omits the reverb send below).
            # In Pro Logic the surround zone is the anti-phase signal, so this stays
            # anti-phase too.  No STFT primary/ambient extraction => no smearing.
            if in_zones:
                bz = in_zones.get("rear", in_zones.get("side"))
                asL, asR = bz[0][:n].copy(), bz[1][:n].copy()
                if pd > 0 and "rear" in in_zones and "side" in in_zones:
                    sz = in_zones["side"]
                    asL = asL - pd * sz[0][:n]
                    asR = asR - pd * sz[1][:n]
            else:
                asL, asR = L, R
            dryl, dryr = decorr2(asL, asR, 9005, 9006)
            dl, dr = place(dryl, dryr, rear_ratio * hs)
            rvL, rvR = scale_pair(prox_reverb("TBL"), prox_reverb("TBR"),
                                  rear_ratio * hs)
            out[pos["TBL"]] = dl + w * _hshelf(rvL)
            out[pos["TBR"]] = dr + w * _hshelf(rvR)

    # top-middle / top-side pair (9.1.6).  Passthrough if the input carries it
    # (16 ch), otherwise GENERATED from the front & back heights with the phase
    # subtraction + a SLIGHT inter-pair decorrelation (per the additional-channel
    # rule), placed between the two height rows.
    if "TSL" in pos and "TSR" in pos:
        if "TSL" in in_heights and "TSR" in in_heights:
            rvL, rvR = scale_pair(prox_reverb("TSL"), prox_reverb("TSR"),
                                  p["height_gain"] * hs)
            out[pos["TSL"]] = in_heights["TSL"] + w * _hshelf(rvL)
            out[pos["TSR"]] = in_heights["TSR"] + w * _hshelf(rvR)
            passthru_keys |= {"TSL", "TSR"}
        else:
            if in_heights:
                fL = in_heights.get("TFL", L); bL = in_heights.get("TBL", L)
                fR = in_heights.get("TFR", R); bR = in_heights.get("TBR", R)
            else:
                # derive from the same dry, time-domain phase-diff sources the
                # front/rear heights use (no STFT primary extraction => no smearing)
                if pd > 0 and in_zones:
                    sub = in_zones.get("rear", in_zones.get("side"))
                    fL, fR = L - pd * sub[0][:n], R - pd * sub[1][:n]
                else:
                    fL, fR = L, R
                if in_zones:
                    bz = in_zones.get("rear", in_zones.get("side"))
                    bL, bR = bz[0][:n], bz[1][:n]
                else:
                    bL, bR = L, R
            mL = 0.5 * (fL + bL); mR = 0.5 * (fR + bR)   # between the two rows
            mcoh = pd * 0.5 * (mL + mR)
            mL = mL - mcoh; mR = mR - mcoh               # phase-sub coherent part
            dryl, dryr = (mL, mR) if no_reverb else slight_decorr2(mL, mR, 9015, 9016)
            dl, dr = place(dryl, dryr, p["height_gain"] * hs)
            rvL, rvR = scale_pair(prox_reverb("TSL"), prox_reverb("TSR"),
                                  p["height_gain"] * hs)
            out[pos["TSL"]] = dl + w * _hshelf(rvL)
            out[pos["TSR"]] = dr + w * _hshelf(rvR)

    if "TC" in pos:                       # voice-of-god (Auro 10.1/11.1)
        if in_heights:
            # 12-16 ch input mixed into an Auro layout: send the IN-PHASE
            # (coherent) content shared across the height channels up to the
            # Voice-of-God, exactly as the centre is extracted from a stereo pair.
            hl = 0.5 * (in_heights.get("TFL", np.zeros(n)) + in_heights.get("TBL", np.zeros(n)))
            hr = 0.5 * (in_heights.get("TFR", np.zeros(n)) + in_heights.get("TBR", np.zeros(n)))
            bc, ac = signal.butter(2, 90 / (sr / 2), btype="high")
            coh = signal.filtfilt(bc, ac, 0.5 * (hl + hr))   # in-phase height mono
            rv = scale_mono(prox_reverb("TC"), p["height_gain"] * 0.6 * hs)
            out[pos["TC"]] = coh + w * rbj_high_shelf(rv, sr, 5500, -(tilt + 1.5))
            passthru_keys.add("TC")
        else:
            src = 0.5 * (L + R) if no_reverb else dap(0.5 * (L + R), 9009)
            dry = place_mono(src, p["height_gain"] * 0.6 * hs)
            rv = scale_mono(prox_reverb("TC"), p["height_gain"] * 0.6 * hs)
            out[pos["TC"]] = dry + w * rbj_high_shelf(rv, sr, 5500, -(tilt + 1.5))
    if "TFC" in pos:                      # centre height (Auro 11.1)
        src = 0.5 * (L + R) if no_reverb else dap(0.5 * (L + R), 9010)
        dry = place_mono(src, p["height_gain"] * 0.45 * hs)
        rv = scale_mono(prox_reverb("TFC"), p["height_gain"] * 0.45 * hs)
        out[pos["TFC"]] = dry + w * rbj_high_shelf(rv, sr, 6000, -(tilt + 1.0))

    # Steer detected atmosphere / objects into the HEIGHT layer.  This is a
    # time-varying, level-NEUTRAL crossfade: where the analyzer reports overhead
    # content (presence `pres`), each height channel morphs toward the detected
    # signal; where nothing is detected the heights are untouched.  Because it
    # crossfades (rather than adds), the height level still obeys the front/
    # surround average, and the bed is never touched.
    if (steer_to_heights or immersive_3d) and steer_hL is not None:
        left = [k for k in ("TFL", "TBL", "TSL") if k in pos]
        right = [k for k in ("TFR", "TBR", "TSR") if k in pos]
        cen = [k for k in ("TC", "TFC") if k in pos]
        seeds = iter([9017, 9018, 9003, 9005, 9015, 9004, 9006, 9016, 9009, 9010])

        def _steer(keys, src):
            for kk_ in keys:
                gen = out[pos[kk_]]
                atmo = dap(src, next(seeds, 9017))
                rg = np.sqrt(np.mean(gen ** 2)); ra = np.sqrt(np.mean(atmo ** 2)) + 1e-9
                atmo = atmo * (rg / ra)                 # match level -> neutral
                wsteer = steer_amount * np.clip(steer_pres / 0.30, 0.0, 1.0)
                out[pos[kk_]] = (1.0 - wsteer) * gen + wsteer * atmo
        _steer(left, steer_hL)
        _steer(right, steer_hR)
        _steer(cen, 0.5 * (steer_hL + steer_hR))

    # front WIDE pair (9.1.6): passthrough if the input carries it (16 ch),
    # otherwise GENERATED from the side surround blended with the front, with the
    # phase subtraction + a SLIGHT inter-pair decorrelation (additional-channel
    # rule), sitting between the front and the side.
    if "FLW" in pos and "FRW" in pos:
        rvL, rvR = scale_pair(surr_reverb("FLW"), surr_reverb("FRW"), p["surr_gain"])
        if "FLW" in in_wide and "FRW" in in_wide:
            out[pos["FLW"]] = in_wide["FLW"] + w * rvL
            out[pos["FRW"]] = in_wide["FRW"] + w * rvR
            passthru_keys |= {"FLW", "FRW"}
        else:
            sz = in_zones.get("side", in_zones.get("rear"))
            sL, sR = (sz[0][:n], sz[1][:n]) if sz else (L, R)
            baseL = 0.5 * (L + sL); baseR = 0.5 * (R + sR)
            coh = 0.5 * (baseL + baseR)
            baseL = baseL - pd * coh; baseR = baseR - pd * coh   # phase subtraction
            dryl, dryr = slight_decorr2(baseL, baseR, 9013, 9014)
            dl, dr = scale_pair(dryl, dryr, p["surr_gain"])
            out[pos["FLW"]] = dl + w * rvL
            out[pos["FRW"]] = dr + w * rvR

    report(85, "Assembling output")
    M = np.zeros((n, len(labels)), dtype=np.float64)
    for i, lbl in enumerate(labels):
        if lbl in out:
            M[:, i] = out[lbl][:n]

    # Classify each output channel by its speaker key so we can level the height
    # layer and protect the bed.
    keyof = {l: key for l, key in layout}
    idx_of = lambda keys: [i for i, l in enumerate(labels) if keyof.get(l) in keys]
    front_i = idx_of({"FL", "FR", "FC"})
    surr_i = idx_of({"BL", "BR", "SL", "SR"})
    # heights subject to levelling = SYNTHESISED heights only (discrete
    # passthrough heights from a 7.1.4/9.1.6 input keep their original level).
    height_i = [i for i, l in enumerate(labels)
                if str(keyof.get(l, "")).startswith("T")
                and keyof.get(l) not in passthru_keys]
    gpk = lambda idx: float(np.max(np.abs(M[:, idx]))) if idx else 0.0

    # Dynamics following: reshape the SYNTHESISED field so its macro-envelope
    # tracks the original mix (reverb/ambience breathes with the source).  Applied
    # to non-bed, non-passthrough channels only; the bed stays bit-exact.
    bed_now = {"FL", "FR"} | set(passthru_keys)
    if in_C is not None:
        bed_now.add("FC")
    if in_LFE is not None:
        bed_now.add("LFE")
    synth_i = [i for i, l in enumerate(labels) if keyof.get(l) not in bed_now]
    if dynamics_follow and dyn_amount > 0 and synth_i and w > 0:
        dry_mono = audio[:n].mean(axis=1)
        synth_mono = M[:, synth_i].sum(axis=1)
        g = dynamics_follow_gain(dry_mono, synth_mono, sr, amount=dyn_amount)
        g = g[:n]
        M[:, synth_i] *= g[:, None]

    # FRONT-DOMINANCE CEILING (ear-level surround field only — heights are levelled
    # explicitly below and are exempt here).  A surround field spread over many
    # speakers can collectively overpower the front anchor and make the original
    # front "drown".  If the summed ear-level surround energy rises above a set
    # margin below the front pair, scale the surrounds down so the front leads.
    # Applied to synthesised (mono/stereo) fields in either mode and to PURE
    # surround→immersive; normal-mode multichannel keeps its discrete surrounds.
    # Only ever reduces, never boosts.
    ceil_db = None
    if not in_zones:
        ceil_db = -9.0          # synthesised field from a mono/stereo source
    elif no_reverb:
        ceil_db = -6.0          # surround → immersive, PURE: keep front clearly on top
    if ceil_db is not None:
        _excl = {"FL", "FR", "FC", "LFE", "FLW", "FRW"}
        fr_i = [i for i in range(len(labels)) if keyof.get(labels[i]) in ("FL", "FR")]
        df_i = [i for i in range(len(labels))
                if keyof.get(labels[i]) not in _excl
                and not str(keyof.get(labels[i], "")).startswith("T")]   # not heights
        if fr_i and df_i:
            fpow = float(np.sum(np.mean(M[:, fr_i] ** 2, axis=0)))
            dpow = float(np.sum(np.mean(M[:, df_i] ** 2, axis=0)))
            if fpow > 1e-12 and dpow > 1e-12:
                ratio_db = 10.0 * np.log10(dpow / fpow)
                if ratio_db > ceil_db:
                    M[:, df_i] *= 10.0 ** ((ceil_db - ratio_db) / 20.0)

    fronth_keys = {"TFL", "TFR", "TFC"}
    fronth_i = [i for i in height_i if keyof.get(labels[i]) in fronth_keys]
    rearh_i = [i for i in height_i if keyof.get(labels[i]) not in fronth_keys]
    lr_i = idx_of({"FL", "FR"})
    sur_i = idx_of({"BL", "BR", "SL", "SR"})

    # SMOOTH HEIGHT SWING-LIMITER.  With reverb the height tail keeps the field
    # continuous, but a PURE / Pro Logic dry extraction collapses whenever the
    # diffuse/anti-phase content it feeds on thins out — the height level then
    # swings tens of dB against the bed and "chops"/jumps.
    #
    # (1) PURE mode: give the height field a continuous floor so it never gates to
    #     silence — but REFLECTION-FREE.  A convolved/diffuse floor smears in time
    #     and reads as room reflections, which pure mode must not have.  Instead use
    #     the bed's own high-passed L/R with a partial side-matrix (subtract part of
    #     the opposite channel) — a pure linear mix, no tail — which keeps the floor
    #     continuous while pulling down the centre (dialogue) content.  Not used in
    #     Pro Logic (heights stay anti-phase only) nor normal mode (already smooth).
    if no_reverb and not prologic and height_i and strength > 0 and lr_i:
        _L = M[:, lr_i[0]].astype(np.float64); _R = M[:, lr_i[1]].astype(np.float64)
        _hb, _ha = signal.butter(2, 450.0 / (sr / 2), "high")
        _fl = signal.filtfilt(_hb, _ha, _L - 0.4 * _R)        # left, centre reduced
        _fr = signal.filtfilt(_hb, _ha, _R - 0.4 * _L)        # right, centre reduced
        _fc = 0.5 * (_fl + _fr)
        floors = {}
        for i in height_i:
            az = SPK_POS.get(keyof.get(labels[i], ""), (0.0, 0.0))[0]
            floors[i] = _fl if az < -5 else (_fr if az > 5 else _fc)
        fl_rms = np.sqrt(np.mean(np.stack([floors[i] for i in height_i]) ** 2)) + 1e-12
        h_rms = float(np.sqrt(np.mean(M[:, height_i] ** 2))) + 1e-12
        k = 0.7 * h_rms / fl_rms
        for i in height_i:
            M[:, i] += (k * floors[i][:n]).astype(M.dtype)

    # (2) Hold the height field within a smoothly-varying band around the bed so
    #     its level glides instead of gating.  The band is wide enough that a
    #     normal (reverb) upmix — already smooth — passes through untouched.
    if height_i and strength > 0:
        def _slow_env(x, ms):
            aa = np.exp(-1.0 / (sr * ms * 1e-3))
            e = signal.lfilter([1 - aa], [1.0, -aa], np.abs(x).astype(np.float64))
            e = signal.lfilter([1 - aa], [1.0, -aa], e[::-1])[::-1]   # zero-phase
            return e + 1e-9
        bed_ref = lr_i if lr_i else fronth_i
        eb = _slow_env(M[:, bed_ref].sum(axis=1), 120.0)
        eh = _slow_env(M[:, height_i].sum(axis=1), 120.0)
        r = eh / eb
        lo, hi = 10.0 ** (-12.0 / 20.0), 10.0 ** (4.0 / 20.0)   # band vs bed
        gain = np.clip(r, lo, hi) / r
        gain = np.clip(gain, 10.0 ** (-24.0 / 20.0), 10.0 ** (12.0 / 20.0))
        ag = np.exp(-1.0 / (sr * 120.0 * 1e-3))                 # de-zipper (zero-phase)
        gain = signal.lfilter([1 - ag], [1.0, -ag], gain)
        gain = signal.lfilter([1 - ag], [1.0, -ag], gain[::-1])[::-1]
        M[:, height_i] *= gain[:n, None]

    def _rms(idx):
        return float(np.sqrt(np.mean(M[:, idx] ** 2))) if idx else 0.0

    def _p99(idx):
        return float(np.percentile(np.abs(M[:, idx]), 99.9)) if idx else 0.0

    def _fit(idx, ref_rms, ref_pk):
        # Match loudness (RMS) to the reference, THEN cap the peak so the height
        # waveform is never taller than the reference's — heights read as "as loud
        # as" the fronts/surrounds without ever sitting above them.
        if not idx:
            return
        cur = _rms(idx)
        if cur > 1e-9 and ref_rms > 1e-9:
            M[:, idx] *= ref_rms / cur
        p = _p99(idx)
        if p > ref_pk > 1e-9:
            M[:, idx] *= ref_pk / p

    # Front heights track the FRONT (L/R); rear/side/overhead heights track the
    # SURROUND.  Mono/stereo source (no real surround): ALL heights track the
    # front.  RMS reference excludes the centre, so loud dialogue/score never
    # inflates the heights.  Applied in EVERY mode (incl. pure) so the heights are
    # always loudness-matched to the fronts, independent of preset/strength.
    f_rms, f_pk = _rms(lr_i), _p99(lr_i)
    if in_zones and sur_i:
        s_rms, s_pk = _rms(sur_i), _p99(sur_i)
    else:
        s_rms, s_pk = f_rms, f_pk          # stereo: heights == fronts
    _fit(fronth_i, f_rms, f_pk)
    _fit(rearh_i, s_rms, s_pk)

    # 3D IMMERSIVE: as detected content is lifted overhead, duck the EAR-LEVEL bed
    # (front L/R + every surround, synthesised or discrete) by up to
    # `immersive_duck_db`, so the atmosphere/objects read as genuinely above you
    # rather than all around at ear level.  The CENTRE (dialogue) and LFE are left
    # alone, and the duck follows the same presence envelope as the height steer —
    # when nothing overhead is detected the bed is at full level (gain = 1).
    if immersive_3d and steer_duck is not None and immersive_duck_db > 0:
        duck_keys = {"FL", "FR", "BL", "BR", "SL", "SR"}
        duck_i = idx_of(duck_keys)
        if duck_i:
            # slow, sustained-ambient presence (wind/storm/reflections/ambient
            # music/rotor wash) mapped to 0..1 — transients & dry/coherent content
            # read ~0, fully-diffuse content reaches the cap
            w_duck = np.clip(steer_duck[:n] / 0.35, 0.0, 1.0)
            gain = 10.0 ** (-(float(immersive_duck_db) / 20.0) * w_duck)
            M[:, duck_i] *= gain[:, None]

    # Clip protection that leaves the BED UNTOUCHED: the passthrough bed channels
    # (the discrete input that is carried through — L/R, plus C/LFE/surround bed
    # and any discrete input height/wide channels) are never scaled, so at
    # strength 0 they are bit-exact.  Only synthesised channels are attenuated if
    # anything would exceed full scale.
    bed_keys = {"FL", "FR"} | set(passthru_keys)
    if in_C is not None:
        bed_keys.add("FC")
    if in_LFE is not None:
        bed_keys.add("LFE")
    for zname, (kl, kr) in out_zones.items():
        if feed[zname]:
            bed_keys |= {kl, kr}
    bed_i = set(idx_of(bed_keys))
    nonbed_i = [i for i in range(len(labels)) if i not in bed_i]
    if np.max(np.abs(M)) > 0.999 if M.size else False:
        npk = float(np.max(np.abs(M[:, nonbed_i]))) if nonbed_i else 0.0
        if npk > 0.999:
            M[:, nonbed_i] *= 0.999 / npk
    report(100, "Done")
    return M.astype(np.float32), mask, labels


def out_zones_other(*_a, **_k):  # retained no-op for backward compat
    return set()


# ==========================================================================
#  WAVE_FORMAT_EXTENSIBLE float32 writer
# ==========================================================================
_SUBTYPE_FLOAT = b"\x03\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"

def write_wav_extensible(path, data, sr, channel_mask):
    data = np.ascontiguousarray(data.astype("<f4"))
    n, ch = data.shape
    bits = 32; block = ch * bits // 8; byte_rate = sr * block
    raw = data.tobytes()
    fmt = (struct.pack("<HHIIHH", 0xFFFE, ch, sr, byte_rate, block, bits)
           + struct.pack("<H", 22) + struct.pack("<H", bits)
           + struct.pack("<I", channel_mask) + _SUBTYPE_FLOAT)
    fact = struct.pack("<I", n)
    riff = 4 + (8 + len(fmt)) + (8 + len(fact)) + (8 + len(raw))
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", riff) + b"WAVE")
        f.write(b"fmt " + struct.pack("<I", len(fmt)) + fmt)
        f.write(b"fact" + struct.pack("<I", len(fact)) + fact)
        f.write(b"data" + struct.pack("<I", len(raw)) + raw)
