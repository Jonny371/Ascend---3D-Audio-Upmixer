"""
Ascend — 3D Height Upmixer  ::  GUI  (PySide6)
==============================================
Run:  python ascend_gui.py
Independent implementation of an Auro-Matic-style upmixer (see engine.py).
"""
from __future__ import annotations
import os, sys, traceback
import numpy as np

try:
    import soundfile as sf
except Exception:
    sf = None

from PySide6.QtCore import Qt, QObject, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QFileDialog, QLabel, QPushButton,
    QComboBox, QSlider, QSpinBox, QDoubleSpinBox, QCheckBox, QProgressBar,
    QPlainTextEdit, QGroupBox, QGridLayout, QVBoxLayout, QHBoxLayout,
    QFormLayout, QFrame, QSizePolicy,
)

import engine as E


# --------------------------------------------------------------------------
# Audio input: WAV/FLAC/AIFF natively (soundfile); MP3/AAC/M4A/DTS-HD/TrueHD
# and any other container via ffmpeg.  If no ffmpeg is found, a bundled binary
# is installed on demand (pip install imageio-ffmpeg).
# --------------------------------------------------------------------------
# Formats routed through ffmpeg (soundfile can't open these directly).
FFMPEG_EXTS = {".mp3", ".aac", ".m4a", ".mp4", ".m4v", ".mov", ".dts", ".dtshd",
               ".dtsma", ".thd", ".ac3", ".eac3", ".ec3", ".mka", ".mkv",
               ".webm", ".ogg", ".opus", ".wma", ".ts", ".m2ts", ".mts"}
INPUT_FILTER = ("Audio / video ("
                "*.wav *.flac *.aiff *.aif *.w64 *.mp3 *.aac *.m4a *.mp4 "
                "*.dts *.dtshd *.dtsma *.thd *.ac3 *.eac3 *.mka *.mkv *.webm "
                "*.ogg *.opus *.wma *.ts *.m2ts);;All files (*.*)")


def _find_ffmpeg(installer=None):
    """Return a path to an ffmpeg binary, installing imageio-ffmpeg if needed.
    `installer` is an optional callback(str) for progress logging."""
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    if installer:
        installer("No ffmpeg found — installing a bundled copy (imageio-ffmpeg)…")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                    "imageio-ffmpeg"], check=True)
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def load_any_audio(path, log=None):
    """Load any supported file to (float32 array, samplerate).  Compressed and
    lossless-bitstream formats (MP3/AAC/M4A/DTS-HD/Dolby TrueHD/…) are decoded to
    full-resolution multichannel PCM with ffmpeg, preserving the channel layout
    in standard WAV order (L R C LFE BL BR SL SR …)."""
    ext = os.path.splitext(path)[1].lower()
    if sf is not None and ext not in FFMPEG_EXTS:
        try:
            return sf.read(path, dtype="float32", always_2d=False)
        except Exception:
            pass  # fall through to ffmpeg (e.g. an exotic WAV codec)
    import subprocess, tempfile
    ffmpeg = _find_ffmpeg(installer=log)
    tmp = tempfile.mktemp(suffix=".wav")
    # decode the (first) audio stream to 32-bit float PCM, all channels, rf64 so
    # long films aren't capped at the 4 GB WAV limit
    cmd = [ffmpeg, "-y", "-i", path, "-map", "0:a:0", "-vn",
           "-c:a", "pcm_f32le", "-rf64", "auto", tmp]
    if log:
        log(f"Decoding {os.path.basename(path)} via ffmpeg…")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0 or not os.path.exists(tmp):
        err = res.stderr.decode("utf-8", "ignore")[-400:]
        raise RuntimeError(f"ffmpeg could not decode this file:\n{err}")
    try:
        data, sr = sf.read(tmp, dtype="float32", always_2d=False)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    return data, sr


# --------------------------------------------------------------------------
# Background worker so the UI never freezes during processing
# --------------------------------------------------------------------------
class Worker(QObject):
    progress = Signal(int, str)
    finished = Signal(object, object, object)   # (matrix, mask, labels)
    failed = Signal(str)

    def __init__(self, audio, sr, layout, preset, strength, kwargs):
        super().__init__()
        self.audio, self.sr = audio, sr
        self.layout, self.preset, self.strength = layout, preset, strength
        self.kwargs = kwargs

    def run(self):
        try:
            M, mask, labels = E.upmix(
                self.audio, self.sr, self.layout, self.preset, self.strength,
                progress=lambda p, m="": self.progress.emit(int(p), m),
                **self.kwargs,
            )
            self.finished.emit(M, mask, labels)
        except Exception:
            self.failed.emit(traceback.format_exc())


# --------------------------------------------------------------------------
class LevelBar(QWidget):
    """Tiny horizontal RMS bar with a channel label, drawn after processing."""
    def __init__(self, label: str, level: float):
        super().__init__()
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 1, 0, 1)
        name = QLabel(label); name.setFixedWidth(110)
        name.setStyleSheet("color:#cdd6f4;")
        bar = QProgressBar(); bar.setRange(0, 100); bar.setTextVisible(False)
        bar.setFixedHeight(12)
        bar.setValue(int(min(1.0, level / 0.5) * 100))
        val = QLabel(f"{20*np.log10(level+1e-9):5.1f} dB" if level > 0 else " -inf  ")
        val.setFixedWidth(64); val.setStyleSheet("color:#94a3b8;")
        lay.addWidget(name); lay.addWidget(bar, 1); lay.addWidget(val)


class Ascend(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ascend — 3D Height Upmixer")
        self.resize(720, 760)
        self.audio = None
        self.sr = None
        self.in_path = None

        root = QWidget(); self.setCentralWidget(root)
        outer = QVBoxLayout(root); outer.setContentsMargins(16, 14, 16, 14); outer.setSpacing(10)

        title = QLabel("ASCEND")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        title.setStyleSheet("color:#89b4fa; letter-spacing:4px;")
        outer.addWidget(title)
        outer.addWidget(self._hline())

        # ---- input file -------------------------------------------------
        in_box = QGroupBox("Input")
        ib = QGridLayout(in_box)
        self.btn_in = QPushButton("Choose audio file…")
        self.btn_in.clicked.connect(self.pick_input)
        self.lbl_in = QLabel("No file selected"); self.lbl_in.setStyleSheet("color:#94a3b8;")
        self.lbl_info = QLabel(""); self.lbl_info.setStyleSheet("color:#7f8ea3;")
        self.lbl_flow = QLabel(""); self.lbl_flow.setWordWrap(True)
        self.lbl_flow.setStyleSheet("color:#5fb0d6;")
        self.cmb_order = QComboBox(); self.cmb_order.setEnabled(False)
        self.cmb_order.addItem("(channel order — load a 5.1/7.1 file)")
        ib.addWidget(self.btn_in, 0, 0)
        ib.addWidget(self.lbl_in, 0, 1)
        ib.addWidget(self.lbl_info, 1, 0, 1, 2)
        ib.addWidget(QLabel("Input order"), 2, 0)
        ib.addWidget(self.cmb_order, 2, 1)
        outer.addWidget(in_box)

        # ---- main controls ---------------------------------------------
        ctl = QGroupBox("Upmix")
        cf = QFormLayout(ctl)
        self.cmb_layout = QComboBox(); self.cmb_layout.addItems(list(E.LAYOUTS.keys()))
        self.cmb_layout.setCurrentText("7.1.4")
        self.cmb_preset = QComboBox(); self.cmb_preset.addItems(list(E.PRESETS.keys()))
        self.cmb_preset.setCurrentText("Medium")
        self.cmb_preset.currentTextChanged.connect(self._update_flow)
        self.cmb_layout.currentTextChanged.connect(self._update_flow)

        self.sld = QSlider(Qt.Horizontal); self.sld.setRange(0, 16); self.sld.setValue(12)
        self.lbl_str = QLabel("12")
        srow = QHBoxLayout(); srow.addWidget(self.sld, 1); srow.addWidget(self.lbl_str)
        self.sld.valueChanged.connect(lambda v: self.lbl_str.setText(str(v)))
        self.sld.valueChanged.connect(self._update_flow)

        cf.addRow("Output layout", self.cmb_layout)
        cf.addRow("Preset", self.cmb_preset)
        cf.addRow("Strength (0–16: dry⟶wet)", self._wrap(srow))
        outer.addWidget(ctl)

        # ---- options (no reverb tuning: presets are calibrated; only the
        #      strength above is adjustable) -------------------------------
        opt = QGroupBox("Options")
        of = QVBoxLayout(opt)
        self.chk_center = QCheckBox("Generate centre from coherent signal"); self.chk_center.setChecked(True)
        self.chk_lfe    = QCheckBox("Generate LFE (low-pass sum) if absent"); self.chk_lfe.setChecked(True)
        self.chk_noverb = QCheckBox("Pure upmix — no reverb / reflections")
        self.chk_noverb.setChecked(False)

        def _noverb_toggled(on):
            self.sld.setEnabled(not on)
            self.lbl_str.setEnabled(not on)
            self._update_flow()
        self.chk_noverb.toggled.connect(_noverb_toggled)
        self.chk_decorr = QCheckBox("Widening"); self.chk_decorr.setChecked(False)
        self.chk_drysur = QCheckBox("3D Reverb Environment")
        self.chk_drysur.setChecked(True)
        self.chk_phase = QCheckBox("Phase-difference height source")
        self.chk_phase.setChecked(True)
        self.chk_pl = QCheckBox("Dolby Pro Logic decode (auto-detected matrix surround)"); self.chk_pl.setChecked(False)
        self.chk_dyn = QCheckBox("Dynamics follow")
        self.chk_dyn.setChecked(True)
        self.chk_steer = QCheckBox("Steer atmosphere / objects to heights")
        self.chk_steer.setChecked(False)
        # 3D Immersive: steer overhead content up AND duck the ear-level bed.
        # The bed duck is fixed at 2 dB.
        self.chk_imm = QCheckBox("3D Immersive")
        self.chk_imm.setChecked(False)
        self.chk_pl.toggled.connect(self._pl_toggled)
        self.chk_drysur.toggled.connect(self._update_flow)
        self.chk_phase.toggled.connect(self._update_flow)
        of.addWidget(self.chk_center)
        of.addWidget(self.chk_lfe)
        of.addWidget(self.chk_noverb)
        of.addWidget(self.chk_decorr)
        of.addWidget(self.chk_drysur)
        of.addWidget(self.chk_phase)
        of.addWidget(self.chk_dyn)
        of.addWidget(self.chk_steer)
        of.addWidget(self.chk_imm)
        of.addWidget(self.chk_pl)
        outer.addWidget(opt)

        # ---- output + run ----------------------------------------------
        run_box = QGroupBox("Output")
        rg = QGridLayout(run_box)
        self.btn_out = QPushButton("Output: (auto)")
        self.btn_out.clicked.connect(self.pick_output)
        self.out_path = None
        self.btn_run = QPushButton("▶  Upmix")
        self.btn_run.setStyleSheet("background:#89b4fa; color:#11111b; font-weight:bold; padding:8px;")
        self.btn_run.clicked.connect(self.run_upmix)
        self.bar = QProgressBar(); self.bar.setValue(0)
        rg.addWidget(self.btn_out, 0, 0, 1, 2)
        rg.addWidget(self.bar, 1, 0)
        rg.addWidget(self.btn_run, 1, 1)
        outer.addWidget(run_box)

        # ---- meters + log ----------------------------------------------
        self.meter_box = QGroupBox("Channel levels (post-upmix)")
        self.meter_lay = QVBoxLayout(self.meter_box)
        self.meter_box.setVisible(False)
        outer.addWidget(self.meter_box)

        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setFixedHeight(96)
        self.log.setStyleSheet("background:#181825; color:#a6adc8; font-family:Consolas,monospace;")
        outer.addWidget(self.log)
        self.setStyleSheet(self._qss())

    # ---- small helpers -------------------------------------------------
    def _hline(self):
        f = QFrame(); f.setFrameShape(QFrame.HLine); f.setStyleSheet("color:#313244;"); return f

    def _wrap(self, layout):
        w = QWidget(); w.setLayout(layout); return w

    def _ispin(self, lo, hi, val, suffix=""):
        s = QSpinBox(); s.setRange(lo, hi); s.setValue(val); s.setSuffix(suffix); return s

    def _dspin(self, lo, hi, step, val, suffix=""):
        s = QDoubleSpinBox(); s.setRange(lo, hi); s.setSingleStep(step)
        s.setValue(val); s.setSuffix(suffix); s.setDecimals(2); return s

    def log_msg(self, m):
        self.log.appendPlainText(m)

    # ---- file IO -------------------------------------------------------
    def _pl_toggled(self, on):
        # Pro Logic supplies a real mono surround, so decorrelation is forced
        # off in that mode.
        self.chk_decorr.setEnabled(not on)

    def pick_input(self):
        if sf is None:
            self.log_msg("ERROR: pip install soundfile"); return
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose audio", "", INPUT_FILTER)
        if not path:
            return
        try:
            data, sr = load_any_audio(path, log=self.log_msg)
        except Exception as e:
            self.log_msg(f"Could not read file: {e}"); return
        self.audio, self.sr, self.in_path = data, sr, path
        ch = 1 if data.ndim == 1 else data.shape[1]
        dur = (len(data)) / sr
        self.lbl_in.setText(os.path.basename(path))
        enc, info = E.detect_prologic(data, sr)
        pl_tag = ""
        if ch == 2:
            pl_tag = (f"  ·  Dolby Pro Logic: DETECTED (ρ={info['rho']:+.2f})"
                      if enc else f"  ·  Pro Logic: not detected (ρ={info['rho']:+.2f})")
        self.chk_pl.setChecked(bool(enc))
        self.in_ch = ch
        self.lbl_info.setText(f"{ch} ch · {sr} Hz · {dur:0.1f}s{pl_tag}")
        # populate input channel-order choices for multichannel input
        self.cmb_order.clear()
        if ch >= 13:
            self.cmb_order.addItems(list(E.ORDERS_16.keys())); self.cmb_order.setEnabled(True)
        elif ch >= 9:
            self.cmb_order.addItems(list(E.ORDERS_12.keys())); self.cmb_order.setEnabled(True)
        elif ch >= 7:
            self.cmb_order.addItems(list(E.ORDERS_8.keys())); self.cmb_order.setEnabled(True)
        elif ch >= 5:
            self.cmb_order.addItems(list(E.ORDERS_6.keys())); self.cmb_order.setEnabled(True)
        else:
            self.cmb_order.addItem("(stereo/mono — no surround order)")
            self.cmb_order.setEnabled(False)
        base, _ = os.path.splitext(path)
        self.out_path = base + "_ascend.wav"
        self.btn_out.setText("Output: " + os.path.basename(self.out_path))
        self._update_flow()
        self.log_msg(f"Loaded {os.path.basename(path)}  ({ch} ch, {sr} Hz)")

    def _update_flow(self, *a):
        ch = getattr(self, "in_ch", 0)
        if not ch:
            self.lbl_flow.setText(""); return
        preset = self.cmb_preset.currentText()
        layout = self.cmb_layout.currentText()
        s = self.sld.value()
        morph = ("PURE UPMIX — no reverb / reflections (dry spatial field only)"
                 if self.chk_noverb.isChecked()
                 else "dry direct only" if s == 0
                 else "dry + full proximity reverb" if s >= 16
                 else f"dry + {int(round(s/16*100))}% reverb")
        spread = self.chk_drysur.isChecked()
        sp = ("60% adjacent / 40% rest by distance" if spread
              else "adjacent speaker only")
        if ch == 2:
            t = (f"Workflow: stereo → {layout}. Bed intact; each surround + "
                 f"height speaker = dry decorrelated direct PLUS reverb of the "
                 f"nearest speakers ({sp}). Strength {s}/16 → {morph}.")
        elif ch >= 6:
            is71 = ch >= 8
            pd = self.chk_phase.isChecked()
            fsub = "L−Ls, R−Rs" if pd else "no subtraction"
            rsub = (" (rear−side Ls−Lss)" if (pd and is71) else "")
            gen = (" Rear/back zone generated from decorrelated Ls/Rs."
                   if not is71 else "")
            t = (f"Workflow: {ch}ch → {layout}. Discrete bed intact. "
                 f"Each generated speaker = dry direct PLUS proximity reverb "
                 f"({sp}): front heights' dry from front−surround ({fsub}); "
                 f"back heights' dry from the surround{rsub}. "
                 f"Strength {s}/16 → {morph}.{gen}")
        else:
            t = (f"Workflow: mono → {layout} (dry direct + proximity reverb, "
                 f"{morph}).")
        self.lbl_flow.setText(t)

    def pick_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save WAV", self.out_path or "", "WAV (*.wav)")
        if path:
            self.out_path = path
            self.btn_out.setText("Output: " + os.path.basename(path))

    # ---- run -----------------------------------------------------------
    def run_upmix(self):
        if self.audio is None:
            self.log_msg("Select an input file first."); return
        if not self.out_path:
            base, _ = os.path.splitext(self.in_path); self.out_path = base + "_ascend.wav"
        self.btn_run.setEnabled(False); self.bar.setValue(0)
        self.meter_box.setVisible(False)
        kwargs = dict(
            center_gen=self.chk_center.isChecked(),
            gen_lfe=self.chk_lfe.isChecked(),
            decorrelate=self.chk_decorr.isChecked(),
            prologic=self.chk_pl.isChecked(),
            surr_dry_lift=self.chk_drysur.isChecked(),
            height_phase_diff=self.chk_phase.isChecked(),
            dynamics_follow=self.chk_dyn.isChecked(),
            steer_to_heights=self.chk_steer.isChecked(),
            immersive_3d=self.chk_imm.isChecked(),
            immersive_duck_db=2.0,
            no_reverb=self.chk_noverb.isChecked(),
            input_order=(self.cmb_order.currentText() if self.cmb_order.isEnabled() else None),
        )
        self.thread = QThread()
        self.worker = Worker(self.audio, self.sr, self.cmb_layout.currentText(),
                             self.cmb_preset.currentText(), self.sld.value(), kwargs)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.thread.start()
        self.log_msg(f"Upmixing → {self.cmb_layout.currentText()} / {self.cmb_preset.currentText()} / strength {self.sld.value()}")

    def on_progress(self, p, m):
        self.bar.setValue(p)
        if m:
            self.log_msg(f"  {p:3d}%  {m}")

    def on_finished(self, M, mask, labels):
        try:
            E.write_wav_extensible(self.out_path, M, self.sr, mask)
            self.log_msg(f"Wrote {os.path.basename(self.out_path)}  ({M.shape[1]} ch, mask 0x{mask:X})")
            self._show_meters(M, labels)
        except Exception as e:
            self.log_msg(f"Write failed: {e}")
        self.thread.quit(); self.thread.wait()
        self.btn_run.setEnabled(True)

    def on_failed(self, tb):
        self.log_msg("FAILED:\n" + tb)
        self.thread.quit(); self.thread.wait()
        self.btn_run.setEnabled(True)

    def _show_meters(self, M, labels):
        while self.meter_lay.count():
            w = self.meter_lay.takeAt(0).widget()
            if w: w.deleteLater()
        rms = np.sqrt(np.mean(M.astype(np.float64) ** 2, axis=0))
        for lbl, r in zip(labels, rms):
            self.meter_lay.addWidget(LevelBar(lbl, float(r)))
        self.meter_box.setVisible(True)

    def _qss(self):
        return """
        QMainWindow, QWidget { background:#11111b; color:#cdd6f4; font-family:'Segoe UI'; font-size:10pt; }
        QGroupBox { border:1px solid #313244; border-radius:8px; margin-top:10px; padding:10px; }
        QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 4px; color:#89b4fa; }
        QPushButton { background:#313244; color:#cdd6f4; border:none; border-radius:6px; padding:7px 12px; }
        QPushButton:hover { background:#45475a; }
        QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit { background:#1e1e2e; border:1px solid #313244; border-radius:5px; padding:4px; }
        QProgressBar { background:#1e1e2e; border:1px solid #313244; border-radius:5px; }
        QProgressBar::chunk { background:#89b4fa; border-radius:4px; }
        QSlider::groove:horizontal { height:5px; background:#313244; border-radius:3px; }
        QSlider::handle:horizontal { background:#89b4fa; width:14px; margin:-5px 0; border-radius:7px; }
        QCheckBox { color:#cdd6f4; }
        """


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    w = Ascend(); w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
