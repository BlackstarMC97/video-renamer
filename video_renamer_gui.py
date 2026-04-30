"""
DVR Video Renamer — Desktop GUI
Requires: pip install customtkinter
All other dependencies (doctr, opencv, ffmpeg) are the same as the CLI version.

To build a standalone executable:
    pip install pyinstaller
    pyinstaller video_renamer_gui.py --onefile --windowed --name "DVR Renamer"
"""

import os
import sys
import threading
import queue
import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk

# ── Import the core renamer (must be in the same folder) ─────────────────────
# We do a lazy import inside the worker thread so GUI launches even if
# heavy ML deps (doctr, torch) are slow to load.

# ── App-wide appearance ───────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

FONT_LABEL  = ("Segoe UI", 12)
FONT_HEADER = ("Segoe UI", 13, "bold")
FONT_MONO   = ("Consolas", 11)
FONT_BTN    = ("Segoe UI", 13, "bold")

# Accent colours used by the log panel
LOG_COLORS = {
    "SUCCESS": "#4ade80",   # green
    "FAILED":  "#f87171",   # red
    "ERROR":   "#f87171",
    "SCAN":    "#60a5fa",   # blue
    "INFO":    "#94a3b8",   # muted
    "WARNING": "#fbbf24",   # amber
}


# ─────────────────────────────────────────────────────────────────────────────
# Redirect stdout/stderr into a queue so the GUI can show live log output
# ─────────────────────────────────────────────────────────────────────────────
class QueueWriter:
    """Replaces sys.stdout so print() output flows into the GUI log."""
    def __init__(self, q: queue.Queue):
        self.q = q

    def write(self, text: str):
        if text.strip():
            self.q.put(text)

    def flush(self):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Collapsible section widget
# ─────────────────────────────────────────────────────────────────────────────
class CollapsibleSection(ctk.CTkFrame):
    def __init__(self, master, title: str, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self._open = True

        self.header = ctk.CTkButton(
            self, text=f"▾  {title}", anchor="w",
            font=FONT_HEADER,
            fg_color="transparent",
            hover_color=("gray85", "gray25"),
            text_color=("gray10", "gray90"),
            command=self._toggle,
        )
        self.header.pack(fill="x", pady=(8, 2))

        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.pack(fill="x", padx=8)

    def _toggle(self):
        if self._open:
            self.body.pack_forget()
            self.header.configure(text=self.header.cget("text").replace("▾", "▸"))
        else:
            self.body.pack(fill="x", padx=8)
            self.header.configure(text=self.header.cget("text").replace("▸", "▾"))
        self._open = not self._open


# ─────────────────────────────────────────────────────────────────────────────
# Helper: labelled row  [Label]  [Widget]
# ─────────────────────────────────────────────────────────────────────────────
def labeled_row(parent, label: str, widget_factory, tooltip: str = ""):
    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(fill="x", pady=3)

    lbl = ctk.CTkLabel(row, text=label, font=FONT_LABEL,
                        anchor="w", width=170)
    lbl.pack(side="left")

    widget = widget_factory(row)
    widget.pack(side="left", fill="x", expand=True)

    if tooltip:
        lbl.bind("<Enter>", lambda e, t=tooltip: _show_tooltip(lbl, t))
        lbl.bind("<Leave>", lambda e: _hide_tooltip())

    return widget


_tooltip_win = None

def _show_tooltip(widget, text):
    global _tooltip_win
    _hide_tooltip()
    x = widget.winfo_rootx() + 10
    y = widget.winfo_rooty() + 24
    _tooltip_win = tk.Toplevel(widget)
    _tooltip_win.wm_overrideredirect(True)
    _tooltip_win.geometry(f"+{x}+{y}")
    tk.Label(_tooltip_win, text=text, bg="#1e293b", fg="#e2e8f0",
             font=("Segoe UI", 10), padx=8, pady=4,
             wraplength=280, justify="left").pack()

def _hide_tooltip():
    global _tooltip_win
    if _tooltip_win:
        _tooltip_win.destroy()
        _tooltip_win = None


# ─────────────────────────────────────────────────────────────────────────────
# Main application window
# ─────────────────────────────────────────────────────────────────────────────
class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("DVR Video Renamer")
        self.geometry("740x600")
        self.update_idletasks()
        self.geometry(f"+{(self.winfo_screenwidth()-740)//2}+{(self.winfo_screenheight()-600)//2}")
        self.minsize(620, 600)
        self.resizable(True, True)

        self._log_queue: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None

        self._build_ui()
        self._poll_log()   # start the 50 ms log-polling loop

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Title bar area ────────────────────────────────────────────────────
        title_bar = ctk.CTkFrame(self, fg_color=("gray90", "gray15"), corner_radius=0)
        title_bar.pack(fill="x")

        ctk.CTkLabel(
            title_bar,
            text="📹  DVR Video Renamer",
            font=("Segoe UI", 16, "bold"),
            anchor="w",
        ).pack(side="left", padx=20, pady=12)

        ctk.CTkLabel(
            title_bar,
            text="Batch-rename recordings using OCR",
            font=("Segoe UI", 11),
            text_color=("gray50", "gray60"),
            anchor="e",
        ).pack(side="right", padx=20)

        # ── Main scrollable panel ─────────────────────────────────────────────
        scroll = ctk.CTkScrollableFrame(self, label_text="", height=300)
        scroll.pack(fill="both", expand=False, padx=16, pady=(12, 0))

        self._build_folder_section(scroll)
        self._build_core_section(scroll)
        self._build_scan_section(scroll)
        self._build_image_section(scroll)
        self._build_flags_section(scroll)

        # ── Divider ───────────────────────────────────────────────────────────
        ctk.CTkFrame(self, height=1, fg_color=("gray80", "gray30")).pack(
            fill="x", padx=16, pady=(8, 0))

        # ── Run button + progress ─────────────────────────────────────────────
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=16, pady=8)

        self.run_btn = ctk.CTkButton(
            btn_frame,
            text="▶  Run Renamer",
            font=FONT_BTN,
            height=42,
            corner_radius=8,
            command=self._on_run,
        )
        self.run_btn.pack(fill="x")

        self.progress = ctk.CTkProgressBar(self, mode="indeterminate")
        # shown only while running

        # ── Log panel ─────────────────────────────────────────────────────────
        log_header = ctk.CTkFrame(self, fg_color="transparent")
        log_header.pack(fill="x", padx=16)

        ctk.CTkLabel(log_header, text="Output log", font=FONT_HEADER,
                     anchor="w").pack(side="left")

        ctk.CTkButton(
            log_header, text="Clear", width=60, height=24,
            font=("Segoe UI", 11),
            fg_color="transparent",
            hover_color=("gray85", "gray25"),
            command=self._clear_log,
        ).pack(side="right")

        self.log_box = ctk.CTkTextbox(
            self,
            font=FONT_MONO,
            height=120,
            wrap="word",
            state="disabled",
        )
        self.log_box.pack(fill="both", expand=False, padx=16, pady=(4, 16))

        # Configure colour tags on the underlying tk.Text widget
        tw = self.log_box._textbox
        for key, color in LOG_COLORS.items():
            tw.tag_configure(key, foreground=color)

    # ── Section builders ──────────────────────────────────────────────────────

    def _build_folder_section(self, parent):
        sec = CollapsibleSection(parent, "Folder")
        sec.pack(fill="x")

        row = ctk.CTkFrame(sec.body, fg_color="transparent")
        row.pack(fill="x", pady=4)

        self.folder_var = tk.StringVar(value="")
        entry = ctk.CTkEntry(row, textvariable=self.folder_var,
                             placeholder_text="Select a folder…", font=FONT_LABEL)
        entry.pack(side="left", fill="x", expand=True, padx=(0, 8))

        ctk.CTkButton(
            row, text="Browse…", width=90, font=FONT_LABEL,
            command=self._browse_folder,
        ).pack(side="left")

    def _build_core_section(self, parent):
        sec = CollapsibleSection(parent, "General settings")
        sec.pack(fill="x")

        self.prefix_var = tk.StringVar(value="0")
        labeled_row(sec.body, "Forced prefix",
                    lambda p: ctk.CTkEntry(p, textvariable=self.prefix_var,
                                           placeholder_text="Leave blank for auto-detect",
                                           font=FONT_LABEL),
                    tooltip="Override the filename prefix instead of using OCR-detected text.")

        self.fallback_var = tk.IntVar(value=30)
        labeled_row(sec.body, "Fallback minutes",
                    lambda p: ctk.CTkEntry(p, textvariable=self.fallback_var,
                                           font=FONT_LABEL, width=100),
                    tooltip="Minutes added/subtracted when only one timestamp is found.")

    def _build_scan_section(self, parent):
        sec = CollapsibleSection(parent, "Multi-frame scan")
        sec.pack(fill="x")

        self.frame_step_var = tk.IntVar(value=5)
        labeled_row(sec.body, "Frame step (s)",
                    lambda p: ctk.CTkEntry(p, textvariable=self.frame_step_var,
                                           font=FONT_LABEL, width=100),
                    tooltip="Seconds between frames when scanning for a confident timestamp. "
                            "Lower = slower but more thorough.")

        self.min_conf_var = tk.DoubleVar(value=0.85)
        labeled_row(sec.body, "Min confidence",
                    lambda p: self._conf_slider_widget(p),
                    tooltip="Minimum mean OCR confidence (0–1) to accept a reading. "
                            "Lower = more permissive, higher = stricter.")

    def _conf_slider_widget(self, parent):
        """Slider + live numeric label for the confidence threshold."""
        frame = ctk.CTkFrame(parent, fg_color="transparent")

        self._conf_label = ctk.CTkLabel(frame, text=f"{self.min_conf_var.get():.2f}",
                                         font=FONT_LABEL, width=40)
        self._conf_label.pack(side="right")

        slider = ctk.CTkSlider(
            frame,
            from_=0.0, to=1.0,
            variable=self.min_conf_var,
            command=lambda v: self._conf_label.configure(text=f"{v:.2f}"),
        )
        slider.pack(side="left", fill="x", expand=True, padx=(0, 8))
        return frame

    def _build_image_section(self, parent):
        sec = CollapsibleSection(parent, "Image preprocessing")
        sec.pack(fill="x")

        self.brightness_var = tk.IntVar(value=180)
        labeled_row(sec.body, "Brightness threshold",
                    lambda p: ctk.CTkEntry(p, textvariable=self.brightness_var,
                                           font=FONT_LABEL, width=100),
                    tooltip="Minimum V (brightness) in HSV to keep. 160 = lenient, 200 = strict.")

        self.sat_max_var = tk.IntVar(value=80)
        labeled_row(sec.body, "Sat max",
                    lambda p: ctk.CTkEntry(p, textvariable=self.sat_max_var,
                                           font=FONT_LABEL, width=100),
                    tooltip="Max saturation to keep. Raise to ~120 for yellow DVR timestamps.")

        # Region: four individual fields so it stays clean
        region_row = ctk.CTkFrame(sec.body, fg_color="transparent")
        region_row.pack(fill="x", pady=3)

        ctk.CTkLabel(region_row, text="Region (x1 y1 x2 y2)",
                     font=FONT_LABEL, anchor="w", width=170).pack(side="left")

        self.region_vars = [tk.StringVar(value=v) for v in ("0", "0", "1", "0.25")]
        placeholders = ["x1", "y1", "x2", "y2"]
        for var, ph in zip(self.region_vars, placeholders):
            ctk.CTkEntry(
                region_row, textvariable=var,
                placeholder_text=ph,
                font=FONT_LABEL, width=58,
            ).pack(side="left", padx=2)

        ctk.CTkLabel(region_row,
                     text="  fractions 0–1",
                     font=("Segoe UI", 10),
                     text_color=("gray50", "gray60")).pack(side="left", padx=4)

    def _build_flags_section(self, parent):
        sec = CollapsibleSection(parent, "Flags")
        sec.pack(fill="x")

        self.radio_var     = tk.BooleanVar(value=False)
        self.debug_var     = tk.BooleanVar(value=False)
        self.aggressive_var = tk.BooleanVar(value=False)

        flags = [
            (self.radio_var,      "Radio mode",
             "Skip brightness/saturation filtering (for simple backgrounds)."),
            (self.debug_var,      "Debug images",
             "Save intermediate preprocessing images next to each frame."),
            (self.aggressive_var, "Aggressive",
             "More aggressive background rejection for white text on TV recordings."),
        ]

        flag_row = ctk.CTkFrame(sec.body, fg_color="transparent")
        flag_row.pack(fill="x", pady=4)

        for var, label, tip in flags:
            cb = ctk.CTkCheckBox(flag_row, text=label, variable=var, font=FONT_LABEL)
            cb.pack(side="left", padx=(0, 20))
            cb.bind("<Enter>", lambda e, t=tip: _show_tooltip(cb, t))
            cb.bind("<Leave>", lambda e: _hide_tooltip())

    # ── Actions ───────────────────────────────────────────────────────────────

    def _browse_folder(self):
        path = filedialog.askdirectory(title="Select video folder")
        if path:
            self.folder_var.set(path)

    def _on_run(self):
        if self._worker and self._worker.is_alive():
            self._log("⚠ Already running — please wait.", tag="WARNING")
            return

        folder = self.folder_var.get().strip()
        if not folder or not os.path.isdir(folder):
            self._log("✗ Please select a valid folder first.", tag="FAILED")
            return

        # Build kwargs for VideoRenamer
        region = self._parse_region()
        kwargs = dict(
            folder_path        = folder,
            fallback_minutes   = self._safe_int(self.fallback_var, 30),
            forced_prefix      = self.prefix_var.get().strip() or None,
            brightness_threshold = self._safe_int(self.brightness_var, 180),
            sat_max            = self._safe_int(self.sat_max_var, 80),
            region             = region,
            radio_mode         = self.radio_var.get(),
            debug              = self.debug_var.get(),
            aggressive         = self.aggressive_var.get(),
            frame_step         = self._safe_int(self.frame_step_var, 5),
            min_confidence     = round(self.min_conf_var.get(), 2),
        )

        self._set_running(True)
        self._worker = threading.Thread(target=self._run_worker, kwargs=kwargs, daemon=True)
        self._worker.start()

    def _run_worker(self, **kwargs):
        # Redirect stdout into the queue so all print() calls appear in the log
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = QueueWriter(self._log_queue)
        sys.stderr = QueueWriter(self._log_queue)

        try:
            from video_renamer import VideoRenamer   # lazy import
            renamer = VideoRenamer(**kwargs)
            renamer.process_folder()
        except ImportError as e:
            self._log_queue.put(
                f"[ERROR] Could not import video_renamer.py — make sure it is in the same folder.\n{e}")
        except Exception as e:
            self._log_queue.put(f"[ERROR] Unexpected error: {e}")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            # Signal completion back to the main thread
            self._log_queue.put("__DONE__")

    # ── Log panel ─────────────────────────────────────────────────────────────

    def _poll_log(self):
        """Called every 50 ms to drain the queue into the log textbox."""
        try:
            while True:
                msg = self._log_queue.get_nowait()
                if msg == "__DONE__":
                    self._set_running(False)
                    self._log("─── Finished ───", tag=None)
                else:
                    self._log(msg)
        except queue.Empty:
            pass
        self.after(50, self._poll_log)

    def _log(self, text: str, tag: str | None = "auto"):
        """Append a line to the log textbox with optional colour tag."""
        tw = self.log_box._textbox
        tw.configure(state="normal")

        line = text.rstrip() + "\n"

        if tag == "auto":
            # Pick a tag based on keywords in the message
            tag = None
            upper = text.upper()
            for key in LOG_COLORS:
                if key in upper:
                    tag = key
                    break

        if tag:
            tw.insert("end", line, tag)
        else:
            tw.insert("end", line)

        tw.configure(state="disabled")
        tw.see("end")   # auto-scroll

    def _clear_log(self):
        tw = self.log_box._textbox
        tw.configure(state="normal")
        tw.delete("1.0", "end")
        tw.configure(state="disabled")

    # ── State helpers ─────────────────────────────────────────────────────────

    def _set_running(self, running: bool):
        """Toggle UI elements when the worker starts/stops."""
        # Must run on the main thread — use `after(0, ...)` when called from worker
        def _apply():
            if running:
                self.run_btn.configure(text="⏳  Running…", state="disabled")
                self.progress.pack(fill="x", padx=16, pady=(0, 4))
                self.progress.start()
            else:
                self.progress.stop()
                self.progress.pack_forget()
                self.run_btn.configure(text="▶  Run Renamer", state="normal")
        self.after(0, _apply)

    def _parse_region(self):
        """Returns a (x1, y1, x2, y2) tuple or None if any field is empty."""
        try:
            vals = [float(v.get()) for v in self.region_vars]
            if all(0.0 <= v <= 1.0 for v in vals):
                return tuple(vals)
        except (ValueError, tk.TclError):
            pass
        return None

    @staticmethod
    def _safe_int(var, default: int) -> int:
        try:
            return int(var.get())
        except (ValueError, tk.TclError):
            return default


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()
