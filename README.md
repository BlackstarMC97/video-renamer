# 📹 DVR Video Renamer

Batch-rename DVR and TV recordings by reading the on-screen timestamp with OCR — no manual renaming, no metadata guessing.

It extracts frames from each video, isolates the timestamp overlay using HSV filtering, runs OCR via [doctr](https://github.com/mindee/doctr), and renames the file to a clean `PREFIX_DATE_STARTTIME_ENDTIME.ext` format.

When the first or last frame is unreadable (busy scene, bad lighting), it automatically scans through the video frame by frame until it finds a confident reading, then does the math to recover the real start/end time.

---

## Features

- Works on MP4, AVI, MOV, MKV, .264
- HSV-based timestamp isolation (handles white, grey, and yellow overlays)
- Multi-frame scanning with confidence scoring — finds the clearest frame automatically
- Back-calculates real start/end time from any mid-video reading
- Desktop GUI (CustomTkinter) — no terminal needed for end users
- Fully self-contained `.exe` build via PyInstaller

---

## Project structure

```
video-renamer/
├── video_renamer.py       # Core logic (OCR, frame extraction, renaming)
├── video_renamer_gui.py   # Desktop GUI (CustomTkinter)
├── ffprobe.exe            # Required for Windows builds (see below)
└── README.md
```

---

## Requirements

- Python 3.10 or 3.11 (3.12 not yet tested with doctr)
- FFmpeg available on PATH **or** bundled via `imageio-ffmpeg` (handled automatically)
- ffprobe binary (only needed for standalone `.exe` builds — see Build section)

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/yourname/video-renamer.git
cd video-renamer
```

### 2. Install Python dependencies

> **Windows users:** install the CPU-only version of PyTorch first to avoid a multi-GB download:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Then install the rest:

```bash
pip install doctr[torch] opencv-python customtkinter imageio-ffmpeg
```

### 3. Run the GUI

```bash
python video_renamer_gui.py
```

Or use the CLI directly:

```bash
python video_renamer.py /path/to/videos
```

---

## GUI usage

1. Click **Browse…** and select your folder of recordings
2. Adjust settings if needed (defaults work for most DVR feeds)
3. Click **▶ Run Renamer**
4. Watch the live log — renamed files appear as `SUCCESS` lines

### Settings explained

| Setting | Default | What it does |
|---|---|---|
| Forced prefix | `0` | Prefix for every filename. Leave blank to auto-detect from OCR. |
| Fallback minutes | `30` | Added/subtracted when only one timestamp is found |
| Frame step | `5` | Seconds between frames when scanning for a clear timestamp |
| Min confidence | `0.85` | OCR confidence required to accept a reading (0–1) |
| Brightness threshold | `180` | Minimum brightness to keep during filtering. Lower = more tolerant |
| Sat max | `80` | Max saturation to keep. Raise to ~120 for yellow timestamps |
| Region | `0 0 1 0.25` | Crop zone for OCR as fractions of image size (x1 y1 x2 y2) |
| Radio mode | off | Skips brightness/saturation filtering for simple backgrounds |
| Aggressive | off | Stronger background rejection for white text on busy scenes |
| Debug images | off | Saves intermediate preprocessing images next to each frame |

---

## CLI usage

```bash
python video_renamer.py /path/to/videos [options]
```

| Option | Default | Description |
|---|---|---|
| `--prefix TEXT` | auto | Force a filename prefix |
| `--fallback N` | `30` | Fallback minutes |
| `--brightness N` | `180` | Brightness threshold (0–255) |
| `--sat-max N` | `80` | Max saturation (0–255) |
| `--region x1 y1 x2 y2` | none | Crop region as 0–1 fractions |
| `--frame-step N` | `5` | Seconds between scanned frames |
| `--min-confidence F` | `0.85` | Minimum OCR confidence |
| `--radio` | off | Skip HSV filtering |
| `--aggressive` | off | Aggressive background rejection |
| `--debug` | off | Save debug images |

**Example — bottom-left timestamp, scan every 3 seconds:**
```bash
python video_renamer.py /recordings --region 0 0.75 0.5 1.0 --frame-step 3
```

---

## Building a standalone `.exe` (Windows)

Users who receive the `.exe` need no Python or dependencies installed.

### 1. Install PyInstaller

```bash
pip install pyinstaller
```

### 2. Download ffprobe

Download `ffprobe.exe` for Windows from **https://ffbinaries.com/downloads** and place it in the project folder next to `video_renamer_gui.py`.

### 3. Build

```bash
python -m PyInstaller video_renamer_gui.py ^
  --onedir --windowed ^
  --name "DVR Renamer" ^
  --add-binary "ffprobe.exe;." ^
  --hidden-import="cv2"
```

> Use `--onedir` during development (fast rebuilds). Switch to `--onefile` for the final release.

The output is in `dist/DVR Renamer/`. Share that entire folder, or zip it up.

### Subsequent builds

After the first run, PyInstaller generates `DVR Renamer.spec`. Rebuild faster with:

```bash
python -m PyInstaller "DVR Renamer.spec"
```

---

## Output filename format

```
PREFIX_YYYY-MM-DD_HH-MM-SS_HH-MM-SS.ext
```

Example:
```
CH1_2024-01-15_14-31-25_15-01-25.mp4
```

---

## How the multi-frame scan works

1. First, the tool tries the very first and last frame of the video
2. If the OCR confidence is below `--min-confidence`, it steps through the video every `--frame-step` seconds
3. Once a confident timestamp is found at offset `T`, it computes:
   - `real_start = timestamp − T`
   - `real_end = timestamp + (duration − T)`
4. If only one side is found, the other is estimated using `--fallback`

---

## Troubleshooting

**Timestamps not detected**
- Try lowering `--brightness` (e.g. `160`) or raising `--sat-max` (e.g. `120` for yellow text)
- Use `--debug` to save intermediate images and inspect what the OCR sees
- Set `--region` to crop tightly around where your DVR puts the timestamp

**Wrong times**
- Lower `--min-confidence` slightly (e.g. `0.75`) if the scan gives up too early
- Lower `--frame-step` so it samples more frames

**Flash of console window on Windows**
- This is fixed in the current version via `CREATE_NO_WINDOW`. Make sure you're on the latest code.

---

## License

MIT
