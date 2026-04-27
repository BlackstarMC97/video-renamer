import os
import subprocess
import re
import cv2
import numpy as np
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import argparse
from datetime import datetime, timedelta


def extract_bright_text(image_path, brightness_threshold=180, sat_max=80, region=None, skip_filter=False, debug=False, aggressive=False):
    """
    Isolates bright, low-saturation pixels from a frame (e.g. DVR timestamps).
    Works in HSV color space so brightness is separated from color,
    which is more robust than naive RGB white detection.

    Args:
        image_path (str): Path to the input image.
        brightness_threshold (int): Minimum V (brightness) value to keep (0-255).
                                    Lower = more tolerant, Higher = stricter.
        sat_max (int): Maximum S (saturation) value to keep (0-255).
                       Keeps near-white/grey text, rejects colored elements.
                       Raise this (e.g. 120) if your DVR uses yellow timestamps.
        region (tuple): Optional crop region as fractions of image dimensions,
                        in the form (x1, y1, x2, y2). Example: (0, 0.85, 0.5, 1.0)
                        targets the bottom-left quarter of the frame.
        skip_filter (bool): If True, skip brightness/saturation filtering (for --radio mode).
        debug (bool): If True, save intermediate images for debugging.
        aggressive (bool): If True, use more aggressive background rejection for white text.

    Returns:
        str: Path to the processed image ready for OCR.
    """
    img = cv2.imread(image_path)

    if img is None:
        print(f"  [WARNING] Could not load image for preprocessing: {image_path}")
        return image_path

    if debug:
        cv2.imwrite(image_path.replace(".jpg", "_debug_original.jpg"), img)

    if region:
        h, w = img.shape[:2]
        x1, y1, x2, y2 = region
        img = img[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]
        if debug:
            cv2.imwrite(image_path.replace(".jpg", "_debug_cropped.jpg"), img)
            print(f"  [DEBUG] Cropped to region {region} - size: {img.shape}")

    if skip_filter:
        processed_path = image_path.replace(".jpg", "_processed.jpg")
        cv2.imwrite(processed_path, img)
        if debug:
            cv2.imwrite(image_path.replace(".jpg", "_debug_final.jpg"), img)
        return processed_path

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 0, brightness_threshold])
    upper = np.array([255, sat_max, 255])
    mask = cv2.inRange(hsv, lower, upper)

    if debug:
        cv2.imwrite(image_path.replace(".jpg", "_debug_mask_before_morph.jpg"), mask)
        print(f"  [DEBUG] Mask has {np.sum(mask > 0)} white pixels")

    result = cv2.bitwise_and(img, img, mask=mask)
    result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    result_gray = cv2.equalizeHist(result_gray)

    if debug:
        cv2.imwrite(image_path.replace(".jpg", "_debug_result.jpg"), result)
        cv2.imwrite(image_path.replace(".jpg", "_debug_result_gray.jpg"), result_gray)

    processed_path = image_path.replace(".jpg", "_processed.jpg")
    cv2.imwrite(processed_path, result_gray)

    if debug:
        cv2.imwrite(image_path.replace(".jpg", "_debug_final.jpg"), result_gray)

    return processed_path


class VideoRenamer:
    def __init__(self, folder_path, lang=['en'], fallback_minutes=30,
                 forced_prefix=None, brightness_threshold=180, sat_max=80,
                 region=None, radio_mode=False, debug=False, aggressive=False,
                 frame_step=5, min_confidence=0.85):
        self.folder_path = folder_path
        self.fallback_minutes = fallback_minutes
        self.forced_prefix = forced_prefix
        self.brightness_threshold = brightness_threshold
        self.sat_max = sat_max
        self.region = region
        self.radio_mode = radio_mode
        self.debug = debug
        self.aggressive = aggressive
        self.frame_step = frame_step          # seconds between scanned frames
        self.min_confidence = min_confidence  # 0.0–1.0 confidence threshold

        self.ocr = ocr_predictor(pretrained=True)

        self.date_pattern = re.compile(r'(\d{4}[-/.]\d{2}[-/.]\d{2})|(\d{2}[-/.]\d{2}[-/.]\d{4})')
        self.time_pattern = re.compile(r'\d{2}\s*[:\-.]\s*\d{2}\s*[:\-.]\s*\d{2}')

    # ------------------------------------------------------------------
    # Frame extraction helpers
    # ------------------------------------------------------------------

    def extract_frame_at(self, video_path, offset_seconds, out_path):
        """Extracts a single frame at a given second offset into the video."""
        subprocess.run([
            'ffmpeg', '-y', '-ss', str(offset_seconds),
            '-i', video_path, '-frames:v', '1', '-q:v', '2', out_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.exists(out_path)

    def get_video_duration(self, video_path):
        """Returns video duration in seconds via ffprobe."""
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', video_path
        ], capture_output=True, text=True)
        try:
            return float(result.stdout.strip())
        except (ValueError, AttributeError):
            return None

    def extract_frames(self, video_path):
        """
        Extracts the first and last frame of a video using FFMPEG.
        Used as a fast path when multi-frame scanning is not needed.
        """
        first_frame = "first_frame.jpg"
        last_frame = "last_frame.jpg"

        subprocess.run([
            'ffmpeg', '-y', '-i', video_path, '-frames:v', '1', '-q:v', '2', first_frame
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        duration = self.get_video_duration(video_path)
        if duration:
            subprocess.run([
                'ffmpeg', '-y', '-ss', str(max(0, duration - 0.5)),
                '-i', video_path, '-frames:v', '1', '-q:v', '2', last_frame
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            last_frame = None

        return first_frame, last_frame, duration

    # ------------------------------------------------------------------
    # String / time helpers
    # ------------------------------------------------------------------

    def clean_string(self, s):
        """Sanitizes a string for use in a filename."""
        return re.sub(r'[^a-zA-Z0-9]', '', s)

    def normalize_time(self, raw_time):
        """Fixes OCR artifacts like '07 : 01:26' or '09:04.19' into HH-MM-SS."""
        if not raw_time:
            return None
        t = raw_time.replace(" ", "")
        t = re.sub(r'[:\.]', '-', t)
        parts = t.split('-')
        if len(parts) == 3:
            return "-".join(parts)
        return None

    def extract_datetime(self, text):
        """Extracts a date and a time from an OCR text string."""
        date_match = self.date_pattern.search(text)
        date = None
        time = None

        if date_match:
            date = date_match.group(0).replace('/', '-').replace('.', '-')
            remaining = text.replace(date_match.group(0), '')
        else:
            remaining = text

        time_match = self.time_pattern.search(remaining)
        if time_match:
            time = self.normalize_time(time_match.group(0))

        return date, time

    def fallback_time(self, base_time_str, add=True):
        """Computes a fallback time by adding or subtracting the configured fallback duration."""
        try:
            t = datetime.strptime(base_time_str, "%H-%M-%S")
            delta = timedelta(minutes=self.fallback_minutes)
            new_time = t + delta if add else t - delta
            return new_time.strftime("%H-%M-%S")
        except Exception:
            return None

    def adjust_time(self, time_str, delta_seconds):
        """
        Shifts a HH-MM-SS string by delta_seconds (can be negative).
        Used to back-calculate the real start time from a mid-video reading.
        """
        try:
            t = datetime.strptime(time_str, "%H-%M-%S")
            t += timedelta(seconds=delta_seconds)
            return t.strftime("%H-%M-%S")
        except Exception:
            return time_str

    # ------------------------------------------------------------------
    # OCR core
    # ------------------------------------------------------------------

    def get_info_from_image(self, image_path):
        """
        Runs preprocessing then OCR to extract date, time, prefix text,
        and an overall confidence score.

        Returns:
            tuple: (date_str, time_str, prefix_str, confidence_float)
                   confidence is the mean word confidence of all OCR words,
                   or 0.0 on failure.
        """
        if not image_path or not os.path.exists(image_path):
            return None, None, "", 0.0

        processed_path = extract_bright_text(
            image_path,
            brightness_threshold=self.brightness_threshold,
            sat_max=self.sat_max,
            region=self.region,
            skip_filter=self.radio_mode,
            debug=self.debug,
            aggressive=self.aggressive
        )

        print(f"\nOCR raw results for {image_path} (preprocessed: {processed_path}):")

        try:
            doc = DocumentFile.from_images(processed_path)
            result = self.ocr(doc)

            date_found = None
            time_found = None
            xxx_parts = []
            all_confidences = []

            lines = []
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        text = " ".join([word.value for word in line.words])
                        # Collect per-word confidences
                        for word in line.words:
                            all_confidences.append(word.confidence)

                        if text.strip():
                            print(f"  -> '{text}'")
                            lines.append(text)

                            d, tm = self.extract_datetime(text)
                            if d and not date_found:
                                date_found = d
                            if tm and not time_found:
                                time_found = tm

                            if not d and not tm:
                                cleaned = self.clean_string(text)
                                if cleaned and len(cleaned) > 2 and not cleaned.isdigit():
                                    xxx_parts.append(cleaned)

            # Word-level pass for finer datetime detection
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            d, tm = self.extract_datetime(word.value)
                            if d and not date_found:
                                date_found = d
                            if tm and not time_found:
                                time_found = tm

            # Full-text fallback
            all_text = " ".join(lines)
            if not date_found and not time_found and all_text:
                d, tm = self.extract_datetime(all_text)
                if d and not date_found:
                    date_found = d
                if tm and not time_found:
                    time_found = tm

            xxx = xxx_parts[0] if xxx_parts else ""

            # Mean confidence over all detected words; 0 if nothing was read
            confidence = float(np.mean(all_confidences)) if all_confidences else 0.0

            print(f"  => Extracted: date={date_found}, time={time_found}, "
                  f"prefix={xxx}, confidence={confidence:.3f}")

        except Exception as e:
            print(f"  [ERROR] Doctr OCR failed: {e}")
            date_found = None
            time_found = None
            xxx = ""
            confidence = 0.0

        if processed_path != image_path and os.path.exists(processed_path):
            os.remove(processed_path)

        return date_found, time_found, xxx, confidence

    # ------------------------------------------------------------------
    # Multi-frame scanning
    # ------------------------------------------------------------------

    def scan_for_timestamp(self, video_path, duration, from_start=True):
        """
        Scans frames every `frame_step` seconds until a reading whose
        confidence meets `min_confidence` is found.

        Args:
            video_path (str): Path to the video file.
            duration (float): Total video duration in seconds.
            from_start (bool): True  → scan forward  (finding start timestamp).
                               False → scan backward (finding end timestamp).

        Returns:
            tuple: (date_str, time_str, prefix_str, frame_offset_seconds)
                   frame_offset_seconds is the position inside the video where
                   the confident reading was made — used to back-calculate the
                   real start / end time of the recording.
                   Returns (None, None, "", None) if no confident frame found.
        """
        if duration is None:
            return None, None, "", None

        # Build the list of offsets to probe
        if from_start:
            # 0, step, 2*step, … up to half the video (no point going further)
            offsets = list(np.arange(0, duration / 2, self.frame_step))
        else:
            # duration, duration-step, … down to half the video
            offsets = list(np.arange(duration, duration / 2, -self.frame_step))

        direction = "forward" if from_start else "backward"
        print(f"\n  [SCAN] Scanning {direction} through {len(offsets)} frame(s) "
              f"(step={self.frame_step}s, min_confidence={self.min_confidence})")

        tmp_frame = "scan_frame.jpg"

        for offset in offsets:
            offset = max(0.0, min(offset, duration))
            print(f"  [SCAN] Trying offset {offset:.1f}s …", end="")

            ok = self.extract_frame_at(video_path, offset, tmp_frame)
            if not ok:
                print(" (frame extraction failed, skipping)")
                continue

            date, time, prefix, conf = self.get_info_from_image(tmp_frame)

            # We require at least the time to be present for a useful reading
            if time and conf >= self.min_confidence:
                print(f"  [SCAN] ✓ Confident reading at {offset:.1f}s "
                      f"(confidence={conf:.3f}): date={date}, time={time}")
                if os.path.exists(tmp_frame):
                    os.remove(tmp_frame)
                return date, time, prefix, offset

            print(f" confidence={conf:.3f} — not good enough, continuing…")

        if os.path.exists(tmp_frame):
            os.remove(tmp_frame)

        print("  [SCAN] No confident reading found during scan.")
        return None, None, "", None

    # ------------------------------------------------------------------
    # Main processing loop
    # ------------------------------------------------------------------

    def process_folder(self):
        """Iterates over the folder and renames video files based on OCR results."""
        files = [f for f in os.listdir(self.folder_path)
                 if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.264'))]

        if not files:
            print("No video files found.")
            return

        for idx, filename in enumerate(files, start=1):
            video_path = os.path.join(self.folder_path, filename)
            print(f"\n{'='*60}")
            print(f"Processing ({idx}/{len(files)}): {filename}")
            print('='*60)

            # ── Step 1: fast path — try first & last frames ──────────────
            first_img, last_img, duration = self.extract_frames(video_path)

            date_start, time_start, ocr_prefix, conf_start = self.get_info_from_image(first_img)
            _, time_end, _, conf_end = self.get_info_from_image(last_img)

            # ── Step 2: multi-frame scan when fast-path confidence is low ─
            #
            # If the first frame reading is not confident enough, scan
            # forward until we find a clear frame, then subtract the
            # in-video offset to recover the real recording start time.
            #
            start_offset = 0.0   # seconds into the video of the winning frame
            end_offset   = duration if duration else 0.0

            if (not time_start or conf_start < self.min_confidence) and duration:
                print(f"\n  [INFO] First-frame confidence too low "
                      f"({conf_start:.3f} < {self.min_confidence}). "
                      f"Activating multi-frame scan for START timestamp…")

                scan_date, scan_time, scan_prefix, start_offset = \
                    self.scan_for_timestamp(video_path, duration, from_start=True)

                if scan_time:
                    # Back-calculate: real_start = timestamp_at_offset − offset
                    real_start = self.adjust_time(scan_time, -start_offset)
                    print(f"  [SCAN] Timestamp at offset {start_offset:.1f}s = {scan_time}  "
                          f"→  estimated recording start = {real_start}")
                    time_start = real_start
                    if scan_date and not date_start:
                        date_start = scan_date
                    if scan_prefix and not ocr_prefix:
                        ocr_prefix = scan_prefix

            if (not time_end or conf_end < self.min_confidence) and duration:
                print(f"\n  [INFO] Last-frame confidence too low "
                      f"({conf_end:.3f} < {self.min_confidence}). "
                      f"Activating multi-frame scan for END timestamp…")

                _, scan_time_end, _, end_offset = \
                    self.scan_for_timestamp(video_path, duration, from_start=False)

                if scan_time_end:
                    # Forward-calculate: real_end = timestamp_at_offset + remaining
                    seconds_remaining = duration - end_offset
                    real_end = self.adjust_time(scan_time_end, seconds_remaining)
                    print(f"  [SCAN] Timestamp at offset {end_offset:.1f}s = {scan_time_end}  "
                          f"→  estimated recording end = {real_end}")
                    time_end = real_end

            # ── Step 3: fallback when only one side was found ────────────
            if time_start and not time_end:
                print("  -> Applying fallback: end time = start + X minutes")
                time_end = self.fallback_time(time_start, add=True)

            elif time_end and not time_start:
                print("  -> Applying fallback: start time = end - X minutes")
                time_start = self.fallback_time(time_end, add=False)

            # ── Step 4: rename ───────────────────────────────────────────
            if date_start and time_start and time_end:
                if self.forced_prefix:
                    prefix = self.forced_prefix
                elif ocr_prefix:
                    prefix = ocr_prefix
                else:
                    prefix = str(idx)

                extension = os.path.splitext(filename)[1]
                new_name = f"{prefix}_{date_start}_{time_start}_{time_end}{extension}"
                new_path = os.path.join(self.folder_path, new_name)

                try:
                    os.rename(video_path, new_path)
                    print(f"\nSUCCESS: {filename} -> {new_name}")
                except Exception as e:
                    print(f"\nERROR renaming {filename}: {e}")
            else:
                print(f"\nFAILED: Incomplete info for {filename}")
                if not date_start: print("  - Date not found")
                if not time_start: print("  - Start time not found")
                if not time_end:   print("  - End time not found")

            # Clean up extracted frames
            for img in ["first_frame.jpg", "last_frame.jpg"]:
                if os.path.exists(img):
                    os.remove(img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch rename DVR video files using OCR.")

    parser.add_argument("folder",
                        help="Path to the folder containing video files.")

    parser.add_argument("--fallback", type=int, default=30,
                        help="Minutes to add/subtract when one timestamp is missing (default: 30).")

    parser.add_argument("--prefix", type=str,
                        help="Force a fixed filename prefix instead of OCR detection.")

    parser.add_argument("--brightness", type=int, default=180,
                        help="Brightness threshold for timestamp extraction (0-255). "
                             "Lower = more tolerant, higher = stricter (default: 180). "
                             "Try 160 for faint text, 200 for cleaner feeds.")

    parser.add_argument("--sat-max", type=int, default=80,
                        help="Maximum saturation for timestamp extraction (0-255). "
                             "Increase to ~120 if your DVR uses yellow timestamps (default: 80).")

    parser.add_argument("--region", type=float, nargs=4,
                        metavar=('X1', 'Y1', 'X2', 'Y2'),
                        help="Crop region for OCR as fractions of image size (0.0-1.0). "
                             "Example: --region 0 0.85 0.5 1.0 targets the bottom-left corner.")

    parser.add_argument("--radio", action="store_true",
                        help="Skip brightness/saturation filtering (for radio recordings).")

    parser.add_argument("--debug", action="store_true",
                        help="Save debug images to see what the OCR is processing.")

    parser.add_argument("--aggressive", action="store_true",
                        help="Use aggressive background rejection for white text on TV recordings.")

    # ── New multi-frame scan arguments ───────────────────────────────────
    parser.add_argument("--frame-step", type=int, default=5,
                        help="Seconds between frames when scanning for a confident timestamp "
                             "(default: 5). Lower = slower but more thorough.")

    parser.add_argument("--min-confidence", type=float, default=0.85,
                        help="Minimum mean OCR confidence (0.0–1.0) required to accept a "
                             "timestamp reading without scanning further (default: 0.85).")

    args = parser.parse_args()

    if os.path.isdir(args.folder):
        renamer = VideoRenamer(
            args.folder,
            fallback_minutes=args.fallback,
            forced_prefix=args.prefix,
            brightness_threshold=args.brightness,
            sat_max=args.sat_max,
            region=tuple(args.region) if args.region else None,
            radio_mode=args.radio,
            debug=args.debug,
            aggressive=args.aggressive,
            frame_step=args.frame_step,
            min_confidence=args.min_confidence,
        )
        renamer.process_folder()
    else:
        print(f"Error: folder '{args.folder}' does not exist.")