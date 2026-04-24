import os
import subprocess
import re
import cv2
import numpy as np
import easyocr
import argparse
from datetime import datetime, timedelta


def extract_bright_text(image_path, brightness_threshold=180, sat_max=80, region=None, skip_filter=False):
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
                        Use this to isolate the known timestamp area and reduce noise.
        skip_filter (bool): If True, skip brightness/saturation filtering (for --radio mode).

    Returns:
        str: Path to the processed (binary mask) image ready for OCR.
    """
    img = cv2.imread(image_path)

    # Graceful fallback: if the image can't be loaded, return the original path
    # so OCR can still attempt to run on the raw frame
    if img is None:
        print(f"  [WARNING] Could not load image for preprocessing: {image_path}")
        return image_path

    # Optional crop to a known timestamp region before filtering.
    # Coordinates are given as fractions (0.0-1.0) of the image dimensions.
    if region:
        h, w = img.shape[:2]
        x1, y1, x2, y2 = region
        img = img[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]

    # If skip_filter is True (--radio mode), save the cropped image directly
    if skip_filter:
        processed_path = image_path.replace(".jpg", "_processed.jpg")
        cv2.imwrite(processed_path, img)
        return processed_path

    # Convert BGR to HSV so we can filter on brightness (V channel) independently
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Keep only very bright, low-saturation pixels (white/grey text)
    # H (hue): full range 0-255, we don't care about the color
    # S (saturation): capped at sat_max to exclude colored overlays and logos
    # V (value/brightness): must exceed brightness_threshold
    lower = np.array([0, 0, brightness_threshold])
    upper = np.array([255, sat_max, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Morphological close: fills small gaps inside characters
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Dilation: reconnects broken digit segments caused by video compression
    # e.g. prevents "07:01:26" from being read as "07:01:2_"
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Save the processed mask next to the original frame
    processed_path = image_path.replace(".jpg", "_processed.jpg")
    cv2.imwrite(processed_path, mask)

    return processed_path


class VideoRenamer:
    def __init__(self, folder_path, lang=['en'], fallback_minutes=30,
                 forced_prefix=None, brightness_threshold=180, sat_max=80, region=None, radio_mode=False):
        self.folder_path = folder_path
        self.fallback_minutes = fallback_minutes
        self.forced_prefix = forced_prefix
        self.brightness_threshold = brightness_threshold
        self.sat_max = sat_max
        self.region = region
        self.radio_mode = radio_mode

        # Initialize OCR reader (CPU only for Plesk/server compatibility)
        self.reader = easyocr.Reader(lang, gpu=False)

        # Regex patterns
        # Date: YYYY-MM-DD or DD-MM-YYYY (accepts - / . as separators)
        self.date_pattern = re.compile(r'(\d{4}[-/.]\d{2}[-/.]\d{2})|(\d{2}[-/.]\d{2}[-/.]\d{4})')
        # Time: flexible pattern to handle messy OCR output
        self.time_pattern = re.compile(r'\d{2}\s*[:\-.]\s*\d{2}\s*[:\-.]\s*\d{2}')

    def extract_frames(self, video_path):
        """Extracts the first and last frame of a video using FFMPEG."""
        first_frame = "first_frame.jpg"
        last_frame = "last_frame.jpg"

        # Extract first frame
        subprocess.run([
            'ffmpeg', '-y', '-i', video_path, '-frames:v', '1', '-q:v', '2', first_frame
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Get video duration via ffprobe, then extract last frame
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', video_path
        ], capture_output=True, text=True)

        try:
            duration = float(result.stdout.strip())
            subprocess.run([
                'ffmpeg', '-y', '-ss', str(max(0, duration - 0.5)),
                '-i', video_path, '-frames:v', '1', '-q:v', '2', last_frame
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            last_frame = None

        return first_frame, last_frame

    def clean_string(self, s):
        """Sanitizes a string for use in a filename."""
        return re.sub(r'[^a-zA-Z0-9]', '', s)

    def normalize_time(self, raw_time):
        """Fixes OCR artifacts like '07 : 01:26' or '09:04.19' into HH-MM-SS."""
        if not raw_time:
            return None

        # Remove all spaces, then normalize all separators to '-'
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
            # Normalize date separators to '-'
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
        except:
            return None

    def get_info_from_image(self, image_path):
        """
        Runs preprocessing then OCR to extract date, time, and optional prefix text.
        Preprocessing isolates bright timestamp pixels before passing to EasyOCR,
        which significantly improves accuracy on TV and DVR recordings.
        """
        if not image_path or not os.path.exists(image_path):
            return None, None, ""

        # Apply brightness/saturation filter before OCR.
        # This step is the key improvement for TV recordings:
        # it keeps only bright, low-saturation pixels (timestamps)
        # and discards colored backgrounds, logos, and scene content.
        processed_path = extract_bright_text(
            image_path,
            brightness_threshold=self.brightness_threshold,
            sat_max=self.sat_max,
            region=self.region,
            skip_filter=self.radio_mode  # Pass the radio_mode flag
        )

        print(f"\nOCR raw results for {image_path} (preprocessed: {processed_path}):")

        # Use allowlist to restrict OCR to timestamp-relevant characters only.
        # This prevents the model from misreading graphic elements as letters.
        results = self.reader.readtext(processed_path, allowlist='0123456789:-.')

        date_found = None
        time_found = None
        xxx_parts = []

        for res in results:
            t = res[1]
            conf = res[2]

            print(f"  -> '{t}' (confidence: {conf:.2f})")

            d, tm = self.extract_datetime(t)

            if d and not date_found:
                date_found = d

            if tm and not time_found:
                time_found = tm

            # Collect non-date/time text as potential filename prefix
            if not d and not tm:
                cleaned = self.clean_string(t)
                if cleaned and len(cleaned) > 2 and not cleaned.isdigit():
                    xxx_parts.append(cleaned)

        xxx = xxx_parts[0] if xxx_parts else ""

        print(f"  => Extracted: date={date_found}, time={time_found}, prefix={xxx}")

        # Clean up the processed mask image
        if processed_path != image_path and os.path.exists(processed_path):
            os.remove(processed_path)

        return date_found, time_found, xxx

    def process_folder(self):
        """Iterates over the folder and renames video files based on OCR results."""
        files = [f for f in os.listdir(self.folder_path)
                 if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.264'))]

        if not files:
            print("No video files found.")
            return

        for idx, filename in enumerate(files, start=1):
            video_path = os.path.join(self.folder_path, filename)
            print(f"Processing: {filename}...")

            first_img, last_img = self.extract_frames(video_path)

            date_start, time_start, ocr_prefix = self.get_info_from_image(first_img)
            _, time_end, _ = self.get_info_from_image(last_img)

            # Fallback logic: if only one timestamp was found, estimate the other
            if time_start and not time_end:
                print("  -> Applying fallback: end time = start + X minutes")
                time_end = self.fallback_time(time_start, add=True)

            elif time_end and not time_start:
                print("  -> Applying fallback: start time = end - X minutes")
                time_start = self.fallback_time(time_end, add=False)

            if date_start and time_start and time_end:
                # Prefix priority: CLI argument > OCR-detected text > file index
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
                    print(f"SUCCESS: {filename} -> {new_name}")
                except Exception as e:
                    print(f"ERROR renaming {filename}: {e}")
            else:
                print(f"FAILED: Incomplete info for {filename}")
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
                             "Example: --region 0 0.85 0.5 1.0 targets the bottom-left corner. "
                             "Highly recommended for TV recordings to reduce false positives.")

    parser.add_argument("--radio", action="store_true",
                        help="Skip brightness/saturation filtering (for radio recordings with simple backgrounds).")

    args = parser.parse_args()

    if os.path.isdir(args.folder):
        renamer = VideoRenamer(
            args.folder,
            fallback_minutes=args.fallback,
            forced_prefix=args.prefix,
            brightness_threshold=args.brightness,
            sat_max=args.sat_max,
            region=tuple(args.region) if args.region else None,
            radio_mode=args.radio
        )
        renamer.process_folder()
    else:
        print(f"Error: folder '{args.folder}' does not exist.")