#!/usr/bin/env python3
"""Download the openly-licensed sample clip used by the demo.

Source: Wikimedia Commons, "Team sailing match-racing start at NHYC"
        by Don Ramey Logan - CC BY-SA 4.0
        https://commons.wikimedia.org/wiki/File:Team_sailing_match-racing_start_at_NHYC_by_Don_Ramey_Logan.webm

License: CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/).
Attribution required wherever frames or derived media from this clip appear
(demo GIF, stills, README). Derived media in this repo inherits BY-SA.

Downloads the 1080p transcode (not the 500+ MB 4K original) and converts to
mp4 for OpenCV. Requires ffmpeg on PATH.

    python scripts/download_sample.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import urllib.request

NAME = "Team_sailing_match-racing_start_at_NHYC_by_Don_Ramey_Logan.webm"
URL = ("https://upload.wikimedia.org/wikipedia/commons/transcoded/9/9d/"
       f"{NAME}/{NAME}.1080p.vp9.webm")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "samples")
WEBM = os.path.join(OUT_DIR, "sample_open.webm")
MP4 = os.path.join(OUT_DIR, "sample_open.mp4")


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(WEBM):
        print(f"downloading {URL}")
        # Wikimedia rejects requests without a descriptive User-Agent
        req = urllib.request.Request(URL, headers={
            "User-Agent": "marine-vision-tracking/0.1 (sample downloader)"})
        with urllib.request.urlopen(req) as r, open(WEBM, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        print(f"  {os.path.getsize(WEBM)/1e6:.1f} MB")
    if not os.path.exists(MP4):
        print("converting to mp4 (h264, 25 fps)")
        r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", WEBM,
                            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                            "-vf", "scale=1280:720", "-r", "25", "-an", MP4])
        if r.returncode != 0:
            print("ffmpeg failed - is it on PATH?")
            return 1
    print(f"ready: {MP4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
