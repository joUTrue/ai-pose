import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

import requests


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.recog import RecogParams, analyze_video_to_timeline, load_models_once


OUTPUT_DIR = ROOT_DIR / "test" / "output"


def download_to_temp(url: str) -> str:
    suffix = Path(url.split("?")[0]).suffix or ".mp4"
    temp_path = Path(tempfile.gettempdir()) / f"pose_test_{uuid.uuid4().hex}{suffix}"

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(temp_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

    return str(temp_path)


def main() -> None:
    video_url = input("S3 video URL: ").strip()
    if not video_url:
        raise ValueError("video URL is required")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[TEST] Loading pose models...")
    load_models_once()

    temp_video_path = None
    try:
        print("[TEST] Downloading video...")
        temp_video_path = download_to_temp(video_url)

        print("[TEST] Running pose analysis...")
        result = analyze_video_to_timeline(
            temp_video_path,
            params=RecogParams(),
        )

        output_path = OUTPUT_DIR / f"pose_result_{uuid.uuid4().hex}.json"
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)

        print(f"[TEST] Analysis complete. Output saved to: {output_path}")
        print(json.dumps(result["video"], ensure_ascii=False, indent=2))
    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)


if __name__ == "__main__":
    main()
