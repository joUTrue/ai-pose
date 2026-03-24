# app/main.py
import os
import uuid
import shutil
import tempfile
from typing import Optional

import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from app.recog import analyze_video_to_timeline, load_models_once, RecogParams

app = FastAPI(title="Pose/Gesture Timeline API")


class InferRequest(BaseModel):
    video_url: str


@app.on_event("startup")
def _startup():
    # 서버 시작 시 모델 1회 로딩
    load_models_once()


def _download_to_temp(url: str) -> str:
    tmp_dir = tempfile.gettempdir()
    filename = f"dl_{uuid.uuid4().hex}.mp4"
    out_path = os.path.join(tmp_dir, filename)

    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
    except Exception as e:
        # 다운로드 실패 시 파일 정리
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except Exception:
                pass
        raise RuntimeError(f"Download failed: {e}")

    return out_path


def _run_pose_analysis(video_url: str, frame_interval: int = 1, window_sec: float = 2.0, det_score_thr: float = 0.5, score_threshold: float = 0.30):
    tmp_path = None
    try:
        tmp_path = _download_to_temp(video_url)

        params = RecogParams(
            frame_interval=frame_interval,
            window_sec=window_sec,
            det_score_thr=det_score_thr,
            score_threshold=score_threshold,
        )

        result = analyze_video_to_timeline(tmp_path, params=params)

        return {
            "ok": True,
            "video": result["video"],
            "timeline": result["timeline"],
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@app.post("/infer")
def infer(req: InferRequest):
    try:
        return _run_pose_analysis(video_url=req.video_url)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")


@app.post("/recognize")
def recognize(
    file: Optional[UploadFile] = File(default=None),
    video_url: Optional[str] = Form(default=None),
    frame_interval: int = Form(default=1),
    window_sec: float = Form(default=2.0),
    det_score_thr: float = Form(default=0.5),
    score_threshold: float = Form(default=0.30),
):
    if file is None and not video_url:
        raise HTTPException(status_code=400, detail="Provide either file or video_url.")

    tmp_path = None
    try:
        # 1) 입력 준비
        if file is not None:
            suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
            tmp_path = os.path.join(tempfile.gettempdir(), f"up_{uuid.uuid4().hex}{suffix}")
            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            params = RecogParams(
                frame_interval=frame_interval,
                window_sec=window_sec,
                det_score_thr=det_score_thr,
                score_threshold=score_threshold,
            )
            result = analyze_video_to_timeline(tmp_path, params=params)
            return {
                "ok": True,
                "video": result["video"],
                "timeline": result["timeline"],
            }

        return _run_pose_analysis(
            video_url=video_url,
            frame_interval=frame_interval,
            window_sec=window_sec,
            det_score_thr=det_score_thr,
            score_threshold=score_threshold,
        )

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
