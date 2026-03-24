# recog.py
import os
import cv2
import time
import torch
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import threading

import mmengine
from mmengine.registry import init_default_scope

from mmdet.utils import register_all_modules as register_mmdet
from mmpose.utils import register_all_modules as register_mmpose
from mmaction.utils import register_all_modules as register_mmaction

from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_model, inference_topdown
from mmaction.apis import init_recognizer, inference_recognizer

import warnings
import logging
from mmengine.logging import MMLogger

# -------------------------
# log/warning minimum
# -------------------------
warnings.filterwarnings("ignore")
for name in ["mmengine", "mmdet", "mmpose", "mmaction"]:
    MMLogger.get_instance(name).setLevel(logging.ERROR)

# -------------------------
# global setup (existing code preserved)
# -------------------------
register_mmdet()
register_mmpose()
register_mmaction()

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

DET_CONFIG = "config/detection/yolox/yolox_s_8xb8-300e_coco.py"
DET_CHECKPOINT = "config/detection/yolox/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth"

POSE_CONFIG = "config/pose/td-hm_hrnet-w48_dark-8xb32-210e_coco-384x288.py"
POSE_CHECKPOINT = "config/pose/td-hm_hrnet-w48_dark-8xb32-210e_coco-384x288-39c3c381_20220916.pth"

STGCN_CONFIG = "config/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_forPresentation_filtered.py"
STGCN_CHECKPOINT = "checkpoints/20260312_1806.pth"

LABEL_NAMES = [
    "Touching head",    #B01
    "Touching face",    #B02
    "Touching body",    #B03
    "Touching hand",     #B04
    #"Shaking head",    #B05 (excluded)
    "Swaying head",     #B06
    "Bowing",     #B07
    "Holding back",       #B08
    #"Swaying hand",     #B09 (excluded)
    "Swaying body",     #B10
    "Slanting",     #B11
    #"Skewing"     #B12 (excluded)
    ]

SKELETON_LINKS = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6)
]

# MMEngine default scope can be shared globally, so concurrent requests need a lock.
_SCOPE_LOCK = threading.Lock()

# -------------------------
# model globals (load once at startup)
# -------------------------
_det_model = None
_pose_model = None
_recognizer = None


def load_models_once() -> None:
    """Load models once during FastAPI startup."""
    global _det_model, _pose_model, _recognizer
    if _det_model is not None:
        return
    print(f"[MODEL] Loading models on {DEVICE}...", flush=True)
    
    with _SCOPE_LOCK:
        detcfg = mmengine.Config.fromfile(DET_CONFIG)
        detcfg.model.type = f"mmdet.{detcfg.model.type}"
        detcfg.default_scope = "mmdet"
        init_default_scope("mmdet")
        _det_model = init_detector(detcfg, DET_CHECKPOINT, device=DEVICE)

    with _SCOPE_LOCK:
        posecfg = mmengine.Config.fromfile(POSE_CONFIG)
        posecfg.default_scope = "mmpose"
        init_default_scope("mmpose")
        _pose_model = init_model(posecfg, POSE_CHECKPOINT, device=DEVICE)

    with _SCOPE_LOCK:
        init_default_scope("mmaction")
        _recognizer = init_recognizer(STGCN_CONFIG, STGCN_CHECKPOINT, device=DEVICE)


@dataclass
class RecogParams:
    frame_interval: int = 5
    det_score_thr: float = 0.5
    window_sec: float = 2.0
    window_size: int = 10
    conf_scale: float = 2500.0
    topk: int = 1
    score_threshold: float = 0.75
    merge_gap_sec: float = 1.00
    min_event_sec: float = 0.50


def _topk_from_probs(probs: np.ndarray, topk: int) -> List[Tuple[str, float]]:
    idxs = np.argsort(probs)[-topk:][::-1]
    return [(LABEL_NAMES[i], float(probs[i])) for i in idxs]


def _render_progress(current_frame: int, total_frames: int, started_at: float) -> str:
    percent = min(100, int((current_frame / max(1, total_frames)) * 100))
    width = 24
    filled = int(width * percent / 100)
    elapsed = time.time() - started_at
    eta = 0.0
    if current_frame > 0 and total_frames > current_frame:
        eta = (elapsed / current_frame) * (total_frames - current_frame)
    bar = "#" * filled + "." * (width - filled)
    return (
        f"[POSE] |{bar}| {percent:3d}% "
        f"({current_frame}/{total_frames}) elapsed={elapsed:.1f}s eta={eta:.1f}s"
    )


def _postprocess_timeline(
    per_frame: List[Dict[str, Any]],
    fps: float,
    params: RecogParams
) -> List[Dict[str, Any]]:
    if not per_frame:
        return []

    cleaned = []
    for p in per_frame:
        if p["score"] >= params.score_threshold:
            cleaned.append(p)
        else:
            cleaned.append({"t": p["t"], "label": None, "score": p["score"]})

    segments = []
    cur = None

    for p in cleaned:
        lbl = p["label"]
        t = p["t"]
        s = p["score"]

        if lbl is None:
            if cur is not None:
                cur["end"] = t
                segments.append(cur)
                cur = None
            continue

        if cur is None:
            cur = {
                "label": lbl,
                "start": t,
                "end": t,
                "sum_score": s,
                "max_score": s,
                "frames": 1
            }
        else:
            if lbl == cur["label"]:
                cur["end"] = t
                cur["sum_score"] += s
                cur["max_score"] = max(cur["max_score"], s)
                cur["frames"] += 1
            else:
                cur["end"] = t
                segments.append(cur)
                cur = {
                    "label": lbl,
                    "start": t,
                    "end": t,
                    "sum_score": s,
                    "max_score": s,
                    "frames": 1
                }

    if cur is not None:
        segments.append(cur)

    merged = []
    for seg in segments:
        if not merged:
            merged.append(seg)
            continue
        prev = merged[-1]
        gap = seg["start"] - prev["end"]
        if seg["label"] == prev["label"] and gap <= params.merge_gap_sec:
            prev["end"] = seg["end"]
            prev["sum_score"] += seg["sum_score"]
            prev["max_score"] = max(prev["max_score"], seg["max_score"])
            prev["frames"] += seg["frames"]
        else:
            merged.append(seg)

    out = []
    for seg in merged:
        dur = float(seg["end"] - seg["start"])
        if dur < params.min_event_sec:
            continue
        out.append({
            "label": seg["label"],
            "start": round(float(seg["start"]), 3),
            "end": round(float(seg["end"]), 3),
            "duration": round(dur, 3),
            "avg_score": round(float(seg["sum_score"] / max(1, seg["frames"])), 4),
            "max_score": round(float(seg["max_score"]), 4),
            "frames": int(seg["frames"])
        })
    return out


def analyze_video_to_timeline(
    video_path: str,
    params: Optional[RecogParams] = None
) -> Dict[str, Any]:
    if params is None:
        params = RecogParams()

    load_models_once()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = (total_frames / fps) if total_frames > 0 else None

    # window_size = max(1, int(params.window_sec * fps))
    skeleton_buffer = deque(maxlen=params.window_size)

    cnt = 0

    per_frame_preds: List[Dict[str, Any]] = []
    started = time.time()
    last_reported_percent = -1

    print(
        f"[POSE] Start analysis: path={video_path}, fps={fps:.2f}, "
        f"total_frames={total_frames}, duration_sec={duration}",
        flush=True,
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if cnt % params.frame_interval == 0:
            current_frame_skeleton = np.zeros((17, 3), dtype=np.float32)

            with _SCOPE_LOCK:
                init_default_scope("mmdet")
                det_result = inference_detector(_det_model, frame)

            instances = det_result.pred_instances
            person_idx = (instances.labels == 0) & (instances.scores > params.det_score_thr)
            person_bboxes = instances.bboxes[person_idx].detach().cpu().numpy() if person_idx.any() else np.zeros((0, 4))

            if len(person_bboxes) > 0:
                largest_idx = int(np.argmax((person_bboxes[:, 2] - person_bboxes[:, 0]) * (person_bboxes[:, 3] - person_bboxes[:, 1])))
                target_bbox = person_bboxes[largest_idx: largest_idx + 1]

                with _SCOPE_LOCK:
                    init_default_scope("mmpose")
                    pose_results = inference_topdown(_pose_model, frame, target_bbox)

                if len(pose_results) > 0:
                    pred_instances = pose_results[0].pred_instances
                    keypoints = pred_instances.keypoints[0]
                    scores = pred_instances.keypoint_scores[0]

                    current_frame_skeleton = np.concatenate([keypoints, scores[:, None]], axis=-1).astype(np.float32)
                    current_frame_skeleton[:, 2] *= float(params.conf_scale)
        

            skeleton_buffer.append(current_frame_skeleton)

            if len(skeleton_buffer) == params.window_size:
                input_data = np.array(skeleton_buffer, dtype=np.float32)[None, ...]

                fake_anno = {
                    "keypoint": input_data,
                    "total_frames": params.window_size,
                    "img_shape": (h, w),
                    "original_shape": (h, w),
                    "label": -1
                }

                with _SCOPE_LOCK:
                    init_default_scope("mmaction")
                    recog_result = inference_recognizer(_recognizer, fake_anno)

                probs = recog_result.pred_score.detach().cpu().numpy()
                top1_label, top1_score = _topk_from_probs(probs, topk=1)[0]

                t_sec = cnt / fps
                per_frame_preds.append({
                    "t": float(t_sec),
                    "label": top1_label,
                    "score": float(top1_score)
                })

        cnt += 1

        if total_frames > 0:
            current_percent = min(100, int((cnt / max(1, total_frames)) * 100))
            if current_percent >= last_reported_percent + 5 or current_percent == 100:
                print(_render_progress(cnt, total_frames, started), flush=True)
                last_reported_percent = current_percent
        elif cnt % 300 == 0:
            print(f"[POSE] processed_frames={cnt} elapsed={time.time() - started:.1f}s", flush=True)

    cap.release()
    elapsed = time.time() - started

    timeline = _postprocess_timeline(per_frame_preds, fps=fps, params=params)

    print(f"[POSE] Finished analysis in {elapsed:.1f}s, processed_frames={cnt}", flush=True)

    return {
        "video": {
            "path": video_path,
            "fps": float(fps),
            "total_frames": int(total_frames),
            "width": int(w),
            "height": int(h),
            "duration_sec": None if duration is None else float(duration),
            "processed_frames": int(cnt),
            "analysis_time_sec": float(elapsed)
        },
        "timeline": timeline,
    }


if __name__ == "__main__":
    test_video = "../test/input/test_holdingback.mp4"
    result = analyze_video_to_timeline(test_video)
    print(result["video"])
    print(result["timeline"][:5])
