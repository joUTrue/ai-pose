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
# 로그/경고 최소화
# -------------------------
warnings.filterwarnings("ignore")
for name in ["mmengine", "mmdet", "mmpose", "mmaction"]:
    MMLogger.get_instance(name).setLevel(logging.ERROR)

# -------------------------
# 전역 설정(원 코드 유지)
# -------------------------
register_mmdet()
register_mmpose()
register_mmaction()

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

DET_CONFIG = "config/detection/yolox/yolox_s_8xb8-300e_coco.py"
DET_CHECKPOINT = "config/detection/yolox/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth"

POSE_CONFIG = "config/pose/td-hm_hrnet-w48_dark-8xb32-210e_coco-384x288.py"
POSE_CHECKPOINT = "config/pose/td-hm_hrnet-w48_dark-8xb32-210e_coco-384x288-39c3c381_20220916.pth"

STGCN_CONFIG = "config/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_forPresentation.py"
STGCN_CHECKPOINT = "checkpoints/20260109_2109.pth"

LABEL_NAMES = ["Touching head", "Touching face", "Touching body", "Touching hand", 
                "Shaking head", "Swaying head", "Bowing", "Holding back", 
                "Swaying hand", "Swaying body", "Slanting", "Skewing"]

SKELETON_LINKS = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6)
]

# MMEngine default scope는 전역 상태를 건드리는 경우가 있어
# 동시 요청이 들어오면 섞일 수 있습니다 -> 락 권장
_SCOPE_LOCK = threading.Lock()

# -------------------------
# 모델 전역(서버 시작 시 1회 로딩)
# -------------------------
_det_model = None
_pose_model = None
_recognizer = None


def load_models_once() -> None:
    """FastAPI startup 이벤트에서 1회 호출 권장."""
    global _det_model, _pose_model, _recognizer
    if _det_model is not None:
        return

    # detector
    with _SCOPE_LOCK:
        detcfg = mmengine.Config.fromfile(DET_CONFIG)
        detcfg.model.type = f"mmdet.{detcfg.model.type}"
        detcfg.default_scope = "mmdet"
        init_default_scope("mmdet")
        _det_model = init_detector(detcfg, DET_CHECKPOINT, device=DEVICE)

    # pose
    with _SCOPE_LOCK:
        posecfg = mmengine.Config.fromfile(POSE_CONFIG)
        posecfg.default_scope = "mmpose"
        init_default_scope("mmpose")
        _pose_model = init_model(posecfg, POSE_CHECKPOINT, device=DEVICE)

    # stgcn
    with _SCOPE_LOCK:
        init_default_scope("mmaction")
        _recognizer = init_recognizer(STGCN_CONFIG, STGCN_CHECKPOINT, device=DEVICE)


@dataclass
class RecogParams:
    frame_interval: int = 2          # 1: every frame, 2: skip 1 frame ...
    det_score_thr: float = 0.5
    window_sec: float = 2.0          # 슬라이딩 윈도우 길이(초)
    conf_scale: float = 2500.0       # 원 코드의 confidence scale 보정
    topk: int = 1                    # 타임라인은 보통 top1만 충분
    score_threshold: float = 0.60    # 이 점수 이상만 이벤트로 인정
    merge_gap_sec: float = 0.25      # 이벤트 구간 사이 작은 gap은 병합
    min_event_sec: float = 0.30      # 너무 짧은 이벤트는 제거


def _topk_from_probs(probs: np.ndarray, topk: int) -> List[Tuple[str, float]]:
    idxs = np.argsort(probs)[-topk:][::-1]
    return [(LABEL_NAMES[i], float(probs[i])) for i in idxs]


def _postprocess_timeline(
    per_frame: List[Dict[str, Any]],
    fps: float,
    params: RecogParams
) -> List[Dict[str, Any]]:
    """
    per_frame: [{"t": float, "label": str, "score": float}, ...]
    -> segments: [{"label","start","end","avg_score","max_score","frames"}]
    """
    if not per_frame:
        return []

    # 1) score threshold 적용 (미만은 label=None 처리)
    cleaned = []
    for p in per_frame:
        if p["score"] >= params.score_threshold:
            cleaned.append(p)
        else:
            cleaned.append({"t": p["t"], "label": None, "score": p["score"]})

    # 2) 연속 구간 만들기
    segments = []
    cur = None

    for p in cleaned:
        lbl = p["label"]
        t = p["t"]
        s = p["score"]

        if lbl is None:
            # 끊김 처리
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

    # 3) gap 병합
    merged = []
    for seg in segments:
        if not merged:
            merged.append(seg)
            continue
        prev = merged[-1]
        gap = seg["start"] - prev["end"]
        if seg["label"] == prev["label"] and gap <= params.merge_gap_sec:
            # 병합
            prev["end"] = seg["end"]
            prev["sum_score"] += seg["sum_score"]
            prev["max_score"] = max(prev["max_score"], seg["max_score"])
            prev["frames"] += seg["frames"]
        else:
            merged.append(seg)

    # 4) 최소 길이 필터 + 출력 포맷 정리
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
    """
    입력: video_path
    출력: {
      "video": {...},
      "timeline": [...segments...],
      "per_frame": [...optional...]
    }
    """
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

    window_size = max(1, int(params.window_sec * fps))
    skeleton_buffer = deque(maxlen=window_size)

    cnt = 0
    last_skeleton = np.zeros((17, 3), dtype=np.float32)

    per_frame_preds: List[Dict[str, Any]] = []
    started = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 스킵 정책
        if cnt % params.frame_interval == 0:
            current_frame_skeleton = np.zeros((17, 3), dtype=np.float32)

            # 1) detection
            with _SCOPE_LOCK:
                init_default_scope("mmdet")
                det_result = inference_detector(_det_model, frame)

            instances = det_result.pred_instances
            person_idx = (instances.labels == 0) & (instances.scores > params.det_score_thr)
            person_bboxes = instances.bboxes[person_idx].detach().cpu().numpy() if person_idx.any() else np.zeros((0, 4))
            # person_scores = instances.scores[person_idx].detach().cpu().numpy() if person_idx.any() else np.zeros((0,))

            # 2) pose (largest person only)
            if len(person_bboxes) > 0:
                largest_idx = int(np.argmax((person_bboxes[:, 2] - person_bboxes[:, 0]) * (person_bboxes[:, 3] - person_bboxes[:, 1])))
                target_bbox = person_bboxes[largest_idx: largest_idx + 1]

                with _SCOPE_LOCK:
                    init_default_scope("mmpose")
                    pose_results = inference_topdown(_pose_model, frame, target_bbox)

                if len(pose_results) > 0:
                    pred_instances = pose_results[0].pred_instances
                    keypoints = pred_instances.keypoints[0]            # (17, 2)
                    scores = pred_instances.keypoint_scores[0]         # (17,)

                    current_frame_skeleton = np.concatenate([keypoints, scores[:, None]], axis=-1).astype(np.float32)
                    current_frame_skeleton[:, 2] *= float(params.conf_scale)

            last_skeleton = current_frame_skeleton.copy()
        else:
            current_frame_skeleton = last_skeleton.copy()

        skeleton_buffer.append(current_frame_skeleton)

        # 3) action recognition (window full)
        if len(skeleton_buffer) >= window_size:
            input_data = np.array(skeleton_buffer, dtype=np.float32)[None, ...]  # (1, T, 17, 3)

            fake_anno = {
                "keypoint": input_data,
                "total_frames": window_size,
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

    cap.release()
    elapsed = time.time() - started

    timeline = _postprocess_timeline(per_frame_preds, fps=fps, params=params)

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
        # 디버그 필요하면 열어두세요. (기본은 너무 커질 수 있어 비추천)
        # "per_frame": per_frame_preds
    }


if __name__ == "__main__":
    # 로컬 테스트용
    test_video = "../test/input/test_holdingback.mp4"
    result = analyze_video_to_timeline(test_video)
    print(result["video"])
    print(result["timeline"][:5])
