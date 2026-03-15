import math
import os
import tempfile
import shutil
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field

class MeasurementResult(BaseModel):
    chest_cm: float   = Field(..., description="Estimated chest circumference in cm")
    waist_cm: float   = Field(..., description="Estimated waist circumference in cm")
    hips_cm: float    = Field(..., description="Estimated hip circumference in cm")
    shoulder_width_cm: float = Field(..., description="Estimated shoulder width in cm")
    frames_processed: int    = Field(..., description="Total frames sampled from the video")
    frames_used: int         = Field(..., description="Frames that passed visibility & alignment filters")

class ErrorResponse(BaseModel):
    detail: str

app = FastAPI(
    title="Haze Body Measurement API",
    description=(
        "Upload a video (or image) of a person standing and receive "
        "estimated body measurements derived from MediaPipe BlazePose."
    ),
    version="0.1.0",
)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".jpg", ".jpeg", ".png"}
IMAGE_EXTENSIONS   = {".jpg", ".jpeg", ".png"}
MIN_SAMPLE_FRAMES = 10
MAX_SAMPLE_FRAMES = 15

def _validate_extension(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )
    return ext

def extract_frames(file_path: str, ext: str) -> list[np.ndarray]:
    if ext in IMAGE_EXTENSIONS:
        img = cv2.imread(file_path)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image file.")
        return [img]

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open video file.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise HTTPException(status_code=400, detail="Video contains no frames.")

    n_samples = min(MAX_SAMPLE_FRAMES, max(MIN_SAMPLE_FRAMES, total_frames))
    indices = np.linspace(0, total_frames - 1, num=n_samples, dtype=int)

    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()

    if not frames:
        raise HTTPException(status_code=400, detail="Could not read any frames from video.")

    return frames

mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5,
)

def detect_landmarks(frames: list[np.ndarray]) -> list[list]:
    results: list[list] = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection = pose_detector.process(rgb)
        if detection.pose_landmarks:
            results.append(detection.pose_landmarks.landmark)
        else:
            results.append(None)
    return results

MIN_VISIBILITY = 0.7
MAX_SHOULDER_Z_DIFF = 0.05
LMK = mp_pose.PoseLandmark
KEY_LANDMARKS = [
    LMK.LEFT_SHOULDER,
    LMK.RIGHT_SHOULDER,
    LMK.LEFT_HIP,
    LMK.RIGHT_HIP,
    LMK.LEFT_ANKLE,
    LMK.RIGHT_ANKLE,
]

def _passes_visibility(landmarks) -> bool:
    for idx in KEY_LANDMARKS:
        if landmarks[idx].visibility < MIN_VISIBILITY:
            return False
    return True

def _passes_frontal_alignment(landmarks) -> bool:
    z_left  = landmarks[LMK.LEFT_SHOULDER].z
    z_right = landmarks[LMK.RIGHT_SHOULDER].z
    return abs(z_left - z_right) <= MAX_SHOULDER_Z_DIFF

def filter_landmarks(landmark_sets: list) -> list:
    surviving: list = []
    for lm in landmark_sets:
        if lm is None:
            continue
        if not _passes_visibility(lm):
            continue
        if not _passes_frontal_alignment(lm):
            continue
        surviving.append(lm)
    return surviving

REFERENCE_HEIGHT_CM = 170.0
DEPTH_RATIO_CHEST = 0.70
DEPTH_RATIO_WAIST = 0.75
DEPTH_RATIO_HIPS  = 0.80
SHOULDER_PADDING_CM = 11.0
HIP_PADDING_CM      = 14.0

def _euclidean_dist(lm_a, lm_b) -> float:
    return math.sqrt(
        (lm_a.x - lm_b.x) ** 2 +
        (lm_a.y - lm_b.y) ** 2 +
        (lm_a.z - lm_b.z) ** 2
    )

def _euclidean_dist_2d(lm_a, lm_b) -> float:
    return math.sqrt(
        (lm_a.x - lm_b.x) ** 2 +
        (lm_a.y - lm_b.y) ** 2
    )

def _ellipse_circumference(width: float, depth: float) -> float:
    a = width  / 2.0
    b = depth  / 2.0
    return math.pi * (3.0 * (a + b) - math.sqrt((3.0 * a + b) * (a + 3.0 * b)))

def _pixel_height(landmarks) -> float:
    left_eye  = landmarks[LMK.LEFT_EYE]
    right_eye = landmarks[LMK.RIGHT_EYE]
    eye_mid_x = (left_eye.x + right_eye.x) / 2.0
    eye_mid_y = (left_eye.y + right_eye.y) / 2.0

    left_ankle  = landmarks[LMK.LEFT_ANKLE]
    right_ankle = landmarks[LMK.RIGHT_ANKLE]
    ankle_mid_x = (left_ankle.x + right_ankle.x) / 2.0
    ankle_mid_y = (left_ankle.y + right_ankle.y) / 2.0

    return math.sqrt(
        (eye_mid_x - ankle_mid_x) ** 2 +
        (eye_mid_y - ankle_mid_y) ** 2
    )

def _pixel_to_cm_ratio(landmarks) -> float:
    ph = _pixel_height(landmarks)
    if ph == 0:
        return 0.0

    total_to_eye_ankle_ratio = 0.88
    effective_height_cm = REFERENCE_HEIGHT_CM * total_to_eye_ankle_ratio

    return effective_height_cm / ph

def measure_single_frame(landmarks) -> dict:
    ratio = _pixel_to_cm_ratio(landmarks)

    skeletal_shoulder_w = _euclidean_dist(
        landmarks[LMK.LEFT_SHOULDER],
        landmarks[LMK.RIGHT_SHOULDER],
    ) * ratio
    shoulder_w_cm = skeletal_shoulder_w + SHOULDER_PADDING_CM

    chest_width_cm = shoulder_w_cm
    chest_depth_cm = chest_width_cm * DEPTH_RATIO_CHEST
    chest_circ     = _ellipse_circumference(chest_width_cm, chest_depth_cm)

    ls = landmarks[LMK.LEFT_SHOULDER]
    rs = landmarks[LMK.RIGHT_SHOULDER]
    lh = landmarks[LMK.LEFT_HIP]
    rh = landmarks[LMK.RIGHT_HIP]

    class _Pt:
        def __init__(self, x, y, z): self.x, self.y, self.z = x, y, z

    waist_left  = _Pt((ls.x + lh.x) / 2, (ls.y + lh.y) / 2, (ls.z + lh.z) / 2)
    waist_right = _Pt((rs.x + rh.x) / 2, (rs.y + rh.y) / 2, (rs.z + rh.z) / 2)

    waist_padding = (SHOULDER_PADDING_CM + HIP_PADDING_CM) / 2.0
    waist_width_cm = (_euclidean_dist(waist_left, waist_right) * ratio) + waist_padding
    waist_depth_cm = waist_width_cm * DEPTH_RATIO_WAIST
    waist_circ     = _ellipse_circumference(waist_width_cm, waist_depth_cm)

    skeletal_hip_w = _euclidean_dist(
        landmarks[LMK.LEFT_HIP],
        landmarks[LMK.RIGHT_HIP],
    ) * ratio
    hip_width_cm = skeletal_hip_w + HIP_PADDING_CM
    hip_depth_cm = hip_width_cm * DEPTH_RATIO_HIPS
    hip_circ     = _ellipse_circumference(hip_width_cm, hip_depth_cm)

    return {
        "chest_cm":          round(chest_circ, 1),
        "waist_cm":          round(waist_circ, 1),
        "hips_cm":           round(hip_circ, 1),
        "shoulder_width_cm": round(shoulder_w_cm, 1),
    }

def aggregate_measurements(per_frame: list[dict]) -> dict:
    keys = ["chest_cm", "waist_cm", "hips_cm", "shoulder_width_cm"]
    result: dict = {}
    for key in keys:
        values = [frame[key] for frame in per_frame]
        result[key] = round(float(np.median(values)), 1)
    return result

@app.post(
    "/estimate",
    response_model=MeasurementResult,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    summary="Estimate body measurements from an uploaded video or image",
)
async def estimate_measurements(file: UploadFile = File(...)):
    ext = _validate_extension(file.filename or "upload.bin")

    tmp_dir = tempfile.mkdtemp(prefix="haze_")
    tmp_path = os.path.join(tmp_dir, f"upload{ext}")

    try:
        with open(tmp_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        frames = extract_frames(tmp_path, ext)
        landmark_sets = detect_landmarks(frames)
        frames_processed = len(frames)

        valid_landmarks = filter_landmarks(landmark_sets)
        frames_used = len(valid_landmarks)

        if frames_used == 0:
            raise HTTPException(
                status_code=422,
                detail=(
                    "No usable frames survived filtering. "
                    "Ensure the subject is fully visible and facing the camera."
                ),
            )

        per_frame_measurements = [
            measure_single_frame(lm) for lm in valid_landmarks
        ]

        final = aggregate_measurements(per_frame_measurements)

        return MeasurementResult(
            chest_cm=final["chest_cm"],
            waist_cm=final["waist_cm"],
            hips_cm=final["hips_cm"],
            shoulder_width_cm=final["shoulder_width_cm"],
            frames_processed=frames_processed,
            frames_used=frames_used,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)