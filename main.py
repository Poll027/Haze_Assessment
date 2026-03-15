"""
Haze Assessment – Body Measurement Estimation API
=================================================
FastAPI server that accepts a video or image upload and returns
estimated body measurements (chest, waist, hips, shoulder width)
in centimetres, using MediaPipe BlazePose for 3D landmark
detection, ellipse-based circumference approximation, and
numpy.median aggregation across filtered frames.
"""

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

# ──────────────────────────────────────────────
# 1.  Pydantic Response Models
# ──────────────────────────────────────────────

class MeasurementResult(BaseModel):
    """
    The final JSON payload returned to the client.

    All circumference values are in **centimetres** and represent the
    *median* across all surviving (filtered) frames.
    """
    chest_cm: float   = Field(..., description="Estimated chest circumference in cm")
    waist_cm: float   = Field(..., description="Estimated waist circumference in cm")
    hips_cm: float    = Field(..., description="Estimated hip circumference in cm")
    shoulder_width_cm: float = Field(..., description="Estimated shoulder width in cm")
    frames_processed: int    = Field(..., description="Total frames sampled from the video")
    frames_used: int         = Field(..., description="Frames that passed visibility & alignment filters")

class ErrorResponse(BaseModel):
    """Standard error envelope."""
    detail: str


# ──────────────────────────────────────────────
# 2.  FastAPI Application
# ──────────────────────────────────────────────

app = FastAPI(
    title="Haze Body Measurement API",
    description=(
        "Upload a video (or image) of a person standing and receive "
        "estimated body measurements derived from MediaPipe BlazePose."
    ),
    version="0.1.0",
)


# ──────────────────────────────────────────────
# 3.  Upload Endpoint
# ──────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".jpg", ".jpeg", ".png"}
IMAGE_EXTENSIONS   = {".jpg", ".jpeg", ".png"}

# Number of frames to sample from a video
MIN_SAMPLE_FRAMES = 10
MAX_SAMPLE_FRAMES = 15

def _validate_extension(filename: str) -> str:
    """Return the lowered extension or raise 400."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )
    return ext


# ──────────────────────────────────────────────
# 4.  Smart Frame Sampling (OpenCV)
# ──────────────────────────────────────────────

def extract_frames(file_path: str, ext: str) -> list[np.ndarray]:
    """
    Return a list of BGR frames from *file_path*.

    • **Image** → returns a single-element list.
    • **Video** → samples 10-15 evenly-spaced frames across the
      video's duration so we never process every frame.
    """
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

    # Decide how many frames to sample (clamp between MIN and MAX).
    n_samples = min(MAX_SAMPLE_FRAMES, max(MIN_SAMPLE_FRAMES, total_frames))

    # Evenly-spaced indices across the video timeline.
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


# ──────────────────────────────────────────────
# 5.  MediaPipe BlazePose Detection
# ──────────────────────────────────────────────

# Initialise the Pose solution once at module level so the model is
# loaded a single time, not on every request.
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(
    static_image_mode=True,       # treat each frame independently
    model_complexity=2,           # highest accuracy (full 33 landmarks)
    enable_segmentation=False,    # we don't need the mask
    min_detection_confidence=0.5, # initial detector threshold
)


def detect_landmarks(frames: list[np.ndarray]) -> list[list]:
    """
    Run BlazePose on every frame and return a list of raw landmark
    lists.  Each element is the 33-point landmark list for one frame,
    or ``None`` if no person was detected.

    MediaPipe expects **RGB** input, so we convert from OpenCV's BGR.
    """
    results: list[list] = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection = pose_detector.process(rgb)
        if detection.pose_landmarks:
            results.append(detection.pose_landmarks.landmark)
        else:
            results.append(None)
    return results


# ──────────────────────────────────────────────
# 6.  Frame Filtering (Visibility + Z-Axis Alignment)
# ──────────────────────────────────────────────

# Minimum visibility score for a landmark to be considered reliable.
MIN_VISIBILITY = 0.7

# Maximum allowed Z-depth difference between left and right shoulders.
# If exceeded, the person is too rotated for accurate width readings.
MAX_SHOULDER_Z_DIFF = 0.05

# MediaPipe BlazePose landmark indices for the key body points
# we need to validate and later measure.
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
    """
    Return True only if **every** key landmark has a visibility score
    ≥ MIN_VISIBILITY.  This discards frames where body parts are
    occluded, cut off, or the model is guessing.
    """
    for idx in KEY_LANDMARKS:
        if landmarks[idx].visibility < MIN_VISIBILITY:
            return False
    return True


def _passes_frontal_alignment(landmarks) -> bool:
    """
    Return True only if the person is facing the camera squarely.

    We compare the Z-coordinates (depth) of the left and right
    shoulders.  If one shoulder is significantly closer to the
    camera than the other, the person is rotated, which distorts
    the left–right width measurements we rely on.
    """
    z_left  = landmarks[LMK.LEFT_SHOULDER].z
    z_right = landmarks[LMK.RIGHT_SHOULDER].z
    return abs(z_left - z_right) <= MAX_SHOULDER_Z_DIFF


def filter_landmarks(landmark_sets: list) -> list:
    """
    Apply both quality gates to every detected landmark set.

    Returns only the landmark sets that:
      1. Had a person detected (not None).
      2. Passed the visibility confidence check.
      3. Passed the frontal Z-axis alignment check.
    """
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


# ──────────────────────────────────────────────
# 7.  Geometry & Measurement Functions
# ──────────────────────────────────────────────

# Reference height used for pixel→cm scaling (no user input).
REFERENCE_HEIGHT_CM = 170.0

# Anatomical depth-to-width ratios for ellipse estimation.
# These approximate how "deep" (front-to-back) the body is
# relative to its measured left-to-right width at each level.
DEPTH_RATIO_CHEST = 0.70   # adjusted up for muscular chest
DEPTH_RATIO_WAIST = 0.75   # waist is rounder
DEPTH_RATIO_HIPS  = 0.80   # hips are the most circular cross-section

# Anatomical Padding (Biometric Offsets)
# MediaPipe joints are internal. We add these cm to account for
# outer muscle, skin, and fat beyond the bone.
SHOULDER_PADDING_CM = 11.0  # total extra width (5.5cm per side)
HIP_PADDING_CM      = 14.0  # total extra width (7cm per side)



def _euclidean_dist(lm_a, lm_b) -> float:
    """
    3D Euclidean distance between two MediaPipe landmarks.

    Uses x, y, z (all in normalised coordinates) so the distance
    is still normalised — we convert to cm later via the scaling
    ratio.
    """
    return math.sqrt(
        (lm_a.x - lm_b.x) ** 2 +
        (lm_a.y - lm_b.y) ** 2 +
        (lm_a.z - lm_b.z) ** 2
    )


def _euclidean_dist_2d(lm_a, lm_b) -> float:
    """
    2D Euclidean distance (x, y only) between two landmarks.

    Used for the pixel-height measurement where Z is irrelevant
    (we just need the vertical span in the image plane).
    """
    return math.sqrt(
        (lm_a.x - lm_b.x) ** 2 +
        (lm_a.y - lm_b.y) ** 2
    )


def _ellipse_circumference(width: float, depth: float) -> float:
    """
    Approximate the perimeter of an ellipse using **Ramanujan’s
    first approximation**:

        P ≈ π * [ 3(a + b) - √((3a + b)(a + 3b)) ]

    where *a* and *b* are the semi-axes (half of width and depth).

    This is accurate to within ~0.1 % for most body-like ellipses
    (eccentricity < 0.95), which is far better than π·d.
    """
    a = width  / 2.0   # semi-major axis  (left–right)
    b = depth  / 2.0   # semi-minor axis  (front–back)
    return math.pi * (3.0 * (a + b) - math.sqrt((3.0 * a + b) * (a + 3.0 * b)))


def _pixel_height(landmarks) -> float:
    """
    Measure the person’s height in *normalised pixel units* from
    the midpoint of the eyes down to the midpoint of the ankles.

    We use eyes (not top-of-head) because MediaPipe has no
    “crown” landmark; the eye midpoint is the highest reliable
    point.
    """
    # Eye midpoint
    left_eye  = landmarks[LMK.LEFT_EYE]
    right_eye = landmarks[LMK.RIGHT_EYE]
    eye_mid_x = (left_eye.x + right_eye.x) / 2.0
    eye_mid_y = (left_eye.y + right_eye.y) / 2.0

    # Ankle midpoint
    left_ankle  = landmarks[LMK.LEFT_ANKLE]
    right_ankle = landmarks[LMK.RIGHT_ANKLE]
    ankle_mid_x = (left_ankle.x + right_ankle.x) / 2.0
    ankle_mid_y = (left_ankle.y + right_ankle.y) / 2.0

    return math.sqrt(
        (eye_mid_x - ankle_mid_x) ** 2 +
        (eye_mid_y - ankle_mid_y) ** 2
    )


def _pixel_to_cm_ratio(landmarks) -> float:
    """
    Compute the conversion factor: how many cm each normalised
    pixel-unit represents.

    All subsequent normalised distances can be multiplied by this
    ratio to get centimetres.
    """
    ph = _pixel_height(landmarks)
    if ph == 0:
        return 0.0  # degenerate case guard

    # BIOMETRIC CORRECTION: The distance from eyes-to-ankles is
    # approximately 88% of a person's total standing height.
    # If we map 170cm to just the eye-ankle span, we over-scale.
    total_to_eye_ankle_ratio = 0.88
    effective_height_cm = REFERENCE_HEIGHT_CM * total_to_eye_ankle_ratio

    return effective_height_cm / ph


def measure_single_frame(landmarks) -> dict:
    """
    From one set of 33 landmarks, compute all four measurements
    in centimetres:

    1. Shoulder width - direct 3D distance, scaled.
    2. Chest circumference - width between shoulders at torso
       level, ellipse-approximated.
    3. Waist circumference - width at the hip midpoint level,
       ellipse-approximated.
    4. Hip circumference - width between hip landmarks,
       ellipse-approximated.

    Returns a dict with keys matching MeasurementResult fields.
    """
    ratio = _pixel_to_cm_ratio(landmarks)

    # ── Shoulder width (skeletal distance + padding) ──
    skeletal_shoulder_w = _euclidean_dist(
        landmarks[LMK.LEFT_SHOULDER],
        landmarks[LMK.RIGHT_SHOULDER],
    ) * ratio
    shoulder_w_cm = skeletal_shoulder_w + SHOULDER_PADDING_CM

    # ── Chest: use the padded shoulder width as the major axis. ──
    chest_width_cm = shoulder_w_cm
    chest_depth_cm = chest_width_cm * DEPTH_RATIO_CHEST
    chest_circ     = _ellipse_circumference(chest_width_cm, chest_depth_cm)

    # ── Waist: natural waist sits between shoulders and hips. ──
    ls = landmarks[LMK.LEFT_SHOULDER]
    rs = landmarks[LMK.RIGHT_SHOULDER]
    lh = landmarks[LMK.LEFT_HIP]
    rh = landmarks[LMK.RIGHT_HIP]

    class _Pt:
        def __init__(self, x, y, z): self.x, self.y, self.z = x, y, z

    waist_left  = _Pt((ls.x + lh.x) / 2, (ls.y + lh.y) / 2, (ls.z + lh.z) / 2)
    waist_right = _Pt((rs.x + rh.x) / 2, (rs.y + rh.y) / 2, (rs.z + rh.z) / 2)

    # Waist padding is interleaved (average of shoulder and hip padding)
    waist_padding = (SHOULDER_PADDING_CM + HIP_PADDING_CM) / 2.0
    waist_width_cm = (_euclidean_dist(waist_left, waist_right) * ratio) + waist_padding
    waist_depth_cm = waist_width_cm * DEPTH_RATIO_WAIST
    waist_circ     = _ellipse_circumference(waist_width_cm, waist_depth_cm)

    # ── Hips: skeletal distance + pelvic padding. ──
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



# ──────────────────────────────────────────────
# 8.  Robust Aggregation (numpy.median)
# ──────────────────────────────────────────────

def aggregate_measurements(per_frame: list[dict]) -> dict:
    """
    Collapse per-frame measurement dicts into a single result
    using **numpy.median** for each metric.

    Why median instead of mean?
    • The mean is pulled towards outliers (e.g. one frame where an
      arm is mistakenly detected as a shoulder).
    • The median picks the *middle* value, so a single bad frame
      cannot distort the final number.

    Returns a dict with the same keys, values rounded to 1 dp.
    """
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
    """
    Accepts a video or image file upload and returns estimated body
    measurements (chest, waist, hips, shoulder width) in centimetres.

    **Processing pipeline:**
    1. Extract 10–15 evenly-spaced frames (smart sampling).
    2. Run MediaPipe BlazePose 33-point detection on each frame.
    3. Discard frames with low landmark visibility (< 0.7).
    4. Discard frames where the subject is rotated (Z-axis check).
    5. Calculate per-frame measurements (ellipse approximation +
       170 cm reference-height scaling).
    6. Aggregate across surviving frames via **numpy.median**.
    """
    ext = _validate_extension(file.filename or "upload.bin")

    # Save the upload to a temporary file so OpenCV can read it by path.
    tmp_dir = tempfile.mkdtemp(prefix="haze_")
    tmp_path = os.path.join(tmp_dir, f"upload{ext}")

    try:
        with open(tmp_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        # ── Step 2: Extract frames & detect landmarks ──
        frames = extract_frames(tmp_path, ext)
        landmark_sets = detect_landmarks(frames)
        frames_processed = len(frames)

        # ── Step 3: Filter for quality & alignment ──
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

        # ── Step 4: Measure each surviving frame ──
        per_frame_measurements = [
            measure_single_frame(lm) for lm in valid_landmarks
        ]

        # ── Step 5: Aggregate via numpy.median ──
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
        # Always clean up the temp directory.
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ──────────────────────────────────────────────
# 9.  Dev-mode runner
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
