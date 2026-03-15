# Haze -- Body Measurement Estimation API

Haze is a high-precision, FastAPI-based system designed to estimate body
measurements (**Chest, Waist, Hips, and Shoulder Width**) from simple
video or image uploads.

The system leverages **MediaPipe BlazePose** for 3D landmark detection
and employs advanced geometric heuristics to map internal skeletal
points to realistic "fleshy" body circumferences.

------------------------------------------------------------------------

# 🚀 Key Features

## 1. Smart Frame Sampling

To prevent CPU timeouts and ensure performance, Haze processes videos by
sampling **10--15 evenly spaced frames** across the duration. This
ensures a broad dataset for calculation without the overhead of
processing every single frame.

## 2. Anatomical Gatekeeping (Filtering)

Not every frame is high quality. Haze uses a two-stage filter:

-   **Visibility Gate**\
    Discards frames where key landmarks (ankles, hips, shoulders) have a
    visibility score below `0.7`.

-   **Z-Axis Alignment Gating**\
    To prevent rotational distortion, the system checks the Z-depth of
    the shoulders. If the subject is not facing the camera squarely
    (shoulder Z-diff \> `0.05`), the frame is discarded.

## 3. Biometric Accuracy (Muscle & Flesh Refinement)

Standard skeletal markers represent internal joint centers, which are
significantly narrower than the physical body.

Haze implements:

-   **Anatomical Padding**\
    Adds precise offsets (`+11cm` for shoulders, `+14cm` for hips) to
    account for deltoids, lats, and soft tissue.

-   **Anatomical Scaling**\
    Maps the **Eye-to-Ankle span as 88% of total height**, ensuring the
    centimeter-to-pixel ratio reflects real human proportions.

-   **Ellipse Complexity**\
    Uses **Ramanujan's First Approximation** for ellipse circumferences
    rather than simple circular math.

## 4. Robust Aggregation

Instead of a simple average (which is vulnerable to glitches), Haze uses
**Median Aggregation** (`numpy.median`) across all survivor frames.

This effectively ignores one-off detection errors and ensures the final
result is stable.

------------------------------------------------------------------------

# 🛠️ Technology Stack

-   **Framework:** FastAPI\
-   **Computer Vision:** OpenCV (`cv2`)\
-   **AI Inference:** MediaPipe (BlazePose GH)\
-   **Math/Data:** NumPy\
-   **Validation:** Pydantic

------------------------------------------------------------------------

# 📦 Installation & Setup

## 1. Clone the Repository

``` bash
git clone https://github.com/Poll027/Haze_Assessment.git
cd Haze_Assessment
```

## 2. Set Up Virtual Environment

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

``` bash
pip install -r requirements.txt
```

## 4. Run the Server

``` bash
python main.py
```

The API will be available at:

    http://localhost:8000

Interactive documentation:

    http://localhost:8000/docs

------------------------------------------------------------------------

# 📸 Best Practices for Accuracy

For the most accurate results, users should follow these guidelines:

-   **Clothing:** Wear form-fitting or athletic gear. Baggy clothing
    will cause measurement inflation.
-   **Posturing:** Stand in an **A-Pose** facing the camera directly.
-   **Framing:** Ensure the full body (head to toe) is visible.
-   **Environment:** Use a well-lit area with a plain background.

------------------------------------------------------------------------

# 📡 API Documentation

## `POST /estimate`

Upload a video (`.mp4`, `.mov`, `.avi`) or image (`.jpg`, `.png`,
`.webp`).

### Request

Multipart form-data with a `file` field.

### Response

``` json
{
  "chest_cm": 102.4,
  "waist_cm": 88.7,
  "hips_cm": 95.2,
  "shoulder_width_cm": 44.1,
  "frames_processed": 15,
  "frames_used": 12
}
```

------------------------------------------------------------------------

*Haze architecture designed for precision body biometrics.*
