# Project Haze: Technical Documentation & Retrospective

**Author:** Chief Technical Writer
**Date:** March 15, 2026
**Status:** Alpha / Proof-of-Concept

## 1. Executive Summary

Project Haze is a high-performance, lightweight body measurement estimation system. It converts RGB video/image input into precise biometric data (Chest, Waist, Hips, Shoulder Width) without requiring a GPU or dense 3D scanners.

## 2. Technical Approach & Architecture

We opted for a **Heuristic-Skeletal Mapping** approach. Instead of attempting to reconstruct a full 3D body mesh (which is computationally expensive), we extract 33 skeletal landmarks and apply high-order geometric approximations.

### The Mathematics of Measurement

* **Normalized Scaling (The Height Anchor):** Since camera distance varies, we use the subject's height as a baseline. We measure the vertical span from **Eye-Midpoint to Ankle-Midpoint**.
  * *Mathematical Correction:* We mapped this span as **88% of total height (170cm reference)** to account for the crown-to-eye and ankle-to-floor proportions.
* **The Ellipse Hack (Circumferences):** MediaPipe provides 3D widths but no volume. We treat the torso as a series of stacked ellipses.
  * *Formula:* We use **Ramanujan’s First Approximation** for ellipse perimeter:  
        $P \approx \pi [3(a+b) - \sqrt{(3a+b)(a+3b)}]$  
        where *a* is the measured width and *b* is the estimated depth (using anatomical ratios).
* **Anatomical Padding (The Flesh Offset):** MediaPipe landmarks sit at the joint centers (bone). To get real-world measurements, we added constants:
  * **Shoulders:** +11cm offset (deltoid and skin thickness).
  * **Hips:** +14cm offset (pelvic tissue and gluteal mass).

## 3. System-Level Capabilities

1. **Smart Frame Sampling:** We extract 10–15 evenly spaced frames from videos to prevent CPU timeouts while ensuring a statistically significant dataset.
2. **Z-Axis Alignment Gating:** A "Z-depth" check on shoulders ensures the user is facing the camera. If the person is rotated (> 0.05 z-diff), the frame is discarded to prevent width distortion.
3. **Median-Robust Aggregation:** We use `numpy.median` across all valid frames. This eliminates outliers caused by momentary blinks, baggy clothes, or motion blur.

## 4. Engineering Challenges & Pivots

The project originally considered the **SMPL-X** (Skinned Multi-Person Linear) model. However, we faced significant **deployment friction**: SMPL-X requires heavy PyTorch dependencies, specialized model weights, and high VRAM, making it unsuitable for a lean FastAPI microservice.

* **The Pivot:** We shifted to **MediaPipe BlazePose (Heavy)**. It offered the best balance of speed and precision for a CPU-bound environment, allowing us to deploy on standard Docker containers like Hugging Face Spaces.

## 5. Honest Assessment: Accuracy & Industry Standing

* **Current State:** Haze is a brilliant **testing and prototyping tool**. It provides consistent results for users in form-fitting clothing with proper framing.
* **Limitations:** Because it uses a sparse skeletal model rather than a dense point cloud or mesh, it is **not yet industry-standard** for medical or high-end tailoring. It cannot account for 100% of body types (e.g., extreme obesity or high-fashion clothing) as effectively as a full SMPL or volumetric scanner.
* **Conclusion:** It is a highly optimized, decent lightweight alternative that punches above its weight class.

## 6. AI-Assisted Development

* **Gemini:** Utilized for high-level **System Architecture**, drafting the multi-stage pipeline, and **Prompt Tuning** to refine the mathematical heuristics.
* **Antigravity:** Used for **Rapid Code Prototyping**, fixing MediaPipe/NumPy dependency conflicts, and resolving Docker deployment hurdles.

## 7. Future Roadmap (The Next 2 Weeks)

If given additional runway, the priority would be:

* **Efficiency-Focused SMPL:** Re-attempt a pivot back to a dense mesh model but stripped down (Quantized/TFLite) to ensure it stays efficient enough for mass public usage without requiring expensive GPU clusters.
* **Dynamic Reference Scaling:** Implementing a feature for users to input their exact height, replacing the 170cm hardcoded assumption for millimeter-perfect precision.
