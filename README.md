<h1 align="center"> Tracking-by-detection framework for simultaneous tree-scale pomegranate yield and fruit loss estimation from UAV </h1>

<p align="center">
 <img width="431" height="689" alt="image" src="https://github.com/user-attachments/assets/5bb5dcc8-5024-4944-91c0-11ffa0c5105d" />
</p>

<h4 align="center"> Author: Yuval Tenenboim </h4>
<h4 align="center"> Supervisors: Prof. Yael Edan, Dr. Tarin Paz-Kagan </h4>

<h4 align="center"> Dept. of Industrial Engineering & Management, Ben-Gurion University of the Negev </h4>
<h4 align="center"> The Jacob Blaustein Institutes for Desert Research, Ben-Gurion University of the Negev, Israel </h4>

---

## Repository Overview

This repository contains the **datasets and code used for the experiments presented in the COMPAG paper**:

**“Tracking-by-detection framework for simultaneous tree-scale pomegranate yield and fruit loss estimation from UAV imagery.”**

The repository provides an end-to-end pipeline for:

- image preprocessing
- object detection of healthy and defected pomegranates
- evaluation of detection models
- visualization of detections
- video-based tracking
- tree-scale yield and fruit loss estimation

The system processes **UAV RGB videos acquired above commercial orchards** and produces:

- annotated detection visualizations
- annotated tracking videos
- per-tree yield estimates
- per-tree fruit loss estimates

---

## Paper Overview

Fruit cracking causes substantial yield losses in pomegranate orchards, yet most vision-based yield estimation systems focus exclusively on counting healthy fruits and ignore losses occurring on the tree or on the ground.

This work proposes a **deep-learning framework for the joint estimation of yield and fruit loss using UAV-acquired RGB video data**.

The framework integrates:

- task-specific object detection models for **healthy (yield)** and **defective (loss)** fruits
- a **tracking-by-detection approach** for video-based fruit counting
- **tree-level yield and loss estimation** under commercial orchard conditions

The system was evaluated on UAV video data collected from multiple commercial orchard plots and validated against **field-based ground-truth measurements**, demonstrating that explicitly modelling fruit loss provides a more realistic and operationally relevant assessment of orchard productivity.

---

## Method Overview

The proposed framework consists of the following stages.

### 1. Image preprocessing

Video frames are enhanced using:

- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- saturation enhancement

These operations improve detection robustness under outdoor orchard lighting conditions.

---

### 2. Object detection

Two independent detection models are trained using **Ultralytics YOLO**:

| Model | Task |
|------|------|
| Yield model | Detect healthy pomegranates |
| Loss model | Detect cracked or defected fruits |

---

### 3. Video-based tracking

Detections are associated across frames using a **tracking-by-detection approach** based on the SORT tracker.

Each fruit receives a unique tracking ID, allowing robust counting across frames.

---

### 4. Fruit counting

Unique tracked IDs are used to estimate:

- total fruit yield
- total fruit loss

Counts are aggregated at the **tree level**.

---

### 5. Tree-scale estimation

UAV videos are captured while flying along orchard rows.

The framework:

- splits detections by **left and right tree sides**
- counts fruits for the relevant tree side
- aggregates counts for yield and loss estimation.

---

<h2>Project Structure</h2>

<p>The repository is organized as follows:</p>

<pre>
pomegranate_yield_and_loss_estimation/
│
├── main.py                     # Entry point for running the pipeline
│
├── preprocess.py               # Image preprocessing (CLAHE + saturation)
├── detection_eval.py           # Detection model evaluation (mAP, confusion matrix)
├── visualize_images.py         # Visualization of detections on images
│
├── tracking.py                 # Tracking utilities and helper functions
├── video_pipeline.py           # Video processing, tracking, and fruit counting
│
├── model_weights/
│   ├── yield_yolo11_weights.pt
│   └── loss_yolo11_weights.pt
│
├── sample_datasets/
│   ├── raw/                    # Original sample datasets included for demonstration
│   └── processed/              # Created automatically after running preprocessing
│
├── sample_videos/              # Sample UAV videos used for demonstration
│
└── outputs/                    # Generated results (created automatically)
    ├── video_visualizations/   # Annotated videos with tracking results
    └── tracking_results/       # CSV files containing yield and loss estimates
</pre>

<p>
Note: the folders <b>sample_datasets/processed</b> and <b>outputs/</b> are generated automatically when running the pipeline and are therefore not included in the repository. They will be created when the relevant scripts are executed.
</p>
