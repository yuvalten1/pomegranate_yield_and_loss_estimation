<h1 align="center"> Tracking-by-detection framework for simultaneous tree-scale pomegranate yield and fruit loss estimation from UAV </h1>

<p align="center">
 <img width="431" height="689" alt="image" src="https://github.com/user-attachments/assets/5bb5dcc8-5024-4944-91c0-11ffa0c5105d" />
</p>

<h4 align="center">
Yuval Tenenboim<sup>1,2</sup>, Yael Edan<sup>1</sup>, Idit Ginzberg<sup>3</sup>, Victor Alchanati<sup>4</sup>, Tarin Paz-Kagan<sup>2</sup>
</h4>

<h4 align="center">
<sup>1</sup> Dept. of Industrial Engineering & Management, Ben-Gurion University of the Negev, Israel
</h4>

<h4 align="center">
<sup>2</sup> The Jacob Blaustein Institutes for Desert Research, Ben-Gurion University of the Negev, Israel
</h4>

<h4 align="center">
<sup>3</sup> Institute of Plant Sciences, Agricultural Research Organization, Volcani Institute, Rishon LeZion, Israel
</h4>

<h4 align="center">
<sup>4</sup> Institute of Agricultural and Biosystems Engineering, Agricultural Research Organization, Volcani Institute, Israel
</h4>

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
<p>
The overall workflow of the proposed framework is illustrated in the figure below. 
It summarizes the main stages of the pipeline, from UAV data acquisition and dataset preparation 
to fruit detection, tracking, and tree-scale yield and fruit loss estimation.
</p>
<p align="center">
<img width="2000" height="789" alt="image" src="https://github.com/user-attachments/assets/9f3700c1-0d0b-4f90-a487-9fa1d653b2be" />
</p>
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

- splits detections according to the **left and right sides of the video frame**
- counts fruits only on the side corresponding to the measured tree
- aggregates these counts to estimate tree-level yield and fruit loss

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

<h2>Video Filename Format</h2>

<p>
The video processing pipeline extracts metadata directly from the video filename.
Therefore, videos must follow a specific naming convention in order to be processed correctly.
</p>

<p>
The filename structure used in this repository is:
</p>

<pre>
treeNumber_side_plot_additional_information.mp4
</pre>

<p>
For example:
</p>

<pre>
1_r_mishmar_hanegev_11_org_vid_row_9_plot_11_1_first_forward.mp4
</pre>

<h3>Parsed Information</h3>

<ul>
<li><b>treeNumber</b> – the tree index in the orchard row</li>
<li><b>side</b> – side of the tree being measured, indicating which side of the video the yield and fruit loss are counted from</li>    <ul>
        <li><b>l</b> – left side</li>
        <li><b>r</b> – right side</li>
    </ul>
</li>
<li><b>plot</b> – orchard plot identifier extracted from the name</li>
</ul>

<p>
The code automatically parses these fields using the function <code>parse_video_name()</code>.
This information is later used to aggregate fruit counts per tree and per orchard plot.
</p>

<h3>Example Parsing</h3>

<pre>
Filename:
1_r_mishmar_hanegev_11_org_vid_row_9_plot_11_1_first_forward.mp4

Parsed values:
tree_number = 1
side = r
plot = mishmar_hanegev_11
</pre>

<h3>Important</h3>

<p>
If the filename does not follow this structure, the pipeline will not be able to extract
tree and plot information correctly and the video may be skipped during processing.
</p>
