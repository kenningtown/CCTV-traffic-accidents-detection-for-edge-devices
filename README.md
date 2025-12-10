# CCTV-traffic-accidents-detection-for-edge-devices

This project explores **automatic traffic accident detection** from fixed CCTV cameras using deep learning, with the constraint that the final system should be deployable on a **resource-constrained edge device** (Raspberry Pi).

I build on the **TAD-benchmark** dataset and a small subset of **TU-DAT** videos to train and compare:

- A **spatial baseline**: fine-tuned **YOLOv8n-cls** on single frames.
- A **temporal model**: a custom **ConvLSTM** network operating on 16-frame clips.

Both models are evaluated on a cleaned test split using accuracy, confusion matrices, per-class precision/recall/F1, and inference speed. The YOLOv8n-cls model is deployed to a Raspberry Pi for edge inference.

---

## 1. Datasets

### 1.1 TAD-benchmark

Main dataset:

- **TAD: A Large-Scale Benchmark for Traffic Accidents Detection From Video Surveillance**  
- Contains 344 real CCTV videos labeled as **accident** or **normal**.

Official paper:  
> TAD: A Large-Scale Benchmark for Traffic Accidents Detection From Video Surveillance

TAD raw videos are **not included** in this repository due to size and licensing.  
You must download them separately and place them in the appropriate folder structure (see below).

### 1.2 TU-DAT additions (test accidents)

During the project I found that many “accident” videos in the official TAD test split only show **post-crash scenes**. To get a more realistic evaluation, I replaced 16 such videos with crash sequences from:

> TU-DAT: A Computer Vision Dataset on Road Traffic Anomalies

These added videos are used only in the **test accident** subset. All derived indices (`splits.json`, `clips_index.json`) reflect this cleaned test set.

---

## 2. Directory Structure (this repo)

A minimal, cleaned version of the project looks like:

```text
traffic-accident-edge/
  notebooks/
    dataset_exploration.ipynb
    clip_index.ipynb
    yolo8_train.ipynb          #or yolo_cls_train.ipynb
    yolo_baseline.ipynb
    convlstm_train.ipynb
  TAD-benchmark/
    splits.json                #video -> split + label
    clips_index.json           #central clip index (16-frame clips)
    TAD-YOLO-CLS/
      train/accident/          #(not in repo, created locally)
      train/normal/
      val/accident/
      val/normal/
      test/accident/
      test/normal/
      tad_cls.yaml             #YOLO classification config
  best.pt                      #fine-tuned YOLOv8n-cls weights 
  convlstm_best.pth            #best ConvLSTM checkpoint
  edge_inference_yolo.py       #Raspberry Pi script
  README.md
