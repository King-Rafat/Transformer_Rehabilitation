<div align="center">

# 🦾 A Point Cloud Transformer for Remote Monitoring and Automated Assessment of Physical Rehabilitation Exercises

**Geometry-aware** skeletal point clouds **+** **axial self-attention** **=** a lightweight, interpretable, real-time rehabilitation scorer.

[![Paper](https://img.shields.io/badge/Paper-IEEE%20JBHI-00629B?logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/)
[![Conference](https://img.shields.io/badge/Status-Accepted-success)](https://github.com/King-Rafat/Transformer_Rehabilitation)
[![Python](https://img.shields.io/badge/Python-3.7%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2011.2%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/King-Rafat/Transformer_Rehabilitation?style=social)](https://github.com/King-Rafat/Transformer_Rehabilitation/stargazers)

<sub>Kazi Rafat · Md. Ismail Hossain · M M Lutfe Elahi · Sifat Momen · Fuad Rahman · Nabeel Mohammed · Shafin Rahman</sub>
<br>
<sub>North South University, Dhaka, Bangladesh · Apurba Technologies Ltd.</sub>

</div>

<p align="center">
  <img src="graphical_abstract_viz" alt="Method overview" width="85%">
</p>

---

## 📌 TL;DR

Patients do roughly **90%** of their rehab exercises at home, with no expert watching. This model scores how well an exercise is performed from skeleton data alone, so progress can be tracked remotely, cheaply, and in real time. It treats the skeleton as a **point cloud** (geometry, not a fixed graph), "walks" curves over it to enrich each joint with positional and geometric context, and uses **axial self-attention** to model space and time efficiently while highlighting the joints that matter most.

> **Three things make it different:**
>
> 🧩 **Geometric, not graph-locked** — curve-based point aggregation preserves bone lengths, directions, and structure that predefined ST-GCN graphs throw away.
>
> ⚡ **Fast and lightweight** — axial attention cuts self-attention cost; the model runs in **real time on a consumer CPU**, no GPU required.
>
> 🔍 **Interpretable** — attention coefficients reveal which joints drive the score, giving patients actionable feedback.

---

## ✨ Why this design

| Prior approach | Limitation | Our fix |
|---|---|---|
| Hand-crafted features | Expert effort, bias, misses key cues | Learned geometry-aware features |
| CNNs | Miss long-range spatio-temporal info | Transformer with axial attention |
| ST-GCNs | Fixed graph, topologically identical every frame, drops bone length/direction | Point-cloud geometry + curve aggregation |
| Full self-attention | Quadratic cost, slow | Axial attention on rearranged frames |

The curve-based aggregation uses **relative encoding** so points sitting in the same local geometry do **not** collapse to identical features (a known failure of local/non-local aggregation). Distinct points stay distinguishable, which helps the network generalize.

---

## 🏗️ Pipeline

```mermaid
flowchart LR
    A[RGBD / Kinect<br>joint coordinates] --> B[Skeleton<br>point cloud]
    B --> C[Curve-based<br>point aggregation<br>geometry-aware]
    C --> D[Positional<br>encoding]
    D --> E[Axial self-attention<br>spatial + temporal]
    E --> F[Transformer<br>layers]
    F --> G[Quality score<br>0 – 1]
    E -.joint importance.-> H[Interpretable<br>feedback]
```

The model accepts **variable-length sequences** with no temporal alignment or segmentation, and outputs a normalized score (0–1) representing kinematic similarity to an expert execution.

---

## 📊 Results

Evaluated on three standard benchmarks: **KIMORE**, **UI-PRMD**, and **IRDS**.

### UI-PRMD — Mean Absolute Deviation (MAD, lower is better)

| Ex | **Ours** | Mourchid | Deb | Song | Liao | Deep CNN |
|----|----------|----------|------|------|------|----------|
| 1  | **0.008** | 0.008 | 0.009 | 0.011 | 0.011 | 0.014 |
| 2  | **0.005** | 0.010 | 0.006 | 0.006 | 0.028 | 0.029 |
| 3  | **0.006** | 0.010 | 0.013 | 0.010 | 0.039 | 0.041 |
| 4  | **0.006** | 0.008 | 0.006 | 0.014 | 0.012 | 0.016 |
| 5  | **0.006** | 0.007 | 0.008 | 0.013 | 0.019 | 0.013 |
| 6  | **0.006** | 0.010 | 0.006 | 0.009 | 0.018 | 0.023 |
| 7  | **0.010** | 0.020 | 0.011 | 0.017 | 0.038 | 0.033 |
| 8  | **0.014** | 0.020 | 0.016 | 0.017 | 0.023 | 0.029 |
| 9  | **0.008** | 0.014 | 0.008 | 0.008 | 0.023 | 0.025 |
| 10 | **0.018** | 0.015 | 0.031 | 0.038 | 0.042 | 0.037 |

### KIMORE — our method, per exercise

| Metric (↓) | Ex1 | Ex2 | Ex3 | Ex4 | Ex5 |
|------------|------|------|------|------|------|
| MAD  | 0.185 | 0.560 | 0.128 | 0.256 | 0.388 |
| RMSE | 0.591 | 1.235 | 0.233 | 0.451 | 0.678 |
| MAPE | 0.543 | 1.891 | 0.336 | 0.766 | 1.199 |

### IRDS — accuracy (higher is better)

| Model | **Ours** | Zheng (Baseline) | Zheng (R.I.) | Zhang | Li |
|-------|----------|------------------|--------------|-------|-----|
| Mean accuracy | **0.9819** | 0.9751 | 0.9741 | 0.9680 | 0.9720 |

### ⚡ Efficiency — inference time on KIMORE Ex5 test set (75 instances / 7,500 frames)

| Model | GPU time (s) | CPU time (s) |
|-------|--------------|--------------|
| STGCN | 13.52 | 85.68 |
| D-STGCNT | 2.05 | 23.47 |
| **Ours** | **1.09** | **10.08** |

**~12.4× faster** than the STGCN baseline on GPU. On CPU that is **~1.34 ms/frame** (~134 ms per exercise video), well under the 33.3 ms/frame budget at 30 fps, so it sustains real-time throughput with **~25× headroom and no specialised hardware**. On an RTX 3070 Ti it drops to **~0.15 ms/frame**.

### 📡 Remote-monitoring footprint

Transmits only skeletal coordinates (25 joints × 3 × 4 bytes = **300 bytes/frame**), so the stream is **~72 kbps**, with an estimated **~50–70 ms** end-to-end latency. Easily fits a typical home broadband uplink.

---

## 📁 Repository structure

```
Transformer_Rehabilitation/
├── core/                  # CurveNet — curve-based point aggregation
├── Data_Proc/             # Dataset loading & preprocessing (KIMORE etc.)
├── Images/                # Figures & diagrams
├── Rehabilitation.ipynb   # Training, inference, validation, model architecture
├── Rehab_T-SNE.ipynb      # t-SNE visualization of learned embeddings
└── README.md
```

---

## 🚀 Getting started

### Dependencies

```bash
python >= 3.7
pytorch  +  cudatoolkit >= 11.2   # may work on lower
```

```bash
git clone https://github.com/King-Rafat/Transformer_Rehabilitation.git
cd Transformer_Rehabilitation
pip install torch numpy scipy scikit-learn matplotlib jupyter
```

### Datasets

| Dataset | Joints | Sensor | Task | Official source | Processed splits |
|---------|--------|--------|------|-----------------|------------------|
| **KIMORE** | 18 | Kinect v2 | Quality score (regression) | [vrai.dii.univpm.it](https://vrai.dii.univpm.it/content/kimore-dataset) | _TODO: add link_ |
| **UI-PRMD** | 22 | Vicon / Kinect | Quality score (regression) | [webpages.uidaho.edu/ui-prmd](https://webpages.uidaho.edu/ui-prmd/) | _TODO: add link_ |
| **IRDS** | 25 | Kinect v2 | Correct / incorrect (classification) | [MDPI Data 6(5):46](https://www.mdpi.com/2306-5729/6/5/46) | _TODO: add link_ |

Download each dataset from the official source above and point the loaders in `Data_Proc/` to your local paths. The exact train/val/test splits used in the paper can be downloaded from the **Processed splits** column (drop in a Google Drive / Hugging Face / Zenodo link once uploaded).

### 💾 Pretrained checkpoints

All pretrained weights (KIMORE, UI-PRMD, IRDS) are available here:

**➡️ [Download checkpoints (Google Drive)](TODO: add Drive link)**

Download and place them under `checkpoints/`.

### Run

Open the main notebook and run the cells end to end:

```bash
jupyter notebook Rehabilitation.ipynb
```

It contains the model architecture, the training loop, inference, validation, and the visualizations. Use `Rehab_T-SNE.ipynb` to reproduce the embedding clusters (accurate / moderate / inaccurate).

---

## 🙏 Acknowledgements

This work builds on several open repositories:

- Curve-based point aggregation: [CurveNet](https://github.com/tiangexiang/CurveNet)
- Loading KIMORE: [KiMoRe_wrapper](https://github.com/petteriTeikari/KiMoRe_wrapper)
- Processing KIMORE: [STGCN-rehab](https://github.com/fokhruli/STGCN-rehab)
- Loading & processing UI-PRMD: [A-Deep-Learning-Framework](https://github.com/avakanski/A-Deep-Learning-Framework-for-Assessing-Physical-Rehabilitation-Exercises)

---

## 📚 Citation

If you use this code, model, or dataset splits, please cite:

```bibtex
@article{rafat2026pointcloud,
  title   = {A Point Cloud Transformer for Remote Monitoring and Automated
             Assessment of Physical Rehabilitation Exercises},
  author  = {Rafat, Kazi and Hossain, Md. Ismail and Elahi, M M Lutfe and
             Momen, Sifat and Rahman, Fuad and Mohammed, Nabeel and Rahman, Shafin},
  journal = {IEEE Journal of Biomedical and Health Informatics (JBHI)},
  year    = {2026},
  note    = {Manuscript ID: JBHI-04660-2025}
}
```

> Update the year, volume, pages, and DOI once the final IEEE Xplore entry is live.

---

<div align="center">
<sub>Maintained by <a href="https://github.com/King-Rafat">King-Rafat</a> · Questions? Open an <a href="https://github.com/King-Rafat/Transformer_Rehabilitation/issues">issue</a>.</sub>
</div>
