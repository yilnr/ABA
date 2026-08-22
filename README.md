# Rebalancing Multimodal Learning via Adaptive Bidirectional Alignment

Official PyTorch implementation of **Adaptive Bidirectional Alignment (ABA)**, accepted by the **35th ACM International Conference on Information and Knowledge Management (CIKM 2026)**.

**Yuhan Zheng** and **Yang Yang**
Nanjing University of Science and Technology

[[Paper](https://doi.org/10.1145/3799682.3840700)] [[Code](https://github.com/yilnr/ABA)]

## Overview

Multimodal joint training can underperform unimodal training because heterogeneous modalities learn at different rates. Existing cross-modal alignment methods are often static, symmetric, and fully coupled, which may force unreliable or underdeveloped features to interact.

ABA follows two principles:

- **Space-aware alignment:** reduce distributional and geometric discrepancies before aligning heterogeneous modality features.
- **Status-aware alignment:** align only reliable, transferable components according to the evolving learning status of each modality.

The framework contains three components:

1. **Class-conditional statistical projection** estimates class-wise Gaussian statistics with exponential moving averages and constructs bidirectional affine transport maps between modality spaces.
2. **Feature-level partial optimal transport** selectively aligns transferable feature components in the projected space instead of enforcing full-space alignment.
3. **Adaptive bidirectional regulation** decouples transported mass from directional weight, controlling both how much information is transferred and which modality provides stronger guidance.

For a source modality \(p\), target modality \(q\), and class \(k\), the transported mass is

```math
m^{(p\rightarrow q,k)} = \rho\, r^{(p,k)}\left(1-r^{(q,k)}\right)
\exp\left(-\frac{\bar{C}^{(p\rightarrow q,k)}}{\kappa}\right),
```

while the directional weight is determined by the relative unimodal reliability:

```math
\lambda^{(p\rightarrow q,k)} =
\frac{\exp\left(r^{(p,k)}/\nu\right)}
{\exp\left(r^{(p,k)}/\nu\right)+\exp\left(r^{(q,k)}/\nu\right)}.
```

The final objective is

```math
\mathcal{L}=\mathcal{L}_{\mathrm{cls}}+\beta\mathcal{L}_{\mathrm{align}}.
```

## Main results

ABA was evaluated on six multimodal benchmarks in the paper.

| Dataset | Metric 1 | Metric 2 |
| --- | ---: | ---: |
| KineticsSounds | ACC 74.91 | MAP 81.39 |
| CREMA-D | ACC 85.37 | MAP 91.77 |
| VGGSound | ACC 55.53 | MAP 58.94 |
| IEMOCAP | ACC 79.67 | MAP 80.47 |
| NVGesture | ACC 85.26 | F1 85.37 |
| Twitter | ACC 75.23 | F1 69.53 |

On KineticsSounds, the complete method improves over the core baseline from **70.32 ACC / 78.53 MAP** to **74.91 ACC / 81.39 MAP**. It also remains more robust under test-time modality missing:

| Missing rate | 10% | 20% | 30% | 50% |
| --- | ---: | ---: | ---: | ---: |
| ABA accuracy | 72.43 | 66.42 | 62.83 | 57.93 |

## Current code release

This repository currently provides the main KineticsSounds experiment.

```text
.
├── train_KS_multimodal_ABA_paper.py   # ABA training entry point
├── data/
│   ├── kinetics_sound.json            # KS configuration
│   └── template.py                    # shared defaults
├── dataset/KS.py                      # KS dataset loader
├── model/AudioVideo.py                # audio-video classifier
├── model/Resnet.py                    # modality encoders
└── utils/                              # training utilities
```

The required source archives have been unpacked into regular directories. No ZIP extraction step is required.

## Data preparation

Set `dataset.data_root` in `data/kinetics_sound.json` to your local KineticsSounds directory. The current loader expects the following structure:

```text
kinetics_sound/
├── annotations/
│   ├── train.csv
│   ├── test.csv
│   └── weight.csv
├── train_img/Image-01-FPS/<youtube_id>/...
├── test_img/Image-01-FPS/<youtube_id>/...
├── train_wav/<youtube_id>.wav
└── test_wav/<youtube_id>.wav
```

Each annotation CSV must contain `youtube_id` and `label` columns.

## Environment

The main experiment requires Python and the following core packages:

- PyTorch, TorchVision, and TensorBoard
- NumPy, pandas, and scikit-learn
- librosa and Pillow
- tqdm

The camera-ready experiments use ResNet-18 encoders for both audio and video. Audio is converted to a \(257\times1004\) spectrogram, and three frames are sampled from each video clip.

## Training

The paper-selected ABA parameters on KineticsSounds are \(\beta=0.6\), \(\rho=0.75\), \(\nu=0.45\), and \(\kappa=0.4\). In the implementation, `aba_tau_lambda` corresponds to \(\nu\), while `aba_tau_c` corresponds to \(\kappa\).

```bash
python train_KS_multimodal_ABA_paper.py \
  --config data/kinetics_sound.json \
  --aba_beta 0.6 \
  --aba_rho 0.75 \
  --aba_tau_lambda 0.45 \
  --aba_tau_c 0.4
```

The camera-ready protocol trains the audio-video models with SGD, an initial learning rate of \(10^{-2}\), momentum 0.9, weight decay \(10^{-1}\), and a batch size of 256. Adjust the batch size when GPU memory is limited.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{zheng2026aba,
  author    = {Yuhan Zheng and Yang Yang},
  title     = {Rebalancing Multimodal Learning via Adaptive Bidirectional Alignment},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management},
  year      = {2026},
  doi       = {10.1145/3799682.3840700}
}
```

## Acknowledgments

This work was supported by the NSFC (62276131), the Natural Science Foundation of Jiangsu Province of China (BK20240081), and the Special Research Project on Teaching Reform of General Artificial Intelligence Courses in Jiangsu Undergraduate Universities (ZNT-10).
