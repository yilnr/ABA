# Rebalancing Multimodal Learning via Adaptive Bidirectional Alignment

**Adaptive Bidirectional Alignment (ABA)**  
*A space-aware and status-aware framework for rebalanced multimodal learning*
[![Task](https://img.shields.io/badge/Task-Multimodal%20Learning-orange)]()

---

## 📌 Overview

Multimodal learning is expected to benefit from complementary information across heterogeneous modalities. However, in practice, different modalities often exhibit **asynchronous learning dynamics**, where a strong modality dominates optimization while a weak modality remains insufficiently learned. This phenomenon is known as **modality imbalance**. To address this problem, we propose **Adaptive Bidirectional Alignment (ABA)**, a rebalanced multimodal learning framework that performs cross-modal alignment in a **space-aware** and **status-aware** manner.


| Key Idea                              | Description                                                  |
| ------------------------------------- | ------------------------------------------------------------ |
| **Space-aware alignment**             | Reduce feature-space discrepancy before cross-modal alignment |
| **Status-aware alignment**            | Selectively align transferable components according to modality learning status |
| **Adaptive bidirectional regulation** | Dynamically control alignment scope and direction            |

---

## ✨ Highlights

- 🔹 **Class-Conditional Statistical Projection**  
  Constructs class-specific cross-modal mappings to reduce modality-space heterogeneity.

- 🔹 **Feature-Level Partial Optimal Transport**  
  Selectively aligns reliable and transferable feature components instead of enforcing full-space alignment.

- 🔹 **Adaptive Bidirectional Alignment**  
  Dynamically determines transported mass and directional weights based on modality reliability.

- 🔹 **Strong Generalization Across Modalities**  
  Supports audio-video, image-text, and multi-modality scenarios.

- 🔹 **Extensive Evaluation**  
  Validated on six widely used multimodal benchmarks.

---

## 🧠 Method

ABA contains three main modules:

### 1. Class-Conditional Statistical Projection

For each class and each modality, we estimate class-wise statistics and construct a Gaussian-based affine mapping between heterogeneous modality spaces.

This provides a more comparable projected space for subsequent alignment.

### 2. Partial Optimal Transport Alignment

Instead of aligning all feature dimensions, ABA formulates alignment as a feature-level partial optimal transport problem. This allows the model to align only reliable and transferable feature components.

### 3. Adaptive Bidirectional Regulation

ABA decouples two important factors:

| Component              | Role                                                  |
| ---------------------- | ----------------------------------------------------- |
| **Transported mass**   | Determines how much information should be aligned     |
| **Directional weight** | Determines which alignment direction contributes more |

The final objective is:

```math
\mathcal{L} = \mathcal{L}_{cls} + \beta \mathcal{L}_{align}
```
