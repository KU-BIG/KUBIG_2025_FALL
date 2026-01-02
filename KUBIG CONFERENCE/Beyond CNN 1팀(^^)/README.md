# Dynamic Convolution Neural Network

This repository presents a study on **dynamic and shift-based convolutional operations** designed to overcome the locality and limited effective receptive field (ERF) of conventional CNNs.  
The project explores lightweight, adaptive alternatives to large-kernel convolutions and attention-based models, with a focus on efficiency and robustness.

---

## Motivation

Convolutional Neural Networks (CNNs) exhibit strong inductive bias toward **locality**, which has driven their success in vision tasks.  
However, this same property leads to a **limited Effective Receptive Field (ERF)**, even in deep architectures.

Recent works such as **Vision Transformers**, **ConvNeXt**, and **large-kernel CNNs** address this limitation, but often at the cost of:
- Increased computational complexity
- Memory overhead
- Reduced suitability for edge devices

This project aims to:
- Expand and diversify ERF **without large kernels or self-attention**
- Maintain efficiency suitable for lightweight and edge-oriented models
- Introduce **dynamic, shift-based inductive biases** into CNNs

---

## Key Ideas

### 1. Jittering-based Convolution
We introduce **feature-level jittering** before convolution to perturb spatial alignment and encourage robustness.

Variants explored:
- Random integer pixel shifts
- Deterministic shifts by channel index
- Sub-pixel (0.5-pixel) shifts via interpolation
- Learnable shift parameters per channel

**Goal:**  
Increase ERF coverage while preserving CNN locality.

---

### 2. Shiftwise & Rotational Convolution
Shiftwise convolution approximates large kernels using multiple shifted small kernels.

We extend this idea by:
- Introducing **diagonal shifts**
- Applying **rotational operations** to diversify ERF distribution
- Aggregating shifted features via summation and pointwise convolution

**Insight:**  
While standard shiftwise convolution expands ERF along horizontal/vertical axes, rotation enables **diagonal and anisotropic dependency modeling**, which is beneficial for texture- and pattern-heavy data.

---

### 3. Pixel-level Mix via MLP
For extremely lightweight networks (e.g., edge devices, super-resolution models), receptive fields quickly shrink due to small kernels and shallow depth.

We propose a **Pixel-level Mix** module that:
- Extracts local feature vectors
- Shifts and concatenates neighboring pixel features
- Applies MLP-based mixing with positional encoding

This enables:
- Pixel-to-pixel interaction without attention
- Preservation of locality
- Transformer-like relational modeling at minimal cost

---

## Methods Overview

- **Backbone:** ResNet-18 (for classification experiments)
- **Datasets:**
  - CIFAR-100 (classification)
  - DTD (zero-shot texture recognition)
  - DIV2K / Set5 / Set14 (super-resolution)
- **Training Setup:**
  - Identical architecture and hyperparameters across variants
  - Multiple random seeds for fair comparison

---

## Experimental Results

### Classification (CIFAR-100)
- Jittering significantly **expanded ERF**
- However, naive jittering degraded accuracy due to instability
- Learnable and structured shifts showed more stable behavior

### Texture Recognition (DTD)
- Models with **diagonal + rotational shift operations** achieved the best performance
- ERF diversity (not just size) was critical for texture datasets

### Super-Resolution
- Shift-based dynamic modules consistently improved PSNR
- Performance gains observed without increasing kernel size or depth

---

## Key Findings

- Expanding ERF alone is insufficient; **ERF diversity matters**
- Shift + rotation operations offer an efficient alternative to large kernels
- Pixel-level dynamic mixing enables relational learning in lightweight CNNs
- Dynamic convolutional inductive biases can bridge the gap between CNNs and attention models

---

## Conclusion

This work demonstrates that:
- CNN limitations can be alleviated through **dynamic, geometry-aware operations**
- Large kernels and attention are not the only paths to global context
- Carefully designed shift, rotation, and pixel-mixing strategies yield strong performance with minimal overhead

---

## Acknowledgement

This project was developed as part of an academic study on advanced convolutional architectures.

---

## License

This repository is intended for research and educational purposes.
