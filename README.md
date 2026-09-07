# MinimaxH3-V100-ComfyUI

Run the Minimax H3 model in FP16 precision on GPUs without native BF16 tensor core support (such as Volta/V100 and Turing/RTX 20-series) in ComfyUI without encountering NaN latent errors or black video outputs.

---

## Overview

When running Minimax H3 on older GPU architectures, standard settings present a severe trade-off between generation speed and output quality:

* **Default (BF16 Emulated → FP32 Fallback):** Produces correct outputs, but execution is extremely slow.
* **Standard FP16 (`ModelComputeDtype = fp16`):** Runs fast, but results in `NaN` values in the latent space, producing a completely black video output.

**MinimaxH3-V100-ComfyUI** fixes the FP16 numeric instabilities, allowing you to get correct video outputs at full FP16 speed—a **~6.7× performance increase** over the default fallback.

---

## Performance Comparison

*Benchmark performed on **Tesla V100-SXM2 32GB** (5-second Text-to-Video at 0.4 MP resolution):*

| Setting / Node | Speed | Output Result | Speedup |
| :--- | :--- | :--- | :--- |
| **Default** (BF16 emulated → FP32) | 94 sec/it | Correct output | 1.0× (Baseline) |
| **Standard FP16** (`ModelComputeDtype = fp16`) | ~14 sec/it | ❌ Black video (`NaN` in latent) | N/A |
| **This Node** | **~14 sec/it** | **✅ Correct output** | **~6.7× Faster** |

---

## Compatibility

* **Verified:** NVIDIA Tesla V100 (Volta architecture)
* **Likely Compatible (Untested):** NVIDIA Turing GPUs (RTX 20 series, T4)

---

## Usage

Insert this node **directly right before** your `BasicGuider` node in the default ComfyUI workflow:
