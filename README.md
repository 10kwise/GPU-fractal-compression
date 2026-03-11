# GPU Fractal Image Compression

> GPU-accelerated partitioned iterated function system (PIFS) image compression with adaptive encoding, compact binary colour coding, and resolution-independent decoding.

**LEWIS KAGIRI** · School of Informatics and Computing · Technical Univesity of Mombasa

---

## What This Is

This system compresses images by describing them as a set of self-similarity rules rather than storing pixels directly. Each region of the image is matched to a similar region elsewhere in the same image. The stored rules are far smaller than the original pixels, and decoding applies the rules iteratively until the image converges — guaranteed by Banach's contraction theorem.

The primary differentiating property over standard codecs (JPEG, WebP) is **resolution-independent decoding**: the same compressed file decodes at 1×, 2×, or 4× resolution without any additional stored data. This makes it suitable for applications where multiple output resolutions are needed from a single source file.

The encoding search — finding the best matching region for each image block — is embarrassingly parallel and maps directly to GPU matrix operations. This system implements the full pipeline in Python/CuPy, reducing encoding time from hours (1990s CPU implementations) to under 15 seconds on an NVIDIA T4 GPU.

---

## Results

Tested on five image types representing different levels of self-similarity.

| Image Type | PSNR | SSIM | File Size | Compression | Notes |
|---|---|---|---|---|---|
| Street movie scene / portrait | 38.23 dB | 0.983 | 75.7 KB | 97.5% | From 3.0 MB raw |
| Architectural render | 40.57 dB | 0.990 | 12.8 KB | 97.8% | Smooth gradients |
| CCTV footage (grayscale) | 26.01 dB | 0.880 | 112.6 KB | 95.8% | Fisheye distortion |
| Brick wall | 25.48 dB | 0.672 | 239.9 KB | 93.6% | Lighting gradient |
| Forest canopy | 20.69 dB | 0.715 | 3418.2KB | 94.4% | Known failure case |

PSNR above 35 dB is considered good quality for natural photographs. The system performs well on smooth-content images (portraits, architecture, gradients) and fails predictably on high-entropy content (foliage, fine texture) — a failure mode that is theoretically characterised in the paper.

---

## Visual Results

### Compression — Street Scene

<!-- Replace the placeholder below with your side-by-side image -->
<!-- Suggested: original (left) vs v7 reconstruction (right), labelled with PSNR and file size -->

![street movie scene](./assets/street%20movie%20scene.png)

---

### Resolution-Independent Decoding — Same File, Scale to 2x and 4x

<!-- Replace the placeholder below with your upscale comparison image -->
<!-- Suggested: three crops at 1×, 2×, 4× from the same .frac file, side by side -->

![street movie scene](./assets/leaf%20upscale%20factor%202.png)
![street movie scene](./assets/leaf%20upscale%20factor%204.png)

---

### Failure Case — Aerial view of a city

<!-- Replace the placeholder below with your failure case image -->
<!-- Suggested: original canopy (left) vs reconstruction (right) showing the blur effect -->

![street movie scene](./assets/aerial%20view%20of%20london.png)

---

## Key Technical Findings

**Why fractal compression blurs — and when it does not.**
The mandatory 2× downsampling of domain blocks during encoding is a lowpass filter. Every decode iteration applies this filter, causing sharpness to decay by a factor of 4× per iteration. After 10 iterations, sharpness is 10⁻⁶ of the original. Images dominated by smooth gradients (portraits, sky, architectural surfaces) contain little high-frequency content to lose. Images with fine detail (foliage, fabric, brick texture) are dominated by high frequencies that this filter destroys. This analysis correctly predicts performance across all five benchmark images and is the primary theoretical contribution of this work.

**Colour channel overhead eliminated.**
Prior implementations stored DCT colour coefficients as Python objects, adding ~28× serialisation overhead (415 KB for data containing ~15 KB of information). A compact variable-length binary codec stores only the nonzero zigzag coefficients per block, zlib-compressed. Colour storage reduced from 415 KB to 10 KB with no change in reconstruction quality.

**Bugs identified and corrected in the binary format.**
Three defects were found and fixed: a signed 8-bit brightness field that clipped bright image regions (fixed to unsigned [0,255]); a 5-bit coordinate field that overflowed on images taller than 2016 pixels (expanded to 6-bit, supporting 4048 px); and a variance-grouping optimisation that provided zero speedup due to grouping by the wrong criterion (removed).

---

## How to Run

The system runs entirely in Google Colab — no local GPU required.

**1. Open the notebook**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/10kwise/gpu-fractal-compression/blob/main/notebooks/fractal_compress_v7_2.ipynb)

<!-- Replace [YOUR-USERNAME] with your actual GitHub username in the link above -->

**2. Set your image path and time budget**

```python
IMAGE_PATH          = '/content/your_image.jpg'   # path to your image in Colab
TIME_BUDGET_SECONDS = 120                          # encoding time limit
COMPRESSION_MODE    = 'LOSSY'                      # 'LOSSY' or 'LOSSLESS'
ENCODE              = True                         # True to encode, False to decode only
```

**3. Run all cells**

The pipeline runs automatically: analysis → pre-sample diagnostic → GPU encode → save `.frac` file → decode → quality metrics.

**Requirements** (installed automatically in Colab):

```
cupy-cuda12x
scipy
Pillow
numpy
```

---

## Project Structure

```
gpu-fractal-compression/
├── notebooks/
│   └── fractal_compress_v7.ipynb   # Main notebook — full pipeline
├── assets/
│   └── [benchmark images]          # Original and reconstructed images
├── README.md
├── LICENSE                         # GPL-3.0
└── CITATION.cff
```

---

## File Format

Compressed images are saved as `.frac` files (version 7). Each transform is packed into 4 bytes:

```
[ cy : 6 bits ][ cx : 6 bits ][ ti : 3 bits ][ contrast : 8 bits ][ brightness : 8 bits ]
```

Supports images up to 4048 × 4048 pixels at maximum domain step. Colour channels stored as variable-length binary DCT + zlib. The format is fully documented in the notebook and in the paper (see Citation).

---

## Limitations

This system is not a general replacement for JPEG or WebP. JPEG achieves comparable quality at 3–4× smaller file sizes on most images. The correct framing is:

- **Use fractal** when resolution-independence matters — multiple output resolutions from one file, game engine texture streaming, super-resolution pipelines on smooth content.
- **Use JPEG/WebP** for general-purpose image compression where file size efficiency is the primary goal.
- **Do not use fractal** on high-entropy content (foliage, fine fabric, complex textures) — the lowpass filter characteristic will produce visibly blurred results regardless of settings.

The pre-sample diagnostic at the start of each encode predicts this automatically and warns when an image is a poor candidate.

---

## Roadmap

- [x] GPU-accelerated PIFS encoder (v7)
- [x] Compact binary DCT colour codec
- [x] Adaptive auto-threshold via pre-sampling
- [x] Resolution-independent 2× and 4× decoder
- [x] Corrected binary .frac file format
- [ ] Rigorous upscaling benchmark vs bicubic / Lanczos / EDSR
- [ ] CNN sharpness recovery decoder (hybrid fractal-CNN pipeline)
- [ ] arXiv preprint
- [ ] Unity / Godot game engine texture plugin
- [ ] Video temporal compression (cross-frame domain search)

---

## Citation

If you use this work, please cite:

```bibtex
@software{Ndegwa_2026_gpu_fractal,
  author    = {Lewis Kagiri Ndegwa},
  title     = {GPU Fractal Image Compression},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/10kwise/GPU-fractal-compression}
}
```

A full academic citation will be available once the accompanying paper is published. See `CITATION.cff` for machine-readable citation metadata.

---

## Licence

This project is licensed under the **GNU General Public License v3.0**.

You are free to use, study, modify, and distribute this work. Any derivative work must also be distributed under the same licence. See [`LICENSE`](LICENSE) for the full terms.

© [2026] [Lewis Kagiri Ndegwa]
