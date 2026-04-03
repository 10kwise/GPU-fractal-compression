# GPU-Accelerated Fractal Image Compression with Adaptive Encoding, Compact Colour Coding, and Resolution-Independent Upscaling

**A complete Python/CuPy implementation of Partitioned Iterated Function System (PIFS) fractal image compression with resolution-independent decoding, validated on the Kodak-24 benchmark dataset.**

Lewis Kagiri Ndegwa 
School of Informatics and Computing, Technical University of Mombasa  
GPL-3.0 License · March 2026

---

## Abstract / What this is

This system encodes images as collections of contractive affine mappings between self-similar regions (PIFS). The stored file decodes at any integer scale factor from the same compressed file with no additional stored data — a property no interpolation-based codec can match. It incorporates **automatic parameter selection**, a **stratified pre-sample diagnostic**, and a **compact binary DCT colour encoder**. 

The core contribution of v7.3 is a **batched GPU encoder** that eliminates the per-block Python loop entirely. All M non-flat range blocks are stacked into a single matrix R ∈ ℝ^(M×64) and transferred to GPU in one operation. The full (M, N) error surface is computed analytically from one batched matrix multiplication R @ D^T, with argmin over N giving the globally optimal transform for all M blocks simultaneously.

---

## Key results

### Compression

| Image | Resolution | PSNR | SSIM | File Size | vs Raw |
|-------|-----------|------|------|-----------|--------|
| Portrait photography | 1366×911 | 37.03 dB | 0.963 | 103.4 KB | 97.2% |
| Architectural render | 626×313 | 40.57 dB | 0.990 | 12.8 KB | 97.8% |
| Aerial footage | 1280×720 | 26.02 dB | 0.880 | 112.6 KB | 95.8% |
| Brick wall | 1500×844 | 25.48 dB | 0.672 | 239.9 KB | 93.6% |
| Forest canopy | 999×666 | 20.70 dB | 0.727 | 172.1 KB | 91.3% |

**Kodak-24 dataset:** mean PSNR 26.61 ± 2.61 dB, mean SSIM 0.791 ± 0.077, compression 92.7–95.7% across 24 images.

### Visual Results

![Portrait photography](assets/portrait%20scene.png)  
*Fig 2a. Portrait photography — original (left) vs. decoded (right). PSNR 37.03 dB, SSIM 0.963, 103.4 KB (97.2% compression).*

![Aerial footage](assets/aerial%20view%20of%20london.png)  
*Fig 2c. Aerial footage — original (left) vs. decoded (right). PSNR 26.02 dB, SSIM 0.880, 112.6 KB (95.8% compression).*

![Brick wall](assets/brick%20wall.png)  
*Fig 2d. Brick wall — original (left) vs. decoded (right). PSNR 25.48 dB, SSIM 0.672, 239.9 KB (93.6% compression).*

### Resolution-independent upscaling

The same `.frac` file decodes at 1×, 2×, 4×, 8×, or 10× with no additional stored data. Validated across 10 Kodak images against bicubic and Lanczos at equal storage budget (round-trip methodology):

| Scale | Fractal wins | Mean Δ PSNR |
|-------|-------------|-------------|
| 2× | 0/10 | −0.89 dB |
| 4× | 10/10 | +2.46 dB |
| 8× | 10/10 | +4.52 dB |
| 10× | 10/10 | +5.02 dB |

Crossover between 2× and 4× is consistent across all 10 images without exception. Fractal quality is stable from 4× onward (varies < 0.04 dB from 4× to 10×). Interpolation degrades monotonically as the storage budget for the downscaled baseline becomes insufficient.

![Forest canopy upscale 2x](assets/leaf%20upscale%20factor%202.png)  
![Forest canopy upscale 4x](assets/leaf%20upscale%20factor%204.png)  
*Forest canopy upscaled at 2× and 4× using fractal decoding.*

### Encode speed (NVIDIA T4, v7.3 batched encoder)

| Image | Encode time |
|-------|------------|
| Portrait 1366×911 | 0.7 s |
| Arch render 626×313 | 0.2 s |
| Aerial 1280×720 | 0.5 s |
| Brick wall 1500×844 | 0.7 s |
| Forest canopy 999×666 | 1.3 s |

---

## Architecture

```
RGB Input
    │
    ▼
Pad to 8px multiple → RGB to YCbCr
    │
    ▼
Image Analysis
  · block variance across all 8×8 range blocks
  · flat block detection (bottom 5th percentile, capped at variance=50)
  · τ = 0.50 × median(non-flat block variance)   ← post-hoc classifier, not search terminator
  · auto domain step from image dimensions
    │
    ▼
Pre-sample Diagnostic
  · 5% stratified sample across low/mid/high variance buckets
  · estimates match success rate before full encode
    │
    ▼
Build Domain Stack  D ∈ ℝ^(N×64)
  · all domain positions × 8 D4 transforms (4 rotations × horizontal flip)
  · single host→GPU transfer
    │
    ▼
Batched GPU Encode Y channel
  · R ∈ ℝ^(M×64) — all non-flat range blocks, single transfer
  · errors = R @ D^T  →  analytical c, b for all (M,N) pairs
  · argmin over N → globally optimal transform per block
  · contractivity clamp: c ∈ [−0.75, 0.75]
    │
    ▼
Compact DCT Encode Cb, Cr
  · 8×8 DCT with JPEG chroma quantisation table
  · variable-length zigzag prefix + zlib compression
  · 4–14 KB per image; 0 KB for greyscale content
    │
    ▼
Write .frac v7 binary
  · 4 bytes per transform: [cy:6][cx:6][ti:3][contrast_q:8][brightness:8]
  · file size = n_blocks × 4 + colour streams  (invariant to domain step)
    │
    ▼
Decode at any scale
  · IFS iterate 10× from flat grey initial image
  · decode time scales sublinearly with output area
```

---

## Design properties verified empirically

**τ is a post-hoc classifier, not a quality control parameter.** The encoder always evaluates all N domain candidates and stores the global argmin regardless of threshold value. τ only classifies whether the optimal match fell below threshold for the match success rate counter. PSNR is invariant to the threshold multiplier k across k ∈ {0.1…0.8} — confirmed on three images with varying content class.

**File size is invariant to domain step.** The output stores exactly one 4-byte transform per 8×8 range block regardless of search density. Domain step controls how many candidates are evaluated, not how many transforms are stored. Halving the candidate pool (step 16→32) costs 0.62–0.69 dB PSNR at no file size penalty across four content classes.

**Residual second-pass encoding does not improve quality.** Evaluated across 24 Kodak images: mean ΔPSNR = −0.002 dB. The post-encoding error lacks exploitable self-similar structure; the second pass finds no useful matches in the same domain pool. The feature is retained in the codebase as it corrects localised grey-patch artefacts in a small number of high-contrast blocks.

![Residual fix on portrait](assets/encode%20residual%20on%20portrait.png)  
*Visual inspection isolated high-contrast blocks where the residual pass occasionally corrected a flat grey patch artefact.*

---

## Installation

```bash
# Google Colab (recommended — T4 GPU included)
!pip install cupy-cuda12x scipy scikit-image -q

# Local with CUDA 12
pip install cupy-cuda12x scipy scikit-image pillow
```

No other dependencies. Falls back to NumPy (CPU) automatically if CuPy is unavailable.

---

## Usage

### Encode

```python
IMAGE_PATH  = '/content/your_image.png'
OUTPUT_PATH = '/content/output.frac'
ENCODE      = True

# Run Cell B (Main Pipeline) in fractal_compress_v7_3.ipynb
```

### Decode at any scale

```python
# Standard decode (1×)
y_rec = decode_fractal(y_tf, (H_pad, W_pad), n_iterations=10)

# Upscale decode (same file, no extra data)
recon_4x = decode_and_upscale(y_tf, cb_enc, cr_enc,
                               padded_shape, orig_shape,
                               scale=4, n_i=12)
```

### Key configuration (Cell A3)

```python
AUTO_THRESHOLD      = True    # derive τ from image variance automatically
TIME_BUDGET_SECONDS = 120     # influences auto domain step selection
ENCODE_RESIDUAL     = True    # second pass (ineffective on average — see paper)
DCT_QUALITY_FACTOR  = 1.0     # colour encoder quality (lower = larger file)
```

---

## Notebook structure

| Cell | Purpose |
|------|---------|
| A1 | Setup — pip install, GPU check |
| A2 | Imports and GPU detection |
| A3 | Configuration |
| A4 | YCbCr colour space and padding |
| A5 | Fractal helpers (downsample_2x, D4 transforms) |
| A6 | Image analysis and adaptive parameter selection |
| A7 | Pre-sample diagnostic |
| A8 | Batched GPU encoder (v7.3) |
| A9 | Fractal decoders (standard + upscale) |
| A10 | Residual encoding (gated) |
| A11 | Compact binary DCT colour encoder |
| A12 | Save / Load (.frac v7 binary format) |
| A13 | Utilities (metrics, display, size report) |
| B | Main pipeline — encode or decode a single image |
| C1 | Upscaling comparison (round-trip evaluation) |
| C2 | JPEG comparison at matched file size |
| C3 | Rate-distortion benchmark |
| D1 | Kodak-24 dataset evaluation |
| D2 | Threshold multiplier sweep |
| D3 | Residual gate justification |
| D4 | Dataset upscaling comparison (Kodak subset) |
| D5 | Domain step sweep with decode timing |

---

## .frac v7 binary format

```
Header (29 bytes):
  4B  magic: 'FRAC'
  4B  version: 7
  4B  padded H
  4B  padded W
  4B  original H
  4B  original W
  4B  n_transforms
  4B  domain_step
  1B  colour_mode (0=LOSSY, 1=LOSSLESS)
  1B  has_residual

Per transform (4 bytes):
  [cy:6][cx:6][ti:3][contrast_q:8][brightness:8]
  cy, cx: domain block index (not pixel coordinate)
  ti:     D4 transform index 0–7
  contrast_q: stored as uint8 via cq + 127
  brightness: uint8 in [0, 255]

Colour streams:
  cb_blob (length stored in header)
  cr_blob (length stored in header)
```

File size = 29 + n\_blocks × 4 + len(cb\_blob) + len(cr\_blob)  
where n\_blocks = (H\_pad ÷ 8) × (W\_pad ÷ 8), invariant to domain step.

---

## Known limitations

- **Quality on high-entropy content.** Images with median block variance > 400 (dense foliage, crowds, complex textures) average 22.4 dB PSNR. The domain pool contains no structurally similar candidates for fine texture; this is an attractor mismatch, not a search failure. A denser domain pool at step=8 on the forest canopy image produced 1.9% match success and 20.69 dB PSNR — 0.01 dB better than step=16.

- **Python/CuPy prototype.** Encode times are for an interpreted implementation. An optimised CUDA C++ system would be substantially faster.

- **Rate-distortion vs transform codecs.** PIFS does not compete with JPEG, WebP, or HEIC on rate-distortion for photographic content. The system's advantage is resolution independence, not compression efficiency at 1×.

- **Decode time grows with scale.** Decode is sublinear but not constant: 4× output scale requires approximately 2.9× more decode time than 1× (measured across four images). 10× outputs take 10–20 seconds on T4.

---

## Citing this work

```bibtex
@misc{ndegwa2026fractal,
  title   = {GPU-Accelerated Fractal Image Compression with Adaptive Encoding,
             Compact Colour Coding, and Resolution-Independent Upscaling},
  author  = {Ndegwa, Lewis Kagiri and Tole},
  year    = {2026},
  month   = {March},
  url     = {https://github.com/10kwise/GPU-fractal-compression},
  note    = {Technical University of Mombasa, GPL-3.0}
}
```

---

## References

1. Barnsley, M. F., & Sloan, A. D. (1988). A better way to compress images. *Byte Magazine*, 13(1), 215–223.
2. Jacquin, A. E. (1992). Image coding based on a fractal theory of iterated contractive image transformations. *IEEE Transactions on Image Processing*, 1(1), 18–30.
3. Fisher, Y. (Ed.). (1995). *Fractal Image Compression: Theory and Application*. Springer-Verlag.
4. Haque et al. (2014). GPU accelerated fractal image compression for medical imaging. arXiv:1404.0774.
5. Al Sideiri et al. (2020). CUDA implementation of fractal image compression. *Journal of Real-Time Image Processing*, 17(5), 1375–1387.
6. Hernandez-Lopez & Muñiz-Pérez (2022). Parallel fractal image compression using quadtree partition. *Journal of Real-Time Image Processing*, 19, 117–130.
7. Wohlberg, B., & De Jager, G. (1999). A review of the fractal image coding literature. *IEEE Transactions on Image Processing*, 8(12), 1716–1729.
8. Bhattacharya, N., & Bhattacharya, M. (2004). Fractal image compression: A randomized approach. *Pattern Recognition Letters*, 25(10), 1167–1180.
9. Li, M., & U, K. T. (2025). An enhanced fractal image compression algorithm based on adaptive non-uniform rectangular partition. *Electronics*, 14(13), 2550.
10. Gilli, G., & Saupe, D. (2000). Adaptive post-processing for fractal image compression. *Proceedings IEEE ICIP*.
11. Kodak Lossless True Color Image Suite. Available: http://r0k.us/graphics/kodak/
12. Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: From error visibility to structural similarity. *IEEE Transactions on Image Processing*, 13(4), 600–612.
