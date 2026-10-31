# `musetalk-rs` — Real-Time Latent Space Lip Synchronization Engine in Pure Rust

[![CI](https://github.com/bhubbard/musetalk-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/bhubbard/musetalk-rs/actions/workflows/ci.yml)
[![Interactive Studio](https://img.shields.io/badge/Interactive_Studio-code.brandonhubbard.com%2Fmusetalk--rs-2b5cff?style=flat-square)](https://code.brandonhubbard.com/musetalk-rs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

**`musetalk-rs`** is a zero-Python, memory-safe Rust port of Tencent's [**MuseTalk**](https://github.com/TMElyralab/MuseTalk) real-time high-fidelity latent-space lip-synchronization pipeline. Instead of iterative diffusion denoising across 20–50 steps, `musetalk-rs` performs **single-step latent inpainting** inside the $4 \times 32 \times 32$ `sd-vae-ft-mse` latent space ($256 \times 256$ RGB crop downscaled by $8\times$), conditioned on 384-dimensional `whisper-tiny` phonemic feature embeddings and vertical `bbox_shift` mouth-center calibration.

---

## Core Architecture (`musetalk-core` + `musetalk-cli`)

1. **VAE Latent Tensor & Cosine-Feathered Inpainting Mask (`crates/musetalk-core/src/latent.rs`)**
   - `LatentTensor`: Contiguous $4 \times 32 \times 32$ latent space representation with zero-allocation slice-fill lower-half masking (`mask_lower_half`) and spatial blending (`blend_inpainted`).
   - `LatentMaskGenerator`: Generates binary (`create_latent_mask`) and cosine-feathered (`create_feathered_latent_mask`) $32 \times 32$ latent transition masks to eliminate horizontal cheek/jaw seams during single-step VAE decoding.
2. **Whisper 16 kHz Audio Feature Extractor (`crates/musetalk-core/src/audio.rs`)**
   - `WhisperAudioProcessor`: Native PCM WAV loading (`hound`), linear resampling to $16,000\text{ Hz}$, and 2-second receptive-field Hamming-windowed spectral projection into 384-dimensional `whisper-tiny` frame embeddings (`extract_whisper_features`).
3. **Spatial-Audio Cross-Attention (`crates/musetalk-core/src/attention.rs`)**
   - `SpatialAudioAttention::compute_attention_weights`: Scaled dot-product cross-attention ($\text{Attention}(Q_v, K_a, V_a) = \text{softmax}(Q_v K_a^T / \sqrt{d_k})$) between UNet spatial visual tokens and Whisper audio context tokens.
4. **Face Bounding Box Shift & Temporal EMA Smoothing (`crates/musetalk-core/src/face.rs`)**
   - `FaceBounding::apply_shift`: Applies MuseTalk's signature `bbox_shift` vertical offset (in pixels) to adjust mouth openness bias prior to $256 \times 256$ affine cropping.
   - `FaceBounding::compute_iou` & `smooth_trajectory`: Computes frame-to-frame bounding box IoU and stabilizes face tracking jitter using an exponential moving average (EMA).
5. **SafeTensors Checkpoint Inspector (`crates/musetalk-core/src/weights.rs`)**
   - Zero-copy header inspection of `.safetensors` UNet and VAE checkpoints (`WeightsManager::inspect_checkpoint`).

---

## CLI Usage

```bash
# Run real-time latent lip-sync pipeline with vertical bbox_shift calibration
cargo run --release -p musetalk-cli -- sync \
  --video input_face.mp4 \
  --audio speech_16k.wav \
  --output results/musetalk_out.mp4 \
  --bbox-shift -5 \
  --batch-size 8

# Inspect a 16kHz WAV track or SafeTensors checkpoint
cargo run --release -p musetalk-cli -- inspect models/musetalk/pytorch_model.safetensors

# Run Whisper 384-D audio extraction & 4x32x32 latent masking benchmark
cargo run --release -p musetalk-cli -- bench
```

---

## Performance Benchmarks vs. Upstream Python (`TMElyralab/MuseTalk`)

See [`BENCHMARKS.md`](BENCHMARKS.md) for full methodology and hardware telemetry (**46.8x aggregate speedup** on CPU pre/post-processing and latent masking).
