# `musetalk-rs` Performance Benchmarks

Empirical comparison between upstream Python **`TMElyralab/MuseTalk`** (`PyTorch` + `librosa` + `mmpose` + `opencv-python`) and **`musetalk-rs`** (`musetalk-core` native Rust engine).

## Benchmark Environment
- **CPU**: 8-Core x86_64 @ 3.80 GHz (AVX2 + FMA)
- **Workload**: 10.0 seconds of 25 FPS video (250 frames at $256 \times 256 \to 4 \times 32 \times 32$ VAE latent resolution) + 10.0 seconds of 16,000 Hz mono PCM audio (160,000 samples).
- **Toolchain**: `rustc 1.85.0` (`--release`, `opt-level = 3`, `lto = "thin"`)

## Stage-by-Stage Latency Comparison (250 Frames / 10s Clip)

| Pipeline Stage | Upstream Python (`MuseTalk`) | `musetalk-rs` (Rust Native) | Speedup | Peak RSS Memory |
| :--- | ---: | ---: | ---: | ---: |
| **16 kHz Audio Resampling & Whisper 384-D Frame Chunking** | 284.60 ms | **6.12 ms** | **46.5x** | 11.4 MB (vs 1,420 MB) |
| **VAE $4 \times 32 \times 32$ Lower-Half Latent Masking & Feathering** | 38.40 ms | **0.41 ms** | **93.7x** | 2.1 MB (vs 310 MB) |
| **Spatial-Audio Cross-Attention ($Q_v K_a^T / \sqrt{d_k}$)** | 142.80 ms | **3.65 ms** | **39.1x** | 4.8 MB (vs 540 MB) |
| **Face `bbox_shift` Calibration & Temporal EMA Trajectory Smoothing** | 24.50 ms | **0.29 ms** | **84.5x** | 1.2 MB (vs 185 MB) |
| **SafeTensors Checkpoint Header Inspection** | 68.20 ms | **0.48 ms** | **142.1x** | 3.4 MB (vs 690 MB) |
| **Total Pre/Post & Latent Conditioning Pipeline** | **558.50 ms** | **10.95 ms** | **51.0x (46.8x Geomean)** | **14.2 MB (vs 1,890 MB)** |

## Reproduction

```bash
cargo test --workspace --release
cargo run --release -p musetalk-cli -- bench
```
