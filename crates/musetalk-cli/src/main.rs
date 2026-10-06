use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::*;
use musetalk_core::{
    LatentTensor, MuseTalkConfig, MuseTalkPipeline, WeightsManager, WhisperAudioProcessor,
};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "musetalk",
    author = "MuseTalk Contributors",
    version = "0.1.0",
    about = "High-performance Rust CLI for MuseTalk: Real-Time High Quality Lip Synchronization in Latent Space"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Synchronize mouth movements in latent space to match driving audio in real-time
    Sync {
        /// Source face video
        #[arg(short = 'v', long = "video", required = true)]
        video: PathBuf,

        /// Driving audio WAV file
        #[arg(short = 'a', long = "audio", required = true)]
        audio: PathBuf,

        /// Destination output MP4 file
        #[arg(short = 'o', long = "output", default_value = "results/musetalk_out.mp4")]
        output: PathBuf,

        /// Vertical bounding box shift offset (adjusts mouth crop center)
        #[arg(long = "bbox-shift", default_value_t = 0)]
        bbox_shift: i32,

        /// Batch size for real-time inference
        #[arg(long = "batch-size", default_value_t = 8)]
        batch_size: usize,
    },

    /// Inspect audio tracks or SafeTensors checkpoints
    Inspect {
        /// Target file
        #[arg(required = true)]
        file: PathBuf,
    },

    /// Benchmark Whisper audio feature processing and VAE latent masking
    Bench {
        /// Optional audio WAV file
        #[arg(short = 'a', long = "audio")]
        audio: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Sync {
            video,
            audio,
            output,
            bbox_shift,
            batch_size,
        } => {
            println!("{}", "=== MuseTalk Latent Lip Synchronization Pipeline ===".bold().cyan());
            println!("  Input Video:  {}", video.display().to_string().yellow());
            println!("  Input Audio:  {}", audio.display().to_string().yellow());
            println!("  BBox Shift:   {}", bbox_shift);
            println!("  Batch Size:   {}", batch_size);

            let config = MuseTalkConfig {
                bbox_shift,
                fps: 25.0,
                batch_size,
                fp16: true,
                whisper_model: "whisper-tiny".to_string(),
                checkpoint_dir: None,
            };

            let start = Instant::now();
            let pipeline = MuseTalkPipeline::new(config);
            let out_file = pipeline.run(&video, &audio, &output)?;

            println!(
                "{} Latent lip synchronization complete in {:.2?} -> {}",
                "✓".green(),
                start.elapsed(),
                out_file.display().to_string().bold()
            );
        }

        Commands::Inspect { file } => {
            println!("{} {}", "Inspecting:".bold().cyan(), file.display());
            let ext = file.extension().and_then(|s| s.to_str()).unwrap_or("");

            match ext.to_lowercase().as_str() {
                "wav" => {
                    let samples = WhisperAudioProcessor::load_wav(&file)?;
                    let feats = WhisperAudioProcessor::extract_whisper_features(&samples, 25.0);
                    println!("  Type:             WAV Audio Track");
                    println!("  Sample Rate:      16000 Hz");
                    println!("  Duration:         {:.2} seconds", samples.len() as f32 / 16000.0);
                    println!("  Whisper Frames:   {} (384 embedding dims)", feats.len());
                }
                "safetensors" => {
                    let tensors = WeightsManager::inspect_checkpoint(&file)?;
                    println!("  Type:             SafeTensors Checkpoint");
                    println!("  Tensors Count:    {}", tensors.len());
                    for t in tensors.iter().take(10) {
                        println!("    • {:<35} {:?} ({})", t.name, t.shape, t.dtype);
                    }
                    if tensors.len() > 10 {
                        println!("    ... and {} more tensors", tensors.len() - 10);
                    }
                }
                _ => {
                    println!("File format .{} not supported for detailed inspection", ext);
                }
            }
        }

        Commands::Bench { audio } => {
            println!("{}", "=== Benchmarking MuseTalk Latent Engine ===".bold().cyan());
            let samples = if let Some(p) = audio {
                WhisperAudioProcessor::load_wav(&p)?
            } else {
                (0..160_000).map(|i| (i as f32 * 0.05).sin() * 0.35).collect()
            };

            let duration = samples.len() as f32 / 16000.0;
            println!("Testing on {:.1} seconds of 16kHz audio...", duration);

            let t0 = Instant::now();
            let feats = WhisperAudioProcessor::extract_whisper_features(&samples, 25.0);
            let audio_dur = t0.elapsed();

            let t1 = Instant::now();
            let iterations = 10_000;
            let mut latent = LatentTensor::new(4, 32, 32);
            for _ in 0..iterations {
                latent.mask_lower_half();
            }
            let latent_dur = t1.elapsed() / iterations;

            println!("\nResults:");
            println!("  Whisper Features (384-dim): {:.2?} ({} frames, {:.1} fps throughput)",
                audio_dur, feats.len(), feats.len() as f64 / audio_dur.as_secs_f64());
            println!("  Latent Face Masking (4x32x32): {:.2?} ({:.1} latents/sec)",
                latent_dur, 1.0 / latent_dur.as_secs_f64());
        }
    }

    Ok(())
}
