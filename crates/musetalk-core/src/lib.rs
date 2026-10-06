pub mod attention;
pub mod audio;
pub mod config;
pub mod face;
pub mod latent;
pub mod pipeline;
pub mod weights;

pub use attention::{CrossAttentionSummary, SpatialAudioAttention};
pub use audio::{WhisperAudioProcessor, WhisperFrameFeature, WHISPER_EMBED_DIM, WHISPER_SAMPLE_RATE};
pub use config::MuseTalkConfig;
pub use face::{FaceBounding, FaceBox};
pub use latent::{LatentMaskGenerator, LatentTensor, DOWNSCALE_FACTOR, LATENT_CHANNELS};
pub use pipeline::MuseTalkPipeline;
pub use weights::{ModelWeightInfo, WeightsManager};
