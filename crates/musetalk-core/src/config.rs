use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MuseTalkConfig {
    /// Vertical shift offset for face bounding box (adjusts mouth center)
    pub bbox_shift: i32,
    /// Video processing frame rate (default 25.0)
    pub fps: f32,
    /// Batch size for real-time inference (default 8)
    pub batch_size: usize,
    /// Use FP16 half precision
    pub fp16: bool,
    /// Whisper model type (tiny, base)
    pub whisper_model: String,
    /// Checkpoint directory override
    pub checkpoint_dir: Option<PathBuf>,
}

impl Default for MuseTalkConfig {
    fn default() -> Self {
        Self {
            bbox_shift: 0,
            fps: 25.0,
            batch_size: 8,
            fp16: true,
            whisper_model: "whisper-tiny".to_string(),
            checkpoint_dir: None,
        }
    }
}
