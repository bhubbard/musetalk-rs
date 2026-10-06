use serde::{Deserialize, Serialize};

/// Attention weights summary between visual feature tokens and Whisper audio tokens
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossAttentionSummary {
    pub num_visual_tokens: usize,
    pub num_audio_tokens: usize,
    pub max_attention_score: f32,
    pub mean_attention_score: f32,
}

pub struct SpatialAudioAttention;

impl SpatialAudioAttention {
    /// Compute cross-attention energy between spatial visual tokens and audio feature tokens
    pub fn compute_attention_weights(
        visual_tokens: &[f32],
        token_dim: usize,
        audio_tokens: &[f32],
        audio_dim: usize,
    ) -> CrossAttentionSummary {
        let num_v = if token_dim > 0 { visual_tokens.len() / token_dim } else { 0 };
        let num_a = if audio_dim > 0 { audio_tokens.len() / audio_dim } else { 0 };

        if num_v == 0 || num_a == 0 {
            return CrossAttentionSummary {
                num_visual_tokens: 0,
                num_audio_tokens: 0,
                max_attention_score: 0.0,
                mean_attention_score: 0.0,
            };
        }

        let scale = 1.0 / (token_dim.min(audio_dim) as f32).sqrt();
        let mut max_score = f32::MIN;
        let mut sum_score = 0.0f32;
        let count = num_v * num_a;

        for v in 0..num_v {
            let v_slice = &visual_tokens[v * token_dim..(v + 1) * token_dim];
            for a in 0..num_a {
                let a_slice = &audio_tokens[a * audio_dim..(a + 1) * audio_dim];
                let dot: f32 = v_slice.iter().zip(a_slice.iter()).map(|(&x, &y)| x * y).sum();
                let score = dot * scale;
                max_score = max_score.max(score);
                sum_score += score;
            }
        }

        CrossAttentionSummary {
            num_visual_tokens: num_v,
            num_audio_tokens: num_a,
            max_attention_score: max_score,
            mean_attention_score: sum_score / count as f32,
        }
    }
}
