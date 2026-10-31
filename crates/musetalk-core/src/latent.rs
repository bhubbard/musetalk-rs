use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

pub const LATENT_CHANNELS: usize = 4;
pub const DOWNSCALE_FACTOR: usize = 8;

/// VAE Latent Tensor (Channels x Height x Width)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentTensor {
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub data: Vec<f32>,
}

impl LatentTensor {
    pub fn new(channels: usize, height: usize, width: usize) -> Self {
        Self {
            channels,
            height,
            width,
            data: vec![0.0f32; channels * height * width],
        }
    }

    /// Zero out lower face area in latent space (inpainting mask) using contiguous slice fills
    pub fn mask_lower_half(&mut self) {
        let mid_h = self.height / 2;
        let plane_size = self.height * self.width;
        let mask_offset = mid_h * self.width;
        for c in 0..self.channels {
            let start = c * plane_size + mask_offset;
            let end = ((c + 1) * plane_size).min(self.data.len());
            if start < end {
                self.data[start..end].fill(0.0);
            }
        }
    }

    /// Blend an inpainted latent tensor into the original upper-face latent tensor using a spatial mask
    pub fn blend_inpainted(&mut self, inpainted: &LatentTensor, spatial_mask: &[f32]) {
        let plane_size = self.height * self.width;
        if inpainted.data.len() != self.data.len() || spatial_mask.len() != plane_size {
            return;
        }
        for c in 0..self.channels {
            let offset = c * plane_size;
            for (idx, &m) in spatial_mask.iter().enumerate() {
                let weight = m.clamp(0.0, 1.0);
                let orig = self.data[offset + idx];
                let gen = inpainted.data[offset + idx];
                self.data[offset + idx] = orig * (1.0 - weight) + gen * weight;
            }
        }
    }
}

pub struct LatentMaskGenerator;

impl LatentMaskGenerator {
    /// Create 1-channel binary mask for VAE latent space (32x32 for 256x256 image)
    pub fn create_latent_mask(height: usize, width: usize) -> Vec<f32> {
        let mut mask = vec![0.0f32; height * width];
        let mid_h = height / 2;
        let start = mid_h * width;
        if start < mask.len() {
            mask[start..].fill(1.0);
        }
        mask
    }

    /// Create 1-channel cosine-feathered soft mask for seamless VAE latent space inpainting
    pub fn create_feathered_latent_mask(
        height: usize,
        width: usize,
        feather_rows: usize,
    ) -> Vec<f32> {
        let mut mask = vec![0.0f32; height * width];
        let mid_h = height / 2;
        let half_feather = feather_rows / 2;
        let ramp_start = mid_h.saturating_sub(half_feather);
        let ramp_end = (mid_h + half_feather.max(1)).min(height);
        let ramp_span = (ramp_end.saturating_sub(ramp_start)).max(1) as f32;

        for h in 0..height {
            let val = if h < ramp_start {
                0.0
            } else if h >= ramp_end {
                1.0
            } else {
                let progress = (h - ramp_start) as f32 / ramp_span;
                0.5 * (1.0 - (PI * progress).cos())
            };
            let row_start = h * width;
            mask[row_start..row_start + width].fill(val);
        }

        mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latent_masking() {
        let mut latent = LatentTensor::new(4, 32, 32);
        latent.data.fill(1.0);
        latent.mask_lower_half();

        // Upper half should remain 1.0
        assert_eq!(latent.data[0], 1.0);
        // Lower half should be 0.0
        let lower_idx = 20 * 32 + 16;
        assert_eq!(latent.data[lower_idx], 0.0);
    }

    #[test]
    fn test_feathered_latent_blending() {
        let mut orig = LatentTensor::new(4, 32, 32);
        orig.data.fill(1.0);
        let mut gen = LatentTensor::new(4, 32, 32);
        gen.data.fill(3.0);

        let mask = LatentMaskGenerator::create_feathered_latent_mask(32, 32, 4);
        orig.blend_inpainted(&gen, &mask);

        assert!((orig.data[0] - 1.0).abs() < 1e-5);
        assert!((orig.data[31 * 32 + 16] - 3.0).abs() < 1e-5);
    }
}
