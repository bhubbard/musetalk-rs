use serde::{Deserialize, Serialize};

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

    /// Zero out lower face area in latent space (inpainting mask)
    pub fn mask_lower_half(&mut self) {
        let mid_h = self.height / 2;
        for c in 0..self.channels {
            for h in mid_h..self.height {
                for w in 0..self.width {
                    let idx = c * (self.height * self.width) + (h * self.width + w);
                    if idx < self.data.len() {
                        self.data[idx] = 0.0;
                    }
                }
            }
        }
    }
}

pub struct LatentMaskGenerator;

impl LatentMaskGenerator {
    /// Create 1-channel binary/soft mask for VAE latent space (32x32 for 256x256 image)
    pub fn create_latent_mask(height: usize, width: usize) -> Vec<f32> {
        let mut mask = vec![0.0f32; height * width];
        let mid_h = height / 2;

        for h in mid_h..height {
            for w in 0..width {
                mask[h * width + w] = 1.0;
            }
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
        let lower_idx = 0 * (32 * 32) + (20 * 32 + 16);
        assert_eq!(latent.data[lower_idx], 0.0);
    }
}
