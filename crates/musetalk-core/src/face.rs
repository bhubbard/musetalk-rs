use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceBox {
    pub xmin: u32,
    pub ymin: u32,
    pub xmax: u32,
    pub ymax: u32,
}

pub struct FaceBounding;

impl FaceBounding {
    /// Apply bbox_shift offset to adjust vertical framing of mouth
    pub fn apply_shift(box_in: &FaceBox, shift_pixels: i32, img_height: u32) -> FaceBox {
        let shifted_ymin = (box_in.ymin as i32 + shift_pixels).clamp(0, img_height as i32) as u32;
        let shifted_ymax = (box_in.ymax as i32 + shift_pixels).clamp(0, img_height as i32) as u32;

        FaceBox {
            xmin: box_in.xmin,
            ymin: shifted_ymin,
            xmax: box_in.xmax,
            ymax: shifted_ymax,
        }
    }
}
