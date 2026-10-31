use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FaceBox {
    pub xmin: u32,
    pub ymin: u32,
    pub xmax: u32,
    pub ymax: u32,
}

impl FaceBox {
    pub fn width(&self) -> u32 {
        self.xmax.saturating_sub(self.xmin)
    }

    pub fn height(&self) -> u32 {
        self.ymax.saturating_sub(self.ymin)
    }

    pub fn area(&self) -> u32 {
        self.width() * self.height()
    }
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

    /// Compute Intersection-over-Union (IoU) between two face bounding boxes
    pub fn compute_iou(a: &FaceBox, b: &FaceBox) -> f32 {
        let inter_xmin = a.xmin.max(b.xmin);
        let inter_ymin = a.ymin.max(b.ymin);
        let inter_xmax = a.xmax.min(b.xmax);
        let inter_ymax = a.ymax.min(b.ymax);

        let inter_w = inter_xmax.saturating_sub(inter_xmin);
        let inter_h = inter_ymax.saturating_sub(inter_ymin);
        let inter_area = inter_w * inter_h;

        let union_area = a.area() + b.area() - inter_area;
        if union_area == 0 {
            0.0
        } else {
            inter_area as f32 / union_area as f32
        }
    }

    /// Smooth a sequence of frame-by-frame face bounding boxes using an exponential moving average
    pub fn smooth_trajectory(boxes: &[FaceBox], alpha: f32) -> Vec<FaceBox> {
        if boxes.is_empty() {
            return Vec::new();
        }
        let weight = alpha.clamp(0.0, 1.0);
        let mut smoothed = Vec::with_capacity(boxes.len());
        let mut cur_xmin = boxes[0].xmin as f32;
        let mut cur_ymin = boxes[0].ymin as f32;
        let mut cur_xmax = boxes[0].xmax as f32;
        let mut cur_ymax = boxes[0].ymax as f32;

        for b in boxes {
            cur_xmin = weight * (b.xmin as f32) + (1.0 - weight) * cur_xmin;
            cur_ymin = weight * (b.ymin as f32) + (1.0 - weight) * cur_ymin;
            cur_xmax = weight * (b.xmax as f32) + (1.0 - weight) * cur_xmax;
            cur_ymax = weight * (b.ymax as f32) + (1.0 - weight) * cur_ymax;

            smoothed.push(FaceBox {
                xmin: cur_xmin.round() as u32,
                ymin: cur_ymin.round() as u32,
                xmax: cur_xmax.round() as u32,
                ymax: cur_ymax.round() as u32,
            });
        }

        smoothed
    }
}
