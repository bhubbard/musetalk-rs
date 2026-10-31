use musetalk_core::{
    FaceBounding, FaceBox, LatentMaskGenerator, LatentTensor, MuseTalkConfig,
    SpatialAudioAttention, WeightsManager, WhisperAudioProcessor, DOWNSCALE_FACTOR,
    LATENT_CHANNELS, WHISPER_EMBED_DIM, WHISPER_SAMPLE_RATE,
};

#[test]
fn test_audio_constants_and_whisper_features() {
    assert_eq!(WHISPER_SAMPLE_RATE, 16000);
    assert_eq!(WHISPER_EMBED_DIM, 384);

    let dummy_audio = vec![0.05f32; 16000 * 2]; // 2 seconds
    let features = WhisperAudioProcessor::extract_whisper_features(&dummy_audio, 25.0);
    assert_eq!(features.len(), 50); // 25 fps * 2s
    assert_eq!(features[0].features.len(), WHISPER_EMBED_DIM);
}

#[test]
fn test_latent_tensor_geometry() {
    assert_eq!(LATENT_CHANNELS, 4);
    assert_eq!(DOWNSCALE_FACTOR, 8);

    let mut latent = LatentTensor::new(4, 32, 32);
    assert_eq!(latent.channels, 4);
    assert_eq!(latent.height, 32);
    assert_eq!(latent.width, 32);
    assert_eq!(latent.data.len(), 4 * 32 * 32);

    latent.data.fill(1.0);
    latent.mask_lower_half();

    // Upper half should remain 1.0
    assert_eq!(latent.data[0], 1.0);

    // Lower half should be masked (0.0)
    let lower_idx = 24 * 32 + 16;
    assert_eq!(latent.data[lower_idx], 0.0);

    let mask = LatentMaskGenerator::create_latent_mask(32, 32);
    assert_eq!(mask.len(), 32 * 32);
    assert_eq!(mask[5 * 32 + 16], 0.0);
    assert_eq!(mask[20 * 32 + 16], 1.0);

    let feathered = LatentMaskGenerator::create_feathered_latent_mask(32, 32, 4);
    assert_eq!(feathered.len(), 32 * 32);
    assert_eq!(feathered[4 * 32 + 16], 0.0);
    assert_eq!(feathered[28 * 32 + 16], 1.0);
    assert!(feathered[15 * 32 + 16] > 0.0 && feathered[15 * 32 + 16] < 1.0);
}

#[test]
fn test_face_bbox_shift_and_smoothing() {
    let bbox = FaceBox {
        xmin: 50,
        ymin: 60,
        xmax: 150,
        ymax: 180,
    };

    let shifted_down = FaceBounding::apply_shift(&bbox, 15, 256);
    assert_eq!(shifted_down.ymin, 75);
    assert_eq!(shifted_down.ymax, 195);
    assert_eq!(shifted_down.xmin, 50);

    let shifted_up = FaceBounding::apply_shift(&bbox, -20, 256);
    assert_eq!(shifted_up.ymin, 40);
    assert_eq!(shifted_up.ymax, 160);

    let iou = FaceBounding::compute_iou(&bbox, &shifted_down);
    assert!(iou > 0.7 && iou < 1.0);

    let jittery = vec![bbox.clone(), shifted_down, bbox];
    let smoothed = FaceBounding::smooth_trajectory(&jittery, 0.35);
    assert_eq!(smoothed.len(), 3);
    assert!(smoothed[1].ymin > 60 && smoothed[1].ymin < 75);
}

#[test]
fn test_spatial_audio_attention() {
    let visual = vec![1.0f32; 16 * 64]; // 16 spatial tokens of dim 64
    let audio = vec![1.0f32; 4 * 64]; // 4 audio tokens of dim 64

    let summary = SpatialAudioAttention::compute_attention_weights(&visual, 64, &audio, 64);
    assert_eq!(summary.num_visual_tokens, 16);
    assert_eq!(summary.num_audio_tokens, 4);
    assert!(summary.max_attention_score > 0.0);
    assert!(summary.mean_attention_score > 0.0);
}

#[test]
fn test_config_defaults() {
    let config = MuseTalkConfig::default();
    assert_eq!(config.batch_size, 8);
    assert_eq!(config.fps, 25.0);
    assert_eq!(config.bbox_shift, 0);
    assert_eq!(config.whisper_model, "whisper-tiny");
}

#[test]
fn test_weights_manager_invalid_path() {
    let res = WeightsManager::inspect_checkpoint("/invalid/path/musetalk.safetensors");
    assert!(res.is_err());
}
