use opencv::core::{Mat, MatTraitConst, MatTraitConstManual, MatTraitManual, Size, CV_32FC3, CV_8UC1};
use opencv::imgproc::{cvt_color, resize, COLOR_BGR2RGB, INTER_LINEAR};
use crate::runner::Runner;
use crate::SegmentImageError;

const MODEL_WIDTH: i32 = 256;
const MODEL_HEIGHT: i32 = 256;

pub fn segment_image(
    runner: &Runner,
    image: &Mat
) -> Result<Mat, SegmentImageError> {
    let mut image_rgb = Mat::default();
    cvt_color(&image, &mut image_rgb, COLOR_BGR2RGB, 0)
        .map_err(|_| SegmentImageError::PreProcessingError)?;

    let mut resized = Mat::default();
    resize(&image_rgb, &mut resized, Size::new(MODEL_WIDTH, MODEL_HEIGHT), 0.0, 0.0, INTER_LINEAR)
        .map_err(|_| SegmentImageError::PreProcessingError)?;

    let mut image_float = Mat::default();
    resized.convert_to(&mut image_float, CV_32FC3, 1.0 / 255.0, 0.0)
        .map_err(|_| SegmentImageError::PreProcessingError)?;
    
    let float_slice: &[f32] = bytemuck::try_cast_slice(image_float.data_bytes().map_err(|_| SegmentImageError::PreProcessingError)?)
        .map_err(|_| SegmentImageError::PreProcessingError)?;
    
    let output_data = runner.run(
        vec![1, MODEL_HEIGHT as usize, MODEL_WIDTH as usize, 3],
        float_slice
    ).map_err(|_| SegmentImageError::ModelExecutionError)?;

    let mut mask = Mat::new_rows_cols_with_default(256, 256, CV_8UC1, 0.into()).unwrap();

    let mask_data = mask.data_bytes_mut().unwrap();
    for i in 0..(MODEL_WIDTH * MODEL_HEIGHT) as usize {
        mask_data[i] = if output_data[i] > 0.5 { 255 } else { 0 };
    }

    Ok(mask)
}