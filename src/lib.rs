use std::ffi::{c_char, c_void, CStr};
use opencv::core::Mat;
use opencv::mod_prelude::Boxed;
use crate::runner::Runner;

mod runner;
mod processing;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum CreateRunnerError {
    None = 0,
    PathMissingOrInvalid = 1,
    ExecutionProviderRegistrationError = 2,
    ModelLoadingError = 3,
    UnknownError = 4,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum SegmentImageError {
    None = 0,
    NoRunner = 1,
    InvalidInput = 2,
    PreProcessingError = 3,
    ModelExecutionError = 4,
    PostProcessingError = 5,
    UnknownError = 6
}


#[repr(C)]
pub struct RunnerOpaque(Box<Runner>);

#[unsafe(no_mangle)]
pub extern "C" fn create_runner(
    path: *const c_char,
    use_gpu: bool,
    out_runner: *mut *mut RunnerOpaque,
) -> CreateRunnerError {
    if path.is_null() {
        return CreateRunnerError::PathMissingOrInvalid;
    }

    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => {
            return CreateRunnerError::PathMissingOrInvalid;
        }
    };

    match Runner::new(path_str, use_gpu) {
        Ok(runner) => {
            let runner_opaque = RunnerOpaque(Box::new(runner));
            unsafe {
                *out_runner = Box::into_raw(Box::new(runner_opaque));
            }
            CreateRunnerError::None
        }
        Err(err) => err
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn destroy_runner(runner: *mut RunnerOpaque) {
    if !runner.is_null() {
        unsafe {
            drop(Box::from_raw(runner));
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn segment_image(
    runner: *mut RunnerOpaque,
    mat_ptr: *mut c_void,
    out_mask_mat_ptr: *mut *mut c_void,
) -> SegmentImageError {
    if runner.is_null() {
        return SegmentImageError::NoRunner;
    }

    if mat_ptr.is_null() {
        return SegmentImageError::InvalidInput;
    }

    let mat = unsafe {
        Mat::from_raw(mat_ptr)
    };

    let runner = unsafe { &*runner };

    let result = match processing::segment_image(&runner.0, &mat) {
        Ok(mask) => {
            let mask_mat_ptr = mask.into_raw();
            unsafe {
                *out_mask_mat_ptr = mask_mat_ptr;
            }
            SegmentImageError::None
        }
        Err(err) => err,
    };

    // We do not own the pointer, so we're forgetting it to prevent rust from dropping it
    std::mem::forget(mat);

    result
}
