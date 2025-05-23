use std::ffi::{c_char, c_int, c_void, CStr};
use opencv::core::Mat;
use opencv::mod_prelude::Boxed;
use tracing::subscriber::set_global_default;
use crate::logger::{FfiSubscriber, LoggerCallback, LOGGER_CALLBACK};
use crate::runner::Runner;

mod runner;
mod processing;
mod logger;

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
pub struct RunnerOpaque(*mut Runner);

#[unsafe(no_mangle)]
pub extern "C" fn create_runner(
    path: *const c_char,
    use_gpu: bool,
    device_id: i32,
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

    let device_id = if device_id < 0 {
        None
    } else {
        Some(device_id)
    };
    match Runner::new(path_str, use_gpu, device_id) {
        Ok(runner) => {
            tracing::info!("Model loaded successfully from {}", path_str);
            let runner_ptr = Box::into_raw(Box::new(runner));
            tracing::info!("Runner pointer: {:?}", runner_ptr);
            let runner_opaque = RunnerOpaque(runner_ptr);
            tracing::info!("Runner opaque created");
            unsafe {
                *out_runner = Box::into_raw(Box::new(runner_opaque));
                tracing::info!("Runner opaque pointer set: {:?}", *out_runner);
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
            let runner_opaque = Box::from_raw(runner);
            let inner_runner_ptr = runner_opaque.0;
            if !inner_runner_ptr.is_null() {
                drop(Box::from_raw(inner_runner_ptr));
            }
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

    let runner = unsafe {
        let runner_opaque = &*runner;
        let inner_runner_ptr = runner_opaque.0;
        if inner_runner_ptr.is_null() {
            return SegmentImageError::NoRunner;
        }
        &*inner_runner_ptr
    };

    let result = match processing::segment_image(runner, &mat) {
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

#[unsafe(no_mangle)]
pub extern "C" fn init_logger(callback: LoggerCallback) -> c_int {
    // Store the callback
    {
        let mut cb_guard = match LOGGER_CALLBACK.lock() {
            Ok(guard) => guard,
            Err(_) => {
                eprintln!("Failed to lock callback mutex");
                return -1;
            }
        };
        *cb_guard = Some(callback);
    }

    // Set the global subscriber
    match set_global_default(FfiSubscriber) {
        Ok(_) => {
            tracing::info!("Tracing initialized successfully");
            0 // Success
        }
        Err(e) => {
            eprintln!("Failed to set tracing subscriber: {}", e);
            -1 // Error
        }
    }
}