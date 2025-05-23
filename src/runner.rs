use ort::inputs;
use ort::session::Session;
use ort::value::Tensor;
use crate::CreateRunnerError;

pub(crate) struct Runner {
    session: Session,
}

impl Runner {
    pub fn new(model_path: &str, use_gpu: bool) -> Result<Self, CreateRunnerError> {
        let providers = if use_gpu {
            vec![
                #[cfg(feature = "cuda")]
                ort::execution_providers::CUDAExecutionProvider::default().build(),
                #[cfg(feature = "directml")]
                ort::execution_providers::DirectMLExecutionProvider::default().build(),
                #[cfg(target_os = "macos")]
                ort::execution_providers::CoreMLExecutionProvider::default().build(),
            ]
        } else {
            vec![ort::execution_providers::CPUExecutionProvider::default().build()]
        };
        
        let session = Session::builder().map_err(|_| CreateRunnerError::UnknownError)?
            .with_execution_providers(providers).map_err(|_| CreateRunnerError::ExecutionProviderRegistrationError)?
            .commit_from_file(model_path).map_err(|_| CreateRunnerError::ModelLoadingError)?;
        Ok(Runner { session })
    }

    pub fn run(&self, dimensions: Vec<usize>, input: &[f32]) -> ort::Result<Vec<f32>> {
        let tensor = Tensor::from_array((dimensions, input))?;
        let outputs = self.session.run(inputs![tensor]?)?;
        let output = outputs[0].try_extract_tensor::<f32>()?;
        let output_vec = output.as_slice()
            .ok_or_else(|| ort::Error::new("Failed to extract output slice"))?
            .to_vec();
        Ok(output_vec)
    }
}