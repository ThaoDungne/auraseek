use crate::log_info;
use crate::log_warn;

pub fn build_session(model_path: &str) -> anyhow::Result<ort::session::Session> {
    use ort::session::Session;
    use ort::execution_providers::ExecutionProviderDispatch;

    let ep_builders: Vec<(&str, Box<dyn Fn() -> ExecutionProviderDispatch>)> = vec![
        ("TensorRT", Box::new(|| ort::execution_providers::TensorRTExecutionProvider::default().build())),
        ("CUDA",     Box::new(|| ort::execution_providers::CUDAExecutionProvider::default().build())),
        ("CoreML",   Box::new(|| ort::execution_providers::CoreMLExecutionProvider::default().build())),
        ("DirectML", Box::new(|| ort::execution_providers::DirectMLExecutionProvider::default().build())),
        ("OpenVINO", Box::new(|| ort::execution_providers::OpenVINOExecutionProvider::default().build())),
    ];

    for (name, builder_fn) in ep_builders {
        let ep = builder_fn();
        match Session::builder()?.with_execution_providers([ep]) {
            Ok(mut builder) => {
                match builder.commit_from_file(model_path) {
                    Ok(session) => {
                        log_info!("model: {:<40} | provider: {}", model_path, name);
                        return Ok(session);
                    }
                    Err(e) => {
                        log_warn!("failed to commit {} with {}: {}", model_path, name, e);
                    }
                }
            }
            Err(e) => {
                log_warn!("provider {} not available for {}: {}", name, model_path, e);
            }
        }
    }

    // fallback to cpu
    let cpu_ep = ort::execution_providers::CPUExecutionProvider::default().build();

    // Tránh dùng `?` trên lỗi SessionBuilder (không From vào anyhow do ràng buộc Send/Sync),
    // thay vào đó map_err thủ công sang anyhow::Error.
    let builder = Session::builder().map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let mut builder = builder
        .with_execution_providers([cpu_ep])
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let session = builder
        .commit_from_file(model_path)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    log_info!("model: {:<40} | provider: cpu", model_path);
    Ok(session)
}
