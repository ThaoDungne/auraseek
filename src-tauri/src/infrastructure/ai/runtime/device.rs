use crate::core::config::{AppConfig, DevicePreference};

pub fn selected_device() -> &'static str {
    match AppConfig::global().device {
        DevicePreference::Cpu => "cpu",
        DevicePreference::Cuda => "cuda",
        DevicePreference::Auto => detect_best_device(),
    }
}

fn detect_best_device() -> &'static str {
    #[cfg(feature = "cuda")]
    {
        if ort::execution_providers::CUDAExecutionProvider::default()
            .build()
            .is_available()
        {
            return "cuda";
        }
    }
    "cpu"
}
