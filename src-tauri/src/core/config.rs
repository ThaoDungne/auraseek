use std::path::PathBuf;
use std::sync::OnceLock;

static GLOBAL_CONFIG: OnceLock<AppConfig> = OnceLock::new();

#[derive(Debug, Clone)]
pub enum DevicePreference {
    Cpu,
    Cuda,
    Auto,
}

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub data_dir: PathBuf,
    pub log_path: PathBuf,
    pub model_dir: PathBuf,
    pub surreal_bin: Option<PathBuf>,

    pub face_threshold: f32,
    pub yolo_confidence: f32,
    pub yolo_iou: f32,
    pub max_batch_size: usize,

    pub device: DevicePreference,
    pub num_threads: usize,

    pub debug: bool,
}

impl Default for AppConfig {
    fn default() -> Self {
        let data_dir = crate::platform::paths::fallback_data_dir();
        let log_path = PathBuf::from(crate::platform::paths::default_log_path());

        Self {
            model_dir: data_dir.clone(),
            data_dir,
            log_path,
            surreal_bin: None,

            face_threshold: 0.33,
            yolo_confidence: 0.25,
            yolo_iou: 0.45,
            max_batch_size: 32,

            device: DevicePreference::Auto,
            num_threads: num_cpus::get().max(1),

            debug: false,
        }
    }
}

fn env_or<T: std::str::FromStr>(key: &str, fallback: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(fallback)
}

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var(key).ok().map(PathBuf::from)
}

fn clamp_f32(val: f32, min: f32, max: f32, name: &str) -> f32 {
    if val < min || val > max {
        let clamped = val.clamp(min, max);
        eprintln!(
            "[config] {} value {:.4} out of range [{:.2}, {:.2}], clamped to {:.4}",
            name, val, min, max, clamped
        );
        clamped
    } else {
        val
    }
}

impl AppConfig {
    pub fn from_env() -> Self {
        let defaults = Self::default();

        let data_dir = env_path("AURASEEK_DATA_DIR")
            .unwrap_or(defaults.data_dir);
        let log_path = env_path("AURASEEK_LOG_PATH")
            .unwrap_or(defaults.log_path);
        let model_dir = env_path("AURASEEK_MODEL_DIR")
            .unwrap_or_else(|| data_dir.clone());
        let surreal_bin = env_path("AURASEEK_SURREAL_BIN")
            .or(defaults.surreal_bin);

        let face_threshold = clamp_f32(
            env_or("AURASEEK_FACE_THRESHOLD", defaults.face_threshold),
            0.0, 1.0, "AURASEEK_FACE_THRESHOLD",
        );
        let yolo_confidence = clamp_f32(
            env_or("AURASEEK_YOLO_CONFIDENCE", defaults.yolo_confidence),
            0.0, 1.0, "AURASEEK_YOLO_CONFIDENCE",
        );
        let yolo_iou = clamp_f32(
            env_or("AURASEEK_YOLO_IOU", defaults.yolo_iou),
            0.0, 1.0, "AURASEEK_YOLO_IOU",
        );
        let max_batch_size = env_or("AURASEEK_MAX_BATCH_SIZE", defaults.max_batch_size)
            .max(1);

        let device = match std::env::var("AURASEEK_DEVICE")
            .unwrap_or_default()
            .to_lowercase()
            .as_str()
        {
            "cpu" => DevicePreference::Cpu,
            "cuda" => DevicePreference::Cuda,
            _ => DevicePreference::Auto,
        };

        let num_threads = env_or("AURASEEK_NUM_THREADS", defaults.num_threads)
            .max(1);

        let debug = env_or("AURASEEK_DEBUG", defaults.debug);

        Self {
            data_dir,
            log_path,
            model_dir,
            surreal_bin,
            face_threshold,
            yolo_confidence,
            yolo_iou,
            max_batch_size,
            device,
            num_threads,
            debug,
        }
    }

    pub fn global() -> &'static AppConfig {
        GLOBAL_CONFIG.get_or_init(|| Self::from_env())
    }

    pub fn init(config: AppConfig) {
        let _ = GLOBAL_CONFIG.set(config);
    }
}
