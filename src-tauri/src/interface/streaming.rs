use axum::{extract::{Query, Request}, response::{Response, IntoResponse}, routing::get, Router};

#[derive(serde::Deserialize)]
struct VideoQuery {
    path: String,
}

async fn stream_video(Query(q): Query<VideoQuery>, req: Request) -> Response {
    use tower::ServiceExt;
    let service = tower_http::services::ServeFile::new(&q.path);
    match service.oneshot(req).await {
        Ok(res) => res.into_response(),
        Err(_) => axum::http::StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

pub fn spawn_stream_server() -> u16 {
    let cors = tower_http::cors::CorsLayer::new()
        .allow_origin(tower_http::cors::Any)
        .allow_methods(tower_http::cors::Any)
        .allow_headers(tower_http::cors::Any);

    let app = Router::new()
        .route("/stream", get(stream_video))
        .layer(cors);

    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    listener.set_nonblocking(true).unwrap();

    tauri::async_runtime::spawn(async move {
        let tokio_listener = tokio::net::TcpListener::from_std(listener).unwrap();
        axum::serve(tokio_listener, app).await.unwrap();
    });

    port
}

#[tauri::command]
pub fn cmd_get_stream_port(state: tauri::State<'_, crate::app::state::AppState>) -> u16 {
    state.stream_port.load(std::sync::atomic::Ordering::Relaxed)
}
