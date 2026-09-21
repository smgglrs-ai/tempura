use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::Response;
use axum::Json;

const OPEN_PATHS: &[&str] = &["/health", "/health/inference", "/ready", "/metrics"];

pub async fn auth_middleware(
    State(state): State<crate::AppState>,
    req: Request,
    next: Next,
) -> Result<Response, (StatusCode, Json<serde_json::Value>)> {
    let Some(ref expected_key) = state.api_key else {
        return Ok(next.run(req).await);
    };

    if OPEN_PATHS.contains(&req.uri().path()) {
        return Ok(next.run(req).await);
    }

    let provided = req
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .map(str::trim)
        .or_else(|| req.headers().get("x-api-key").and_then(|v| v.to_str().ok()));

    if provided == Some(expected_key.as_str()) {
        Ok(next.run(req).await)
    } else {
        Err((
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({
                "error": {
                    "message": "Invalid API key",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key"
                }
            })),
        ))
    }
}
