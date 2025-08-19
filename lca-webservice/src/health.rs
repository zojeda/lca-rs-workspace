#[utoipa::path(
    get,
    path = "/",
    responses(
        (status = 200, description = "Service is healthy", body = String)
    )
)]
pub async fn health_check() -> &'static str {
    tracing::info!("Health check endpoint hit");
    "LCA Webservice is running!"
}
