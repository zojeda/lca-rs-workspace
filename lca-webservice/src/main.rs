use axum::{routing::get, Router};
use handler::{sse_handler, test_json_array_stream};
use std::net::SocketAddr;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, fmt};
use utoipa::OpenApi; // Re-added: Needed for ApiDoc::openapi() trait method
use utoipa_swagger_ui::SwaggerUi;

// Module declarations for our application structure
mod error;
mod handler;
mod model;
mod openapi;
mod sse;

// Re-export for convenience if needed elsewhere, or keep private
use crate::openapi::ApiDoc;

#[tokio::main]
async fn main() {
    // Initialize tracing to capture logs from `log` crate and `tracing` calls
    // Fallback to "info" level if RUST_LOG is not set.
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,lca_webservice=debug,lca_rs=debug"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt::layer()) // Standard formatting layer
        .init();

    // Initialize tracing_log to bridge log crate events to tracing
    if let Err(e) = tracing_log::LogTracer::init() {
        eprintln!("Failed to set logger: {}", e); // Use eprintln for early errors
    }

    tracing::info!("Tracing initialized. Starting LCA webservice...");

    // Define CORS layer
    let cors = CorsLayer::new()
        .allow_origin(Any) // Allow any origin
        .allow_methods(Any) // Allow all methods
        .allow_headers(Any); // Allow all headers

    // Build our application router
    let app = Router::new()
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi())) // Restored .url()
        .route("/", get(health_check))
        .route("/calculate-lca", axum::routing::post(handler::calculate_lca_handler))
        .route("/sse", get(sse_handler))
        .route("/other", get(test_json_array_stream))
        // Add more routes here as needed
        .layer(TraceLayer::new_for_http()) // Layer for HTTP tracing
        .layer(cors); // Apply CORS middleware

    // Run the server
    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    tracing::info!("Listening on {}", addr);

    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(listener) => listener,
        Err(e) => {
            tracing::error!("Failed to bind to address {}: {}", addr, e);
            return;
        }
    };

    if let Err(e) = axum::serve(listener, app).await {
        tracing::error!("Server error: {}", e);
    }
}

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

// The main ApiDoc is defined in openapi.rs and merged into the router.
// No need for a duplicate or placeholder ApiDoc struct here.
