use std::net::SocketAddr;
use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

// Module declarations for our application structure
mod app;
mod error;
mod handler;
mod model;
mod openapi;
mod sse;

// Re-export for convenience if needed elsewhere, or keep private

mod health;
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

    // Build our application router
    let app = app::build_router();

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
