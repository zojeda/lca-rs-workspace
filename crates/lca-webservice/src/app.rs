use axum::{Router, routing::get};
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use crate::{handler, health::health_check, openapi::ApiDoc};

/// Build the Axum Router for the webservice.
pub fn build_router() -> Router {
    // Define CORS layer
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .route("/", get(health_check))
        .route(
            "/calculate-lca",
            axum::routing::post(handler::calculate_lca_handler),
        )
        .route("/sse", get(handler::sse_handler))
        .route("/other", get(handler::test_json_array_stream))
        .layer(TraceLayer::new_for_http())
        .layer(cors)
}
