use crate::{handler, model}; // Import necessary modules
use utoipa::OpenApi;

#[derive(OpenApi)]
#[openapi(
    paths(
        crate::health_check, // Changed from crate::main::health_check
        handler::calculate_lca_handler,
    ),
    components(
        schemas(
            model::LcaRequest,
            model::ErrorResponse,
            model::ProgressUpdate,
            // model::CalculationResponse, // This might be part of SSE stream, not a direct response schema for the POST
            
            // Schemas from lca_rs::model that are part of LcaRequest
            // These are already included by LcaRequest schema generation if they are public
            // and LcaRequest uses them directly. If LcaRequest has its own mirrored structs (like LcaRequest itself),
            // then those are the ones to list. The current model.rs defines mirrored structs.
            model::DbImportRequest,
            model::ProcessRequest,
            model::OutputItemRequest,
            model::InputItemRequest,
            model::BiosphereItemRequest,
            model::ProductRefRequest,
            model::SubstanceRefRequest,
            model::ExternalRefRequest,
            model::AmountRequest,
            model::ProductRequest,
            model::SubstanceRequest,
            model::ImpactRequest,
            model::SubstanceTypeRequest,
            model::DemandItemRequest
        )
    ),
    tags(
        (name = "LCA Webservice", description = "Endpoints for Life Cycle Assessment calculations and management")
    ),
    info(
        title = "LCA Webservice API",
        version = "0.1.0",
        description = "An API for performing Life Cycle Assessments (LCA) and streaming progress updates.",
        contact(
            name = "zojeda ",
            url = "http://example.com",
            email = "zojeda@example.com"
        ),
        license(
            name = "MIT/Apache-2.0",
            url = "https://opensource.org/licenses/MIT"
        )
    )
)]
pub struct ApiDoc;
