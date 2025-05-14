use crate::{
    error::AppError,
    model::{ErrorResponse as ModelErrorResponse, LcaRequest, ProgressUpdate, ProgressUpdateType},
    sse::{send_sse_status, SseTracingLayer},
};


use axum_streams::StreamBodyAs;
// Removed: use futures::stream::Stream; 
use lca_core::GpuDevice;
use lca_rs::model::LcaModel;
use serde::Serialize;
// Removed std::convert::Infallible
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream; // Added for MPSC to Stream conversion
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use validator::Validate;


use axum::{
  response::{sse::{Event, Sse}, IntoResponse}, routing::get, Json, Router
};
use axum_extra::headers;
use axum_extra::TypedHeader;
use futures::stream::{self, Stream};
use std::{convert::Infallible, path::PathBuf, time::Duration};
use tokio_stream::StreamExt as _;


// Shared state for GpuDevice, wrapped in Arc for thread safety if needed.
// For simplicity, we might initialize it per request or have a global static/lazy_static.
// Let's assume we initialize it per request for now to avoid complexities with global state
// unless performance dictates otherwise. If lca-core's GpuDevice::new() is cheap, this is fine.
// If it's expensive, a shared instance (e.g. via Axum State extractor) would be better.
// For now, no shared GpuDevice state. It will be created in the spawned task.

#[utoipa::path(
    post,
    path = "/calculate-lca",
    request_body = LcaRequest,
    responses(
        (status = 200, description = "LCA calculation started, streaming progress via SSE.", body = String, content_type = "text/event-stream"),
        (status = 400, description = "Invalid request payload or model compilation error.", body = ModelErrorResponse), // Use aliased ErrorResponse
        (status = 500, description = "Internal server error during calculation.", body = ModelErrorResponse)  // Use aliased ErrorResponse
    ),
    tag = "LCA Webservice"
)]
pub async fn calculate_lca_handler(
    Json(payload): Json<LcaRequest>,
) -> Sse<impl Stream<Item = Result<Event, axum::Error>>> {
    tracing::info!(target: "lca_webservice::handler", "Received LCA calculation request: {:?}", payload.case_name);

    // 1. Validate the incoming request payload using `validator`
    payload
        .validate()
        .map_err(AppError::Validation).unwrap();
    tracing::debug!(target: "lca_webservice::handler", "Request payload validated successfully.");

    // 2. Convert LcaRequest (web model) to LcaModel (lca-rs model)
    let lca_model_rs: LcaModel = payload.into();
    tracing::debug!(target: "lca_webservice::handler", "LcaRequest converted to lca_rs::LcaModel.");

    // 3. Create an MPSC channel for SSE events
    let (sse_tx, sse_rx) = mpsc::channel::<Result<Event, axum::Error>>(100); // Buffer size 100

    // Clone the sender for the spawned task
    let task_sse_tx = sse_tx.clone();

    // Send initial "Processing started" message
    send_sse_status(
        &task_sse_tx,
        ProgressUpdateType::Status,
        "Request received, starting processing.".to_string(),
    )
    .await;

    // 4. Spawn a Tokio task to perform the compilation and calculation
    tokio::spawn(async move {
        // This task will own its LcaModel and perform the work.
        // The SseTracingLayer will be active only within this task's tracing context.
        let sse_layer = SseTracingLayer::new(task_sse_tx.clone());

        // Construct a temporary registry for this task with the SSE layer
        let registry = tracing_subscriber::registry().with(sse_layer);

        // Execute the LCA logic within the context of this specialized subscriber
        let _guard = registry.set_default(); // Apply this subscriber as default for current scope/task

        tracing::info!(target: "lca_webservice::handler", "Worker task started. Compiling LCA model...");
        send_sse_status(
            &task_sse_tx,
            ProgressUpdateType::Compilation,
            "Compiling LCA model...".to_string(),
        )
        .await;

        // 5. Compile the LCA model
        let lca_system = match lca_model_rs.compile() {
            Ok(system) => {
                tracing::info!(target: "lca_webservice::handler", "LCA model compiled successfully.");
                send_sse_status(
                    &task_sse_tx,
                    ProgressUpdateType::Compilation,
                    "LCA model compiled successfully.".to_string(),
                )
                .await;
                system
            }
            Err(e) => {
                tracing::error!(target: "lca_webservice::handler", "LCA model compilation failed: {:?}", e);
                let err_payload = ProgressUpdate {
                    event_type: ProgressUpdateType::Error,
                    message: format!("LCA model compilation failed: {}", e),
                    details: Some(serde_json::json!({"compilation_error": e.to_string()})),
                    result_data: None,
                };
                if let Ok(json_str) = serde_json::to_string(&err_payload) {
                    let _ = task_sse_tx.send(Ok(Event::default().data(json_str))).await;
                }
                // Close the channel by dropping the sender or sending a specific "end" event if desired
                return; // Exit the task
            }
        };

        // 6. Initialize GPU Device (can be slow, do it after compilation)
        tracing::info!(target: "lca_webservice::handler", "Initializing GPU device...");
        send_sse_status(
            &task_sse_tx,
            ProgressUpdateType::Status,
            "Initializing GPU device...".to_string(),
        )
        .await;
        let device: Arc<GpuDevice> = match GpuDevice::new().await { // Added type annotation
            Ok(d) => {
                tracing::info!(target: "lca_webservice::handler", "GPU device initialized successfully.");
                send_sse_status(
                    &task_sse_tx,
                    ProgressUpdateType::Status,
                    "GPU device initialized successfully.".to_string(),
                )
                .await;
                Arc::new(d) // Wrap in Arc if it needs to be shared or is not Clone
            }
            Err(e) => {
                tracing::error!(target: "lca_webservice::handler", "Failed to initialize GPU device: {:?}", e);
                 let err_payload = ProgressUpdate {
                    event_type: ProgressUpdateType::Error,
                    message: "Failed to initialize GPU device.".to_string(),
                    details: Some(serde_json::json!({"gpu_error": e.to_string()})),
                    result_data: None,
                };
                if let Ok(json_str) = serde_json::to_string(&err_payload) {
                    let _ = task_sse_tx.send(Ok(Event::default().data(json_str))).await;
                }
                return;
            }
        };
        
        // 7. Evaluate the LCA system
        // The SseTracingLayer will automatically pick up log::info/debug calls from lca_rs::evaluate
        tracing::info!(target: "lca_webservice::handler", "Starting LCA evaluation...");
        // Note: lca_system.evaluate takes demand and methods as Option<Vec<...>>
        // For now, we'll pass None, meaning it uses default demands from the model if any.
        // This could be parameterized in LcaRequest later.
        match lca_system.evaluate(device.as_ref(), None, None).await {
            Ok(results) => {
                tracing::info!(target: "lca_webservice::handler", "LCA evaluation completed successfully. Results: {:?}", results.len());
                let success_payload = ProgressUpdate {
                    event_type: ProgressUpdateType::Result,
                    message: "LCA evaluation completed successfully.".to_string(),
                    details: None,
                    result_data: Some(results),
                };
                if let Ok(json_str) = serde_json::to_string(&success_payload) {
                    let _ = task_sse_tx.send(Ok(Event::default().data(json_str))).await;
                }
            }
            Err(e) => {
                tracing::error!(target: "lca_webservice::handler", "LCA evaluation failed: {:?}", e);
                let err_payload = ProgressUpdate {
                    event_type: ProgressUpdateType::Error,
                    message: format!("LCA evaluation failed: {}", e),
                    details: Some(serde_json::json!({"evaluation_error": e.to_string()})),
                    result_data: None,
                };
                if let Ok(json_str) = serde_json::to_string(&err_payload) {
                    let _ = task_sse_tx.send(Ok(Event::default().data(json_str))).await;
                }
            }
        }
        tracing::info!(target: "lca_webservice::handler", "Worker task finished.");
        // The mpsc channel will be closed when task_sse_tx is dropped, signaling the end of the stream.
    });

    // 5. Return the SSE stream
    let receiver_stream = ReceiverStream::new(sse_rx);

    Sse::new(receiver_stream).keep_alive(
      axum::response::sse::KeepAlive::new()
          .interval(Duration::from_secs(1))
          .text("keep-alive-text"),
    )
}


pub async fn sse_handler(
  TypedHeader(user_agent): TypedHeader<headers::UserAgent>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
  println!("`{}` connected", user_agent.as_str());

  // A `Stream` that repeats an event every second
  //
  // You can also create streams from tokio channels using the wrappers in
  // https://docs.rs/tokio-stream
  let stream = stream::repeat_with(|| Event::default().data("hi!"))
      .map(Ok)
      .throttle(Duration::from_secs(1))
      .take(3);

  Sse::new(stream).keep_alive(
      axum::response::sse::KeepAlive::new()
          .interval(Duration::from_secs(1))
          .text("keep-alive-text"),
  )
}

#[derive(Debug, Clone, Serialize)]
struct MyTestStructure {
    some_test_field: String
}

// Your possibly stream of objects
fn my_source_stream() -> impl Stream<Item=MyTestStructure> {
    // Simulating a stream with a plain vector and throttling to show how it works
    use tokio_stream::StreamExt;
    futures::stream::iter(vec![
        MyTestStructure {
            some_test_field: "test1".to_string()
        }; 1000
    ])
    .throttle(std::time::Duration::from_millis(1500))
    .take(3)
}

pub async fn test_json_array_stream() -> impl IntoResponse {
  StreamBodyAs::json_nl(my_source_stream())
}
