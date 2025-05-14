use crate::model::{ProgressUpdate, ProgressUpdateType};
use axum::response::sse::Event; // Changed from axum_streams::Event
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;
use tokio::sync::mpsc;
use tracing::{field::Visit, Level, Subscriber, Event as TracingEvent};
use tracing_subscriber::{layer::Context, Layer}; // Removed FilterExt

// SseTracingLayer will hold the sender part of an MPSC channel
// to send formatted log messages to the SSE stream.
#[derive(Clone)]
pub struct SseTracingLayer {
    sse_tx: mpsc::Sender<Result<Event, axum::Error>>,
    // We only care about events from specific targets, e.g., "lca_rs" or "lca_webservice::handler"
    // And specific levels, e.g., INFO, WARN, ERROR
    // This could be enhanced with more specific filtering if needed.
}

impl SseTracingLayer {
    pub fn new(sse_tx: mpsc::Sender<Result<Event, axum::Error>>) -> Self {
        Self { sse_tx }
    }
}

// Visitor to extract fields from a tracing event
struct SseTracingVisitor<'a> {
    fields: BTreeMap<String, Value>,
    message: Option<String>,
    sse_tx: &'a mpsc::Sender<Result<Event, axum::Error>>,
    level: &'a Level,
    target: &'a str,
}

impl<'a> Visit for SseTracingVisitor<'a> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn fmt::Debug) {
        let field_name = field.name();
        if field_name == "message" { // Standard field for log messages via log::* macros
            self.message = Some(format!("{:?}", value));
        } else {
            self.fields
                .insert(field_name.to_string(), Value::String(format!("{:?}", value)));
        }
    }

    // Handle other types if necessary (e.g., record_str, record_i64)
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        let field_name = field.name();
        if field_name == "message" {
            self.message = Some(value.to_string());
        } else {
            self.fields.insert(field_name.to_string(), Value::String(value.to_string()));
        }
    }
}

impl<'a> SseTracingVisitor<'a> {
    fn new(
        sse_tx: &'a mpsc::Sender<Result<Event, axum::Error>>,
        level: &'a Level,
        target: &'a str,
    ) -> Self {
        SseTracingVisitor {
            fields: BTreeMap::new(),
            message: None,
            sse_tx,
            level,
            target,
        }
    }

    // Consumes the visitor and sends the event
    fn finish_event(self) {
        let sse_message_str = self.message.unwrap_or_else(|| {
            // If no explicit message, try to construct one from fields or use a default
            if !self.fields.is_empty() {
                format!("Fields: {:?}", self.fields)
            } else {
                "Log event".to_string()
            }
        });

        let event_type = match *self.level {
            Level::ERROR => ProgressUpdateType::Error,
            Level::WARN => ProgressUpdateType::Info, // Or a specific Warn type
            Level::INFO => ProgressUpdateType::Info,
            Level::DEBUG => ProgressUpdateType::Info, // Or a specific Debug type, often too verbose for SSE
            Level::TRACE => return, // Usually too verbose for SSE
        };
        
        // Only send if target is relevant, e.g. lca_rs or our own handler's specific logs
        // This basic filter can be enhanced.
        if !(self.target.starts_with("lca_rs") || 
             self.target.starts_with("lca_core") || 
             self.target.starts_with("lca_webservice::handler") ||
             self.target.starts_with("lca_webservice::sse")) { // Allow logs from sse module itself for status
            return;
        }


        let progress_update = ProgressUpdate {
            event_type,
            message: sse_message_str,
            details: if self.fields.is_empty() { None } else { Some(Value::Object(self.fields.into_iter().collect())) },
            result_data: None,
        };

        match serde_json::to_string(&progress_update) {
            Ok(json_string) => {
                let event = Event::default().data(json_string);
                // Try to send, but don't block/panic if the receiver is dropped (e.g., client disconnected)
                if self.sse_tx.try_send(Ok(event)).is_err() {
                    // Optional: log that the SSE channel was closed/full
                    // tracing::warn!(target: "lca_webservice::sse", "SSE channel closed or full, could not send log event.");
                }
            }
            Err(e) => {
                tracing::error!(target: "lca_webservice::sse", "Failed to serialize ProgressUpdate for SSE: {}", e);
            }
        }
    }
}

impl<S: Subscriber + Send + Sync + for<'lookup> tracing_subscriber::registry::LookupSpan<'lookup>> Layer<S>
    for SseTracingLayer
{
    fn on_event(&self, event: &TracingEvent<'_>, _ctx: Context<'_, S>) {
        // Filter for events from "lca_rs", "lca_core", or specific parts of "lca_webservice"
        let metadata = event.metadata();
        let target = metadata.target();
        let level = metadata.level();

        // More sophisticated filtering can be added here based on target, level, or fields
        if *level <= Level::INFO || target.starts_with("lca_rs") || target.starts_with("lca_core") || target.starts_with("lca_webservice::handler") {
            let mut visitor = SseTracingVisitor::new(&self.sse_tx, level, target);
            event.record(&mut visitor);
            visitor.finish_event();
        }
    }
    
    // Optionally, implement other Layer methods like on_enter, on_exit if needed for spans
}

// This function will be called by the handler to set up the SSE stream
// and the tracing layer for the duration of that request.
// It's a simplified concept; actual integration in the handler will be more involved,
// likely involving spawning a task that has this layer active in its tracing context.
pub async fn stream_lca_progress(
    // This function would conceptually take the lca_model and return a stream
    // For now, it's a placeholder for where the SseTracingLayer would be used.
    // The actual setup will be in handler.rs
) -> impl axum::response::IntoResponse {
    // This is just a conceptual placeholder.
    // The real implementation will be in the handler,
    // creating a channel, spawning the LCA work with the SseTracingLayer active for that task's context.
    "SSE streaming setup placeholder"
}

// Helper to send a simple status message over SSE
pub async fn send_sse_status(
    tx: &mpsc::Sender<Result<Event, axum::Error>>,
    event_type: ProgressUpdateType,
    message: String,
) {
    let update = ProgressUpdate {
        event_type,
        message,
        details: None,
        result_data: None,
    };
    if let Ok(json_string) = serde_json::to_string(&update) {
        let event = Event::default().data(json_string);
        let _ = tx.try_send(Ok(event)); // Ignore error if client disconnected
    }
}
