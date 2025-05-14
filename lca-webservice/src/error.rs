use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use lca_rs::error::{LcaError as LcaRsError, LcaModelCompilationError};
use serde::Serialize;
use thiserror::Error;
use validator::ValidationErrors;

use crate::model::ErrorResponse; // Using the ErrorResponse model we defined

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Input validation failed")]
    Validation(#[from] ValidationErrors),

    #[error("LCA Model compilation failed: {0}")]
    Compilation(#[from] LcaModelCompilationError),

    #[error("LCA calculation error: {0}")]
    LcaCalculation(#[from] LcaRsError),

    #[error("Failed to initialize GPU device: {0}")]
    GpuInit(String),

    #[error("Internal server error: {0}")]
    Internal(String),
    // Add other specific error types as needed
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_type, error_message, details) = match self {
            AppError::Validation(err) => {
                let messages: Vec<String> = err
                    .field_errors()
                    .into_iter()
                    .flat_map(|(_, errors)| errors.iter().map(|e| e.to_string()))
                    .collect();
                (
                    StatusCode::BAD_REQUEST,
                    "Validation Error".to_string(),
                    "One or more input fields failed validation.".to_string(),
                    Some(messages),
                )
            }
            AppError::Compilation(err) => (
                StatusCode::BAD_REQUEST,
                "LCA Model Compilation Error".to_string(),
                err.to_string(),
                None,
            ),
            AppError::LcaCalculation(err) => (
                StatusCode::INTERNAL_SERVER_ERROR, // Or BAD_REQUEST if it's due to bad input not caught by compile
                "LCA Calculation Error".to_string(),
                err.to_string(),
                None,
            ),
            AppError::GpuInit(err_msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "GPU Initialization Error".to_string(),
                err_msg,
                None,
            ),
            AppError::Internal(err_msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Internal Server Error".to_string(),
                err_msg,
                None,
            ),
        };

        let body = Json(ErrorResponse {
            status_code: status.as_u16(),
            error: error_type,
            message: error_message,
            details,
        });

        (status, body).into_response()
    }
}

// Helper for SSE to serialize errors if needed, or use AppError directly
#[derive(Serialize, Debug)]
pub struct SseErrorPayload {
    pub error_type: String,
    pub message: String,
    pub details: Option<Vec<String>>,
}

impl From<&AppError> for SseErrorPayload {
    fn from(err: &AppError) -> Self {
        match err {
            AppError::Validation(validation_err) => {
                let messages: Vec<String> = validation_err
                    .field_errors()
                    .into_iter()
                    .flat_map(|(_, errors)| errors.iter().map(|e| e.to_string()))
                    .collect();
                SseErrorPayload {
                    error_type: "ValidationError".to_string(),
                    message: "Input validation failed.".to_string(),
                    details: Some(messages),
                }
            }
            AppError::Compilation(compilation_err) => SseErrorPayload {
                error_type: "CompilationError".to_string(),
                message: compilation_err.to_string(),
                details: None,
            },
            AppError::LcaCalculation(lca_err) => SseErrorPayload {
                error_type: "LcaCalculationError".to_string(),
                message: lca_err.to_string(),
                details: None,
            },
            AppError::GpuInit(msg) => SseErrorPayload {
                error_type: "GpuInitError".to_string(),
                message: msg.clone(),
                details: None,
            },
            AppError::Internal(msg) => SseErrorPayload {
                error_type: "InternalError".to_string(),
                message: msg.clone(),
                details: None,
            },
        }
    }
}
