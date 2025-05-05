use crate::buffer::Buffer;
use crate::context::GpuContext;
use crate::error::LsolverError;
use std::sync::Arc;

/// Represents a dense matrix stored in row-major order on the GPU.
#[derive(Debug)]
pub struct DenseMatrixf64 {
    context: Arc<GpuContext>,
    buffer: Buffer<f64>,
    rows: usize,
    cols: usize,
}

impl DenseMatrixf64 {
    /// Creates a new DenseMatrixf64 on the GPU from CPU data.
    pub fn from_data(
        context: Arc<GpuContext>,
        data: &[f64],
        rows: usize,
        cols: usize,
    ) -> Result<Self, LsolverError> {
        if data.len() != rows * cols {
            return Err(LsolverError::InvalidDimensions(format!(
                "Data length ({}) does not match dimensions ({}x{})",
                data.len(),
                rows,
                cols
            )));
        }

        let buffer = Buffer::new_init(
            &context,
            Some("dense_matrix_f64 buffer"),
            data,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST, // Add usages as needed
        )?;

        Ok(Self {
            context,
            buffer,
            rows,
            cols,
        })
    }

    /// Reads the matrix data back from the GPU to the CPU.
    /// Note: This is typically slow and should be avoided in performance-critical paths.
    pub async fn read_back(&self) -> Result<Vec<f64>, LsolverError> {
        // Use context's read_buffer_to_cpu instead of removed Buffer::read_contents
        let content = self.context
            .read_buffer_to_cpu(self.buffer.inner(), self.buffer.size())
            .await?;
        Ok(content)
    }

    // --- Getters ---
    pub fn buffer(&self) -> &Buffer<f64> {
        &self.buffer
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn context(&self) -> &Arc<GpuContext> {
        &self.context
    }

    pub fn size_bytes(&self) -> wgpu::BufferAddress {
        (self.rows * self.cols * std::mem::size_of::<f64>()) as wgpu::BufferAddress
    }
}

// TODO: Add Vector types (potentially reusing DenseMatrixf64 with cols=1 or a dedicated type)
// TODO: Add traits for different matrix properties (Symmetric, PositiveDefinite, etc.) later.
