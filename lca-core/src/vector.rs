use crate::context::GpuContext;
use crate::error::LcaCoreError;
use crate::traits::Vector;
use std::fmt::Debug;
use std::{mem, sync::Arc};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;
use wgpu::PollType;

/// A generic wrapper around a `wgpu::Buffer` to manage typed vector data on the GPU.
#[cfg_attr(feature = "wasm", wasm_bindgen)]
#[derive(Debug)]
pub struct GpuVector {
    // Renamed from Buffer, kept Zeroable bound
    buffer: wgpu::Buffer,
    size: usize, // Number of elements of type f64
    size_bytes: u64,
    usage: wgpu::BufferUsages,
    label: String,
    // Add internal context reference
    pub(crate) context: Arc<GpuContext>,
}

// Added Zeroable bound here as well to match struct definition

impl GpuVector {
    /// Internal constructor used by GpuDevice.
    pub(crate) fn new_internal(
        buffer: wgpu::Buffer,
        size: usize,
        usage: wgpu::BufferUsages,
        label: String,
        context: Arc<GpuContext>,
    ) -> Self {
        let size_bytes = (size * mem::size_of::<f64>()) as u64;
        Self {
            buffer,
            size,
            size_bytes,
            usage,
            label,
            context,
        }
    }

    /// Returns the underlying `wgpu::Buffer`.
    /// Use with caution, prefer higher-level methods on GpuVector or GpuDevice.
    pub(crate) fn inner(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Returns the number of elements the vector can hold.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns the size of the vector's buffer in bytes.
    pub fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    /// Returns the buffer's usage flags.
    pub fn usage(&self) -> wgpu::BufferUsages {
        self.usage
    }

    /// Returns the buffer's label.
    pub fn label(&self) -> &str {
        self.label.as_str()
    }

    /// Returns a `BindingResource` for the entire buffer.
    pub fn as_entire_binding(&self) -> wgpu::BindingResource {
        self.buffer.as_entire_binding()
    }

    /// Reads the vector's contents back to the CPU using the internal context.
    /// Note: This is an async operation and involves GPU-CPU synchronization.
    pub async fn read_contents(&self) -> Result<Vec<f64>, LcaCoreError> {
        // Use the internal context
        self.context
            .read_buffer_to_cpu(self.inner(), self.size())
            .await
    }

    /// Writes data from a CPU slice into this GPU vector using the internal context.
    /// Note: This is an async operation.
    pub async fn write_contents(&self, data: &[f64]) -> Result<(), LcaCoreError> {
        if data.len() != self.size {
            return Err(LcaCoreError::InvalidDimensions(format!(
                "Data length ({}) does not match GpuVector size ({})",
                data.len(),
                self.size
            )));
        }
        // Use the internal context
        self.context.write_buffer(self.inner(), data).await
    }

    /// Clones the content from another GpuVector into this one using the internal context.
    /// Both vectors must have the same size and belong to the same logical device (context).
    /// Handles necessary synchronization.
    pub fn clone_from(&mut self, source: &GpuVector) -> Result<(), LcaCoreError> {
        // Optional: Add check if contexts are the same if that's a requirement
        // if !Arc::ptr_eq(&self.context, &source.context) {
        //     return Err(LsolverError::Internal("Cannot clone_from vectors with different contexts".to_string()));
        // }
        if self.size != source.size {
            return Err(LcaCoreError::InvalidDimensions(format!(
                "Vector sizes for clone_from mismatch: {} != {}",
                self.size, source.size
            )));
        }
        if self.size_bytes == 0 {
            return Ok(()); // Nothing to copy
        }

        // Ensure destination buffer has COPY_DST usage
        if !self.usage.contains(wgpu::BufferUsages::COPY_DST) {
            return Err(LcaCoreError::UnsupportedOperation(
                "Destination vector buffer requires COPY_DST usage for clone_from".to_string(),
            ));
        }
        // Ensure source buffer has COPY_SRC usage
        if !source.usage.contains(wgpu::BufferUsages::COPY_SRC) {
            return Err(LcaCoreError::UnsupportedOperation(
                "Source vector buffer requires COPY_SRC usage for clone_from".to_string(),
            ));
        }

        // Use the internal context
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("GpuVector Clone From Encoder"),
                });
        encoder.copy_buffer_to_buffer(source.inner(), 0, self.inner(), 0, self.size_bytes);
        self.context.queue.submit(Some(encoder.finish()));

        // We need to wait for the copy to complete before potentially using the destination buffer
        // in subsequent operations within the same async block or function call sequence.
        // Polling is a simple way to ensure this on native. WASM relies on implicit synchronization.
        // Use the internal context for polling
        cfg_if::cfg_if! {
            if #[cfg(not(target_arch = "wasm32"))] {
                // Consider making polling optional or configurable if performance is critical
                let _ = self.context.device.poll(PollType::Wait);
            }
            // On WASM, the browser's event loop and WebGPU implementation handle synchronization implicitly
            // between queue submissions. Awaiting subsequent operations that depend on this buffer
            // should be sufficient.
        }
        Ok(())
    }
}

// Implement the Vector trait for GpuVector
impl Vector for GpuVector {
    type Value = f64;

    fn len(&self) -> usize {
        self.size()
    }
}
