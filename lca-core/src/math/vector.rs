use crate::context::GpuContext;
use crate::error::LcaCoreError;
use crate::math::traits::Vector;
use std::{mem, sync::Arc};
#[cfg(feature = "wasm-bindings")]
use wasm_bindgen::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use wgpu::PollType;

/// A generic wrapper around a `wgpu::Buffer` to manage typed vector data on the GPU.
#[cfg_attr(feature = "wasm-bindings", wasm_bindgen)]
#[derive(Debug)]
pub struct GpuVector {
    buffer: wgpu::Buffer,
    size: usize,
    size_bytes: u64,
    usage: wgpu::BufferUsages,
    label: String,
    pub(crate) context: Arc<GpuContext>,
}

impl GpuVector {
    pub(crate) fn new_internal(
        buffer: wgpu::Buffer,
        size: usize,
        usage: wgpu::BufferUsages,
        label: String,
        context: Arc<GpuContext>,
    ) -> Self {
        let size_bytes = (size * mem::size_of::<f64>()) as u64;
        Self { buffer, size, size_bytes, usage, label, context }
    }

    pub(crate) fn inner(&self) -> &wgpu::Buffer { &self.buffer }
    pub fn size(&self) -> usize { self.size }
    pub fn size_bytes(&self) -> u64 { self.size_bytes }
    pub fn usage(&self) -> wgpu::BufferUsages { self.usage }
    pub fn label(&self) -> &str { self.label.as_str() }
    pub fn as_entire_binding(&self) -> wgpu::BindingResource<'_> { self.buffer.as_entire_binding() }

    pub async fn read_contents(&self) -> Result<Vec<f64>, LcaCoreError> {
        self.context.read_buffer_to_cpu(self.inner(), self.size()).await
    }

    pub async fn write_contents(&self, data: &[f64]) -> Result<(), LcaCoreError> {
        if data.len() != self.size {
            return Err(LcaCoreError::InvalidDimensions(format!(
                "Data length ({}) does not match GpuVector size ({})",
                data.len(), self.size
            )));
        }
        self.context.write_buffer(self.inner(), data).await
    }

    pub fn clone_from(&mut self, source: &GpuVector) -> Result<(), LcaCoreError> {
        if self.size != source.size {
            return Err(LcaCoreError::InvalidDimensions(format!(
                "Vector sizes for clone_from mismatch: {} != {}",
                self.size, source.size
            )));
        }
        if self.size_bytes == 0 { return Ok(()); }
        if !self.usage.contains(wgpu::BufferUsages::COPY_DST) {
            return Err(LcaCoreError::UnsupportedOperation(
                "Destination vector buffer requires COPY_DST usage for clone_from".to_string(),
            ));
        }
        if !source.usage.contains(wgpu::BufferUsages::COPY_SRC) {
            return Err(LcaCoreError::UnsupportedOperation(
                "Source vector buffer requires COPY_SRC usage for clone_from".to_string(),
            ));
        }

        let mut encoder = self.context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GpuVector Clone From Encoder"),
        });
        encoder.copy_buffer_to_buffer(source.inner(), 0, self.inner(), 0, self.size_bytes);
        self.context.queue.submit(Some(encoder.finish()));

        cfg_if::cfg_if! {
            if #[cfg(not(target_arch = "wasm32"))] {
                let _ = self.context.device.poll(PollType::Wait);
            }
        }
        Ok(())
    }
}

impl Vector for GpuVector {
    type Value = f64;
    fn len(&self) -> usize { self.size() }
}
