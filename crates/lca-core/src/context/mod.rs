use crate::error::LcaCoreError; // Use error from this crate
use bytemuck::{Pod, Zeroable};
use cfg_if::cfg_if;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
#[cfg(not(target_arch = "wasm32"))]
use wgpu::PollType;
use wgpu::util::DeviceExt; // For create_buffer_init

pub mod transfer_stats;

/// Wrapper for WGPU instance, adapter, device, and queue, including transfer counters.
/// This is internal to the crate.
#[derive(Debug, Clone)]
pub(crate) struct GpuContext {
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
    /// Tracks bytes transferred from CPU to GPU via instrumented methods.
    pub(crate) bytes_to_gpu: Arc<AtomicU64>,
    /// Tracks bytes transferred from GPU to CPU via instrumented methods.
    pub(crate) bytes_from_gpu: Arc<AtomicU64>,
}

impl GpuContext {
    /// Initializes the WGPU context asynchronously (Native Version).
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) async fn new() -> Result<Self, LcaCoreError> {
        log::info!("Initializing native WGPU context");

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY, // Vulkan, Metal, DX12
            ..Default::default()
        });

        log::debug!("Requesting native adapter");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None, // No surface needed for compute
                force_fallback_adapter: false,
            })
            .await
            .map_err(|_| {
                LcaCoreError::WgpuInitError("No suitable native adapter found".to_string())
            })?;

        log::info!("Selected Native Adapter: {:?}", adapter.get_info());
        log::info!("Adapter Features: {:?}", adapter.features());

        log::debug!("Requesting native device and queue with adjusted limits");
        let mut limits = wgpu::Limits::default().using_resolution(adapter.limits());
        // Ensure we can use storage buffers in compute shaders for SpMV etc.
        limits.max_storage_buffers_per_shader_stage =
            limits.max_storage_buffers_per_shader_stage.max(4);
        log::debug!("Adjusted limits: {:?}", limits);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("lca_core device"),
                trace: wgpu::Trace::Off,
                memory_hints: wgpu::MemoryHints::Performance,
                required_features: wgpu::Features::SHADER_F64, // Enable f64 support in shaders
                required_limits: limits,                       // Use adjusted limits
            })
            .await
            .map_err(|e| LcaCoreError::WgpuInitError(format!("Failed to request device: {}", e)))?;

        log::info!("Device and queue obtained successfully");
        log::info!("Device Features: {:?}", device.features());

        // Explicitly check if SHADER_F64 was granted
        if !device.features().contains(wgpu::Features::SHADER_F64) {
            log::warn!("Requested SHADER_F64 feature was NOT granted by the device!");
        } else {
            log::info!("SHADER_F64 feature successfully enabled.");
        }

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            bytes_to_gpu: Arc::new(AtomicU64::new(0)),
            bytes_from_gpu: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Initializes the WGPU context asynchronously (WASM Version).
    #[cfg(target_arch = "wasm32")]
    pub(crate) async fn new_wasm() -> Result<Self, LcaCoreError> {
        log::info!("Initializing WASM WGPU context");

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL,
            ..Default::default()
        });

        log::debug!("Requesting WASM adapter");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|_| {
                LcaCoreError::WgpuInitError("No suitable WASM adapter found".to_string())
            })?;

        log::info!("Selected WASM Adapter: {:?}", adapter.get_info());
        log::info!("WASM Adapter Features: {:?}", adapter.features());

        log::debug!("Requesting WASM device and queue with adjusted limits");
        let mut limits =
            wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits());
        limits.max_storage_buffers_per_shader_stage =
            limits.max_storage_buffers_per_shader_stage.max(4);
        log::debug!("Adjusted WASM limits: {:?}", limits);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("lca_core WASM device"),
                trace: wgpu::Trace::Off,
                memory_hints: wgpu::MemoryHints::Performance,
                required_features: wgpu::Features::SHADER_F64,
                required_limits: limits,
            })
            .await
            .map_err(|e| {
                LcaCoreError::WgpuInitError(format!("Failed to request WASM device: {}", e))
            })?;

        log::info!("WASM Device and queue obtained successfully");
        log::info!("WASM Device Features: {:?}", device.features());

        if !device.features().contains(wgpu::Features::SHADER_F64) {
            log::warn!("Requested SHADER_F64 feature was NOT granted by the WASM device!");
        } else {
            log::info!("SHADER_F64 feature successfully enabled on WASM device.");
        }

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            bytes_to_gpu: Arc::new(AtomicU64::new(0)),
            bytes_from_gpu: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Helper to create a GPU buffer with initial data and track the transfer size.
    pub(crate) fn create_gpu_buffer_with_data(
        &self,
        label: &str,
        contents: &[u8],
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        let byte_len = contents.len() as u64;
        log::debug!("Creating GPU buffer '{}' with {} bytes", label, byte_len);
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage,
            });
        self.bytes_to_gpu.fetch_add(byte_len, Ordering::Relaxed);
        log::trace!(
            "bytes_to_gpu incremented by {}, now: {}",
            byte_len,
            self.bytes_to_gpu.load(Ordering::Relaxed)
        );
        buffer
    }

    /// Helper function to write data from CPU slice `data` to an existing GPU `buffer`.
    pub(crate) async fn write_buffer<T: Pod>(
        &self,
        buffer: &wgpu::Buffer,
        data: &[T],
    ) -> Result<(), LcaCoreError> {
        let byte_len = (data.len() * std::mem::size_of::<T>()) as u64;
        if byte_len == 0 {
            log::debug!("Skipping write for 0 bytes");
            return Ok(());
        }
        if buffer.size() < byte_len {
            return Err(LcaCoreError::Internal(format!(
                "Target buffer size ({}) is smaller than data size ({})",
                buffer.size(),
                byte_len
            )));
        }
        if buffer.usage().contains(wgpu::BufferUsages::MAP_WRITE) {
            return Err(LcaCoreError::Internal(
                "Direct writing to MAP_WRITE buffers not implemented via this helper. Use queue.write_buffer.".to_string(),
            ));
        }
        if !buffer.usage().contains(wgpu::BufferUsages::COPY_DST) {
            return Err(LcaCoreError::Internal(
                "Target buffer must have COPY_DST usage".to_string(),
            ));
        }

        log::debug!("Writing {} bytes to buffer", byte_len);

        self.queue
            .write_buffer(buffer, 0, bytemuck::cast_slice(data));

        self.bytes_to_gpu.fetch_add(byte_len, Ordering::Relaxed);
        log::trace!(
            "bytes_to_gpu incremented by {} (write_buffer), now: {}",
            byte_len,
            self.bytes_to_gpu.load(Ordering::Relaxed)
        );

        Ok(())
    }

    /// Helper to create an empty GPU buffer (useful for shader outputs).
    pub(crate) fn create_empty_buffer(
        &self,
        label: &str,
        size: u64,
        usage: wgpu::BufferUsages,
        mapped_at_creation: bool,
    ) -> wgpu::Buffer {
        log::debug!("Creating empty GPU buffer '{}' of size {}", label, size);
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation,
        })
    }

    /// Reads the contents of a GPU buffer back to the CPU.
    pub(crate) async fn read_buffer_to_cpu<T: Pod + Zeroable>(
        &self,
        buffer: &wgpu::Buffer,
        element_count: usize,
    ) -> Result<Vec<T>, LcaCoreError> {
        let element_size = std::mem::size_of::<T>();
        if element_size == 0 {
            return Err(LcaCoreError::Internal(
                "Cannot read zero-sized types".to_string(),
            ));
        }
        let size_bytes = (element_count * element_size) as u64;

        if size_bytes == 0 {
            log::debug!("Skipping readback for 0 bytes");
            return Ok(Vec::new());
        }
        if buffer.size() < size_bytes {
            return Err(LcaCoreError::Internal(format!(
                "GPU buffer size ({}) is smaller than expected size based on element count ({})",
                buffer.size(),
                size_bytes
            )));
        }

        log::debug!(
            "Creating staging buffer for readback ({} bytes)",
            size_bytes
        );
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_buffer_for_readback"),
            size: size_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        log::debug!("Encoding buffer copy command for {} bytes", size_bytes);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("read_buffer_encoder"),
            });

        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size_bytes);

        log::debug!("Submitting buffer copy command");
        self.queue.submit(std::iter::once(encoder.finish()));

        log::debug!("Mapping staging buffer");
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            if let Err(e) = sender.send(result) {
                log::error!("Failed to send map result back: {:?}", e);
            }
        });

        self.bytes_from_gpu.fetch_add(size_bytes, Ordering::Relaxed);
        log::trace!(
            "bytes_from_gpu incremented by {}, now: {}",
            size_bytes,
            self.bytes_from_gpu.load(Ordering::Relaxed)
        );

        cfg_if! {
            if #[cfg(not(target_arch = "wasm32"))] {
                log::debug!("Polling device to wait for buffer mapping (readback - native)");
                let _ = self.device.poll(PollType::Wait);
            } else {
                 log::debug!("Awaiting buffer mapping (readback - wasm)");
            }
        }

        match receiver.await {
            Ok(Ok(())) => {
                log::debug!("Staging buffer mapped successfully");
                let result = {
                    let data = buffer_slice.get_mapped_range();
                    let mapped_len = data.len();

                    if mapped_len != size_bytes as usize {
                        drop(data);
                        staging_buffer.unmap();
                        return Err(LcaCoreError::Internal(format!(
                            "Mapped data size ({}) does not match expected byte size ({})",
                            mapped_len, size_bytes
                        )));
                    }

                    if mapped_len % element_size != 0 {
                        drop(data);
                        staging_buffer.unmap();
                        return Err(LcaCoreError::Internal(format!(
                            "Mapped data size ({}) is not a multiple of element size ({})",
                            mapped_len, element_size
                        )));
                    }
                    let cast_result: Vec<T> = bytemuck::cast_slice(&data).to_vec();
                    cast_result
                };

                staging_buffer.unmap();
                log::debug!(
                    "Buffer readback complete and unmapped ({} bytes)",
                    size_bytes
                );
                Ok(result)
            }
            Ok(Err(e)) => {
                log::error!("Failed to map buffer: {:?}", e);
                Err(LcaCoreError::WgpuError(format!(
                    "Buffer mapping failed: {}",
                    e
                )))
            }
            Err(_) => {
                log::error!("Channel receive error during buffer mapping");
                Err(LcaCoreError::Internal(
                    "Channel receive error during buffer mapping".to_string(),
                ))
            }
        }
    }

    /// Returns the current transfer statistics.
    pub(crate) fn get_transfer_stats(&self) -> (u64, u64) {
        (
            self.bytes_to_gpu.load(Ordering::Relaxed),
            self.bytes_from_gpu.load(Ordering::Relaxed),
        )
    }

    /// Resets the transfer statistics counters to zero.
    pub(crate) fn reset_transfer_stats(&self) {
        self.bytes_to_gpu.store(0, Ordering::Relaxed);
        self.bytes_from_gpu.store(0, Ordering::Relaxed);
        log::info!("GPU transfer counters reset.");
    }
}
