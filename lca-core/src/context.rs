use crate::error::LcaCoreError; // Use error from this crate
use bytemuck::{Pod, Zeroable};
use cfg_if::cfg_if;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use wgpu::{util::DeviceExt, PollType}; // For create_buffer_init

/// Wrapper for WGPU instance, adapter, device, and queue, including transfer counters.
/// This is internal to the lca-lsolver crate.
#[derive(Debug, Clone)]
pub(crate) struct GpuContext {
    // Fields remain accessible within the crate
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
    /// Tracks bytes transferred from CPU to GPU via instrumented methods.
    pub(crate) bytes_to_gpu: Arc<AtomicU64>,
    /// Tracks bytes transferred from GPU to CPU via instrumented methods.
    pub(crate) bytes_from_gpu: Arc<AtomicU64>,
}

impl GpuContext {
    /// Initializes the WGPU context asynchronously (Native Version).
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
        log::info!("Adapter Features: {:?}", adapter.features()); // Log supported features

        log::debug!("Requesting native device and queue with adjusted limits");
        let mut limits = wgpu::Limits::default().using_resolution(adapter.limits());
        // Ensure we can use storage buffers in compute shaders for SpMV etc.
        limits.max_storage_buffers_per_shader_stage =
            limits.max_storage_buffers_per_shader_stage.max(4); // Request at least 4, or keep adapter default if higher
        log::debug!("Adjusted limits: {:?}", limits);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("lca_core device"),
                trace: wgpu::Trace::Off,
                memory_hints: wgpu::MemoryHints::Performance,
                required_features: wgpu::Features::SHADER_F64, // Enable f32 support in shaders
                required_limits: limits,                       // Use adjusted limits
            })
            .await
            .map_err(|e| LcaCoreError::WgpuInitError(format!("Failed to request device: {}", e)))?;

        log::info!("Device and queue obtained successfully");
        log::info!("Device Features: {:?}", device.features()); // Log enabled features

        // Explicitly check if SHADER_F64 was granted
        if !device.features().contains(wgpu::Features::SHADER_F64) {
            log::warn!("Requested SHADER_F64 feature was NOT granted by the device!");
            // return Err(LsolverError::WgpuInitError("Required SHADER_F64 feature not supported".to_string()));
        } else {
            log::info!("SHADER_F64 feature successfully enabled.");
        }

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            bytes_to_gpu: Arc::new(AtomicU64::new(0)), // Initialize counter
            bytes_from_gpu: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Initializes the WGPU context asynchronously (WASM Version).
    #[cfg(feature = "wasm")]
    pub(crate) async fn new_wasm() -> Result<Self, LcaCoreError> {
        log::info!("Initializing WASM WGPU context");

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL, // WebGPU first, fallback to WebGL
            ..Default::default()
        });

        log::debug!("Requesting WASM adapter");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None, // No surface needed for compute
                force_fallback_adapter: false,
            })
            .await
            .map_err(|_| {
                LcaCoreError::WgpuInitError("No suitable WASM adapter found".to_string())
            })?;

        log::info!("Selected WASM Adapter: {:?}", adapter.get_info());
        log::info!("WASM Adapter Features: {:?}", adapter.features()); // Log supported features

        log::debug!("Requesting WASM device and queue with adjusted limits");
        let mut limits = wgpu::Limits::downlevel_webgl2_defaults() // Use WebGL2 defaults for broader compatibility
            .using_resolution(adapter.limits());
        // Ensure we can use storage buffers in compute shaders for SpMV etc.
        limits.max_storage_buffers_per_shader_stage =
            limits.max_storage_buffers_per_shader_stage.max(4); // Request at least 4, or keep adapter default if higher
        log::debug!("Adjusted WASM limits: {:?}", limits);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("lca_core WASM device"), // Updated label
                trace: wgpu::Trace::Off,
                memory_hints: wgpu::MemoryHints::Performance,
                required_features: wgpu::Features::SHADER_F64,
                required_limits: limits, // Use adjusted limits
            })
            .await
            .map_err(|e| {
                LcaCoreError::WgpuInitError(format!("Failed to request WASM device: {}", e))
            })?;

        log::info!("WASM Device and queue obtained successfully");
        log::info!("WASM Device Features: {:?}", device.features()); // Log enabled features

        // Explicitly check if SHADER_F64 was granted
        if !device.features().contains(wgpu::Features::SHADER_F64) {
            log::warn!("Requested SHADER_F64 feature was NOT granted by the WASM device!");
            // return Err(LsolverError::WgpuInitError("Required SHADER_F64 feature not supported by WASM device".to_string()));
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
    /// Note: This now takes raw bytes (`contents`) directly.
    pub(crate) fn create_gpu_buffer_with_data(
        &self,
        label: &str,
        contents: &[u8], // Takes raw bytes
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        let byte_len = contents.len() as u64;
        log::debug!("Creating GPU buffer '{}' with {} bytes", label, byte_len);
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents, // Use the raw bytes directly
                usage,
            });
        // Increment the counter
        self.bytes_to_gpu.fetch_add(byte_len, Ordering::Relaxed);
        log::trace!(
            "bytes_to_gpu incremented by {}, now: {}",
            byte_len,
            self.bytes_to_gpu.load(Ordering::Relaxed)
        );
        buffer
    }

    /// Helper function to write data from CPU slice `data` to an existing GPU `buffer`.
    /// Tracks the transfer size.
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
                "Direct writing to MAP_WRITE buffers not implemented via this helper. Use queue.write_buffer.".to_string()
            ));
        }
        if !buffer.usage().contains(wgpu::BufferUsages::COPY_DST) {
            return Err(LcaCoreError::Internal(
                "Target buffer must have COPY_DST usage".to_string(),
            ));
        }

        // Reverted log message: Cannot easily access label from wgpu::Buffer ref here.
        log::debug!("Writing {} bytes to buffer", byte_len);

        // Use queue.write_buffer for direct writing (efficient for non-mappable buffers)
        self.queue
            .write_buffer(buffer, 0, bytemuck::cast_slice(data));

        // Increment counter
        self.bytes_to_gpu.fetch_add(byte_len, Ordering::Relaxed);
        log::trace!(
            "bytes_to_gpu incremented by {} (write_buffer), now: {}",
            byte_len,
            self.bytes_to_gpu.load(Ordering::Relaxed)
        );

        // For WASM, we might need to ensure the queue is flushed or work is submitted.
        // However, write_buffer itself doesn't require explicit submission like command encoders.
        // If subsequent operations depend on this write, their submission will handle it.
        // Consider adding a queue.submit([]) if immediate effect is needed before other async ops.
        // self.queue.submit(None); // Example if needed

        // Polling might be needed on native if strict ordering is required before CPU proceeds,
        // but often not necessary just for a write if subsequent GPU commands handle dependencies.
        // cfg_if! {
        //     if not(target_arch = "wasm32") {
        //         self.device.poll(wgpu::Maintain::Wait); // Or Maintain::Poll
        //     }
        // }

        Ok(())
    }

    /// Helper to create an empty GPU buffer (useful for shader outputs).
    /// Does not count towards `bytes_to_gpu` as no data is initially transferred.
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
    ///
    /// # Arguments
    /// * `buffer` - The GPU buffer to read from (must have `COPY_SRC` usage).
    /// * `element_count` - The number of elements of type `T` expected in the buffer.
    ///
    /// # Returns
    /// A `Vec<T>` containing the data read from the buffer.
    ///
    /// # Errors
    /// Returns `LsolverError` if the buffer read fails or if the data size doesn't match expectations.
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
            return Ok(Vec::new()); // Handle empty buffer case
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

        // Map the staging buffer
        log::debug!("Mapping staging buffer");
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            if let Err(e) = sender.send(result) {
                // Log error if receiver is dropped, indicates issue elsewhere
                log::error!("Failed to send map result back: {:?}", e);
            }
        });

        // Increment counter *before* waiting for the actual data transfer
        self.bytes_from_gpu.fetch_add(size_bytes, Ordering::Relaxed);
        log::trace!(
            "bytes_from_gpu incremented by {}, now: {}",
            size_bytes,
            self.bytes_from_gpu.load(Ordering::Relaxed)
        );

        // Wait for mapping - Poll only on native platforms
        cfg_if! {
            if #[cfg(not(target_arch = "wasm32"))] {
                log::debug!("Polling device to wait for buffer mapping (readback - native)");
                let _ = self.device.poll(PollType::Wait); // Wait for GPU to finish copy and map
            } else {
                 log::debug!("Awaiting buffer mapping (readback - wasm)");
                 // On WASM, just awaiting the receiver is sufficient and avoids blocking
            }
        }

        match receiver.await {
            Ok(Ok(())) => {
                log::debug!("Staging buffer mapped successfully");
                let result = {
                    // Create a scope to manage the lifetime of `data`
                    // Get the mapped view
                    let data = buffer_slice.get_mapped_range();
                    let mapped_len = data.len();

                    // Check size consistency before casting
                    if mapped_len != size_bytes as usize {
                        // Error path: drop(data) happens implicitly at scope end if we return early
                        // We still need to unmap before returning the error.
                        drop(data); // Drop explicitly before unmap
                        staging_buffer.unmap();
                        return Err(LcaCoreError::Internal(format!(
                            "Mapped data size ({}) does not match expected byte size ({})",
                            mapped_len, size_bytes
                        )));
                    }

                    // Success path: Use bytemuck for safe casting
                    // Ensure the slice length is a multiple of the element size
                    if mapped_len % element_size != 0 {
                        drop(data);
                        staging_buffer.unmap();
                        return Err(LcaCoreError::Internal(format!(
                            "Mapped data size ({}) is not a multiple of element size ({})",
                            mapped_len, element_size
                        )));
                    }
                    // Convert the mapped data to a slice of T and then to Vec<T>
                    let cast_result: Vec<T> = bytemuck::cast_slice(&data).to_vec();
                    // `data` will be dropped automatically at the end of this inner scope
                    cast_result // Return the result from the scope
                }; // `data` is dropped here

                // Now that `data` (BufferView) is dropped, we can unmap
                staging_buffer.unmap();
                log::debug!(
                    "Buffer readback complete and unmapped ({} bytes)",
                    size_bytes
                );
                Ok(result)
            }
            Ok(Err(e)) => {
                log::error!("Failed to map buffer: {:?}", e);
                // Note: No need to unmap here, as mapping failed.
                Err(LcaCoreError::WgpuError(format!(
                    "Buffer mapping failed: {}",
                    e
                )))
            }
            Err(_) => {
                log::error!("Channel receive error during buffer mapping");
                // Note: No need to unmap here, as mapping failed.
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
