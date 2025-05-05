use crate::context::GpuContext;
use crate::error::LcaCoreError;
use crate::ops; // Import internal ops module
use crate::sparse_matrix::{SparseMatrix, SparseMatrixGpu};
use crate::vector::GpuVector;
use cfg_if::cfg_if;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;
// Added error log
use std::{borrow::Cow, mem, sync::Arc}; // Added Cow, mem import

/// Marker trait for execution devices (CPU, GPU).
/// Needs Send + Sync to be safely passed between async tasks/threads.
pub trait Device: std::fmt::Debug {}

/// Represents a CPU execution device.
#[derive(Debug, Clone, Default)]
pub struct CpuDevice {}
impl Device for CpuDevice {}

/// Represents a GPU execution device, holding the WGPU context.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct GpuDevice {
    // Context is now internal and managed by GpuDevice.
    pub(crate) context: Arc<GpuContext>,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl GpuDevice {
    /// Creates a new GpuDevice, initializing the underlying WGPU context asynchronously.
    /// This is the primary entry point for using the GPU capabilities.
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub async fn new() -> Result<Self, LcaCoreError> {
        cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
              log::info!("GpuDevice::new() creating WASM context");
              let context = GpuContext::new_wasm().await?;
            } else {
              log::info!("GpuDevice::new() creating native context");
              let context = GpuContext::new().await?;
            }
        }
        log::info!("GpuDevice created successfully");
        Ok(Self {
            context: Arc::new(context),
        })
    }

    // --- Resource Creation ---

    /// Creates a GpuVector initialized with data from a CPU slice.
    pub fn create_vector(
        &self,
        label: &str,
        data: &[f64], // Allow specifying usage
    ) -> Result<GpuVector, LcaCoreError> {
        let size = data.len();
        if size == 0 {
            return Err(LcaCoreError::InvalidDimensions(
                "Cannot create GPU vector from empty slice".to_string(),
            ));
        }
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let buffer = self.context.create_gpu_buffer_with_data(
            label,
            bytemuck::cast_slice(data),
            // Ensure necessary usages are always present
            usage
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        Ok(GpuVector::new_internal(
            buffer,
            size,
            usage, // Store the original requested usage + mandatory ones
            String::from(label),
            Arc::clone(&self.context), // Pass Arc clone
        ))
    }

    /// Creates an empty (uninitialized) GpuVector with a specified size.
    pub fn create_empty_vector(&self, label: &str, size: usize) -> Result<GpuVector, LcaCoreError> {
        if size == 0 {
            return Err(LcaCoreError::InvalidDimensions(
                "Cannot create empty GPU vector with size 0".to_string(),
            ));
        }
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        let byte_size = (size * mem::size_of::<f64>()) as u64;
        let label_str = String::from(label);
        let buffer = self.context.create_empty_buffer(
            &label_str,
            byte_size,
            // Ensure necessary usages are always present
            usage
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            false, // Not mapped at creation
        );
        Ok(GpuVector::new_internal(
            buffer,
            size,
            usage, // Store the original requested usage + mandatory ones
            label_str,
            Arc::clone(&self.context), // Pass Arc clone
        ))
    }

    /// Creates a SparseMatrixGpu from a CPU SparseMatrix<f64>.
    pub fn create_sparse_matrix(
        &self,
        cpu_matrix: &SparseMatrix,
    ) -> Result<SparseMatrixGpu, LcaCoreError> {
        let rows = cpu_matrix.rows();
        let cols = cpu_matrix.cols();
        let nnz = cpu_matrix.nnz();

        // Convert indices to u32 for GPU compatibility
        let col_indices_u32: Vec<u32> = cpu_matrix.col_indices.iter().map(|&x| x as u32).collect();
        let row_ptr_u32: Vec<u32> = cpu_matrix.row_ptr.iter().map(|&x| x as u32).collect();

        // Create GPU buffers using the internal context's helper
        let values_contents = bytemuck::cast_slice(&cpu_matrix.values); // Access field directly
        let values_buffer = self.context.create_gpu_buffer_with_data(
            "GPU Sparse Matrix Values Buffer",
            values_contents,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );

        let col_indices_contents = bytemuck::cast_slice(&col_indices_u32);
        let col_indices_buffer = self.context.create_gpu_buffer_with_data(
            "GPU Sparse Matrix Col Indices Buffer",
            col_indices_contents,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        let row_pointers_contents = bytemuck::cast_slice(&row_ptr_u32);
        let row_pointers_buffer = self.context.create_gpu_buffer_with_data(
            "GPU Sparse Matrix Row Pointers Buffer",
            row_pointers_contents,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        // Use the internal constructor for SparseMatrixGpu
        Ok(SparseMatrixGpu::new_internal(
            rows,
            cols,
            nnz,
            values_buffer,
            col_indices_buffer,
            row_pointers_buffer,
            Arc::clone(&self.context), // Pass Arc clone
        ))
    }

    // --- Operations ---

    /// Performs `y = alpha * x + y` on the GPU (axpy).
    /// Currently only supports f64.
    pub async fn axpy(
        &self,
        alpha: f64,
        x: &GpuVector,
        y: &mut GpuVector,
    ) -> Result<(), LcaCoreError> {
        if x.size() != y.size() {
            return Err(LcaCoreError::InvalidDimensions(format!(
                "Vector sizes for axpy mismatch: {} != {}",
                x.size(),
                y.size()
            )));
        }
        // Call the internal implementation, passing the context from self
        ops::internal_axpy(&self.context, alpha, x, y).await
    }

    /// Calculates the dot product `x^T * y` on the GPU.
    /// Currently only supports f64.
    pub async fn dot(&self, x: &GpuVector, y: &GpuVector) -> Result<f64, LcaCoreError> {
        if x.size() != y.size() {
            return Err(LcaCoreError::InvalidDimensions(format!(
                "Vector sizes for dot product mismatch: {} != {}",
                x.size(),
                y.size()
            )));
        }
        // Call the internal implementation, passing the context from self
        ops::internal_dot(&self.context, x, y).await
    }

    /// Extracts the diagonal elements of a sparse matrix into a vector on the GPU.
    /// Assumes the matrix is in CSR format internally.
    pub async fn extract_diagonal(
        &self,
        matrix: &SparseMatrixGpu,
        output_vector: &mut GpuVector,
    ) -> Result<(), LcaCoreError> {
        let n = matrix.rows();
        if output_vector.size() != n {
            return Err(LcaCoreError::InvalidDimensions(format!(
                "Output vector size ({}) must match matrix rows ({}) for extract_diagonal",
                output_vector.size(),
                n
            )));
        }
        if n == 0 {
            return Ok(()); // Nothing to do for an empty matrix
        }

        let shader_source = Cow::Borrowed(include_str!("shaders/extract_diagonal.wgsl"));
        // Access create_shader_module via self.context.device and pass descriptor by value
        let shader_module =
            self.context
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    // Pass descriptor by value
                    label: Some("extract_diagonal shader"), // Corrected label type
                    source: wgpu::ShaderSource::Wgsl(shader_source),
                });

        // Bind group layout
        let bind_group_layout =
            self.context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Extract Diagonal Bind Group Layout"), // Corrected label type
                    entries: &[
                        // row_pointers (u32)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // col_indices (u32)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // values (f64)
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // diagonal_output (f64)
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false }, // Writable
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        // Pipeline layout
        let pipeline_layout =
            self.context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Extract Diagonal Pipeline Layout"), // Corrected label type
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        // Compute pipeline
        let pipeline =
            self.context
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Extract Diagonal Pipeline"), // Corrected label type
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None, // Ensure cache field is present
                });

        // Bind group
        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, // Set label to None
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: matrix.row_pointers_buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: matrix.col_indices_buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: matrix.values_buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        // Use the public as_entire_binding method directly on GpuVector
                        resource: output_vector.as_entire_binding(), // Corrected buffer access
                    },
                ],
            });

        // Command encoding and submission
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Extract Diagonal Command Encoder"), // Corrected label type
                });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Extract Diagonal Compute Pass"), // Corrected label type
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Calculate dispatch size
            let workgroup_size = 64; // Must match shader's @workgroup_size
            let dispatch_x = (n as u32 + workgroup_size - 1) / workgroup_size; // Ceiling division
            compute_pass.dispatch_workgroups(dispatch_x, 1, 1);
        }
        self.context.queue.submit(Some(encoder.finish()));

        // Note: No explicit await/poll needed here if using queue.submit.
        // The caller might need to await subsequent operations that depend on this result.
        Ok(())
    }

    /// Computes the element-wise inverse (1.0 / x) of a vector on the GPU.
    /// Handles near-zero values by outputting 0.0.
    pub async fn invert_elements(
        &self,
        input_vector: &GpuVector,
        output_vector: &mut GpuVector,
    ) -> Result<(), LcaCoreError> {
        let n = input_vector.size();
        if output_vector.size() != n {
            return Err(LcaCoreError::InvalidDimensions(format!(
                "Input ({}) and output ({}) vector sizes must match for invert_elements",
                n,
                output_vector.size()
            )));
        }
        if n == 0 {
            return Ok(()); // Nothing to do for empty vectors
        }

        let shader_source = Cow::Borrowed(include_str!("shaders/invert_elements.wgsl"));
        // Pass descriptor by value (remove &)
        let shader_module =
            self.context
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    // Pass descriptor by value
                    label: Some("invert_elements shader"), // Corrected label type
                    source: wgpu::ShaderSource::Wgsl(shader_source),
                });

        // Bind group layout
        let bind_group_layout =
            self.context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Invert Elements Bind Group Layout"), // Corrected label type
                    entries: &[
                        // input_vector (f64)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // output_vector (f64)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false }, // Writable
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        // Pipeline layout
        let pipeline_layout =
            self.context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Invert Elements Pipeline Layout"), // Corrected label type
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        // Compute pipeline
        let pipeline =
            self.context
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Invert Elements Pipeline"), // Corrected label type
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });

        // Bind group
        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, // Set label to None
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_vector.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output_vector.as_entire_binding(),
                    },
                ],
            });

        // Command encoding and submission
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Invert Elements Command Encoder"), // Corrected label type
                });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Invert Elements Compute Pass"), // Corrected label type
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Calculate dispatch size
            let workgroup_size = 64; // Must match shader's @workgroup_size
            let dispatch_x = (n as u32 + workgroup_size - 1) / workgroup_size; // Ceiling division
            compute_pass.dispatch_workgroups(dispatch_x, 1, 1);
        }
        self.context.queue.submit(Some(encoder.finish()));

        Ok(())
    }

    /// Computes the element-wise multiplication z = x * y of vectors on the GPU.
    pub async fn elementwise_mul(
        &self,
        x: &GpuVector,
        y: &GpuVector,
        z: &mut GpuVector,
    ) -> Result<(), LcaCoreError> {
        let n = x.size();
        if y.size() != n || z.size() != n {
            return Err(LcaCoreError::InvalidDimensions(format!(
                "Vector sizes must match for elementwise_mul: x={}, y={}, z={}",
                n,
                y.size(),
                z.size()
            )));
        }
        if n == 0 {
            return Ok(()); // Nothing to do for empty vectors
        }

        let shader_source = Cow::Borrowed(include_str!("shaders/elementwise_mul.wgsl"));
        let shader_module =
            self.context
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("elementwise_mul shader"),
                    source: wgpu::ShaderSource::Wgsl(shader_source),
                });

        // Bind group layout
        let bind_group_layout =
            self.context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Elementwise Mul Bind Group Layout"),
                    entries: &[
                        // x (f64)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // y (f64)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // z (f64)
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false }, // Writable
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        // Pipeline layout
        let pipeline_layout =
            self.context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Elementwise Mul Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        // Compute pipeline
        let pipeline =
            self.context
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Elementwise Mul Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });

        // Bind group
        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, // Use None for label
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: x.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: y.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: z.as_entire_binding(),
                    },
                ],
            });

        // Command encoding and submission
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Elementwise Mul Command Encoder"),
                });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Elementwise Mul Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Calculate dispatch size
            let workgroup_size = 64; // Must match shader's @workgroup_size
            let dispatch_x = (n as u32 + workgroup_size - 1) / workgroup_size; // Ceiling division
            compute_pass.dispatch_workgroups(dispatch_x, 1, 1);
        }
        self.context.queue.submit(Some(encoder.finish()));

        Ok(())
    }

    // --- Utility ---

    /// Returns the underlying WGPU adapter information. Useful for debugging or advanced configuration.
    // pub fn adapter_info(&self) -> wgpu::AdapterInfo {
    //     self.context.adapter.get_info()
    // }

    /// Returns the current GPU transfer statistics (bytes_to_gpu, bytes_from_gpu).
    pub fn get_transfer_stats(&self) -> TransferStats {
        let (bytes_to_gpu, bytes_from_gpu) = self.context.get_transfer_stats();
        TransferStats {
            bytes_to_gpu,
            bytes_from_gpu,
        }
    }

    /// Resets the GPU transfer statistics counters to zero.
    pub fn reset_transfer_stats(&self) {
        self.context.reset_transfer_stats();
    }
}
impl Device for GpuDevice {}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct TransferStats {
    pub bytes_to_gpu: u64,
    pub bytes_from_gpu: u64,
}
