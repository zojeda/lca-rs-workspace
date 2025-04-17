// This module contains the internal implementation of GPU compute operations.
// These functions are intended to be called by the operator overloads
// and methods defined in the main library file or on the GpuVector/SparseMatrixGpu types.

use crate::{
    context::GpuContext, error::LcaCoreError, sparse_matrix::SparseMatrixGpu, vector::GpuVector,
};
use bytemuck::{Pod, Zeroable};
use cfg_if::cfg_if;
use std::mem;

// --- Helper Structs (Internal) ---
#[repr(C, packed)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AxpyParams {
    // Removed pub
    alpha: f32,
    size: u32,
    _padding: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DotParams {
    // Removed pub
    size: u32,
    _padding: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SpmvParams {
    // Removed pub
    rows: u32,
    cols: u32,
    nnz: u32,
    _padding: u32,
}

// --- Internal GPU Operation Functions ---

/// Internal implementation for y = alpha * x + y on the GPU
pub(crate) async fn internal_axpy(
    // Renamed and changed visibility
    context: &GpuContext,
    alpha: f32,
    x: &GpuVector,     // Updated type
    y: &mut GpuVector, // Updated type
) -> Result<(), LcaCoreError> {
    let device = &context.device;
    let size = x.size() as u32;

    // Updated shader path
    let shader_source = include_str!("./shaders/axpy.wgsl");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Internal AXPY Shader"), // Updated label
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let params = AxpyParams {
        alpha,
        size,
        _padding: [0; 2],
    };
    let params_buffer = context.create_gpu_buffer_with_data(
        "Internal AXPY Params Buffer", // Updated label
        bytemuck::bytes_of(&params),
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Internal AXPY Bind Group Layout"), // Updated label
        entries: &[
            wgpu::BindGroupLayoutEntry {
                // params
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(mem::size_of::<AxpyParams>() as u64),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                // x
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                // y
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Internal AXPY Pipeline Layout"), // Updated label
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Internal AXPY Pipeline"), // Updated label
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("main"),
        cache: None,
        compilation_options: Default::default(),
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Internal AXPY Bind Group"), // Updated label
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: x.as_entire_binding(), // Use GpuVector method
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: y.as_entire_binding(), // Use GpuVector method
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Internal AXPY Encoder"), // Updated label
    });
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Internal AXPY Compute Pass"), // Updated label
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroup_size = 256; // TODO: Make configurable or query limits?
        let workgroup_count = (size + workgroup_size - 1) / workgroup_size;
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }
    context.queue.submit(std::iter::once(encoder.finish()));

    // Polling might be needed depending on how these ops are chained.
    // If called from an async fn that awaits another op, it might be okay.
    // If results are needed immediately on CPU, polling is required.
    cfg_if! { if #[cfg(not(target_arch = "wasm32"))] {
        // Consider if polling is always needed here or should be handled by the caller
        // context.device.poll(wgpu::Maintain::Wait);
    }}
    Ok(())
}

/// Internal implementation for calculating dot_product = x^T * y on the GPU
pub(crate) async fn internal_dot(
    // Renamed and changed visibility
    context: &GpuContext,
    x: &GpuVector, // Updated type
    y: &GpuVector, // Updated type
) -> Result<f32, LcaCoreError> {
    let device = &context.device;
    let queue = &context.queue;
    let size = x.size() as u32;
    if y.size() != x.size() {
        return Err(LcaCoreError::InvalidDimensions(format!(
            "Vector sizes for dot product mismatch: {} != {}",
            x.size(),
            y.size()
        )));
    }
    let workgroup_size = 256u32; // TODO: Configurable?

    // --- Pass 1 ---
    // Updated shader path
    let shader_pass1 = include_str!("./shaders/dot_product_pass1.wgsl");
    let module_pass1 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Internal Dot Product Pass 1 Shader"), // Updated label
        source: wgpu::ShaderSource::Wgsl(shader_pass1.into()),
    });
    let num_workgroups = (size + workgroup_size - 1) / workgroup_size;
    let partial_results_buffer = context.create_empty_buffer(
        "Internal Dot Product Partial Results Buffer", // Updated label
        num_workgroups as u64 * mem::size_of::<f32>() as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        false,
    );
    let params_pass1 = DotParams {
        size,
        _padding: [0; 3],
    };
    let params_buffer_pass1 = context.create_gpu_buffer_with_data(
        "Internal Dot Product Pass 1 Params Buffer", // Updated label
        bytemuck::bytes_of(&params_pass1),
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    let layout_pass1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Internal Dot Product Pass 1 Layout"), // Updated label
        entries: &[
            wgpu::BindGroupLayoutEntry {
                // params
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                // x
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                // y
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                // partial_results
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout_pass1 = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Internal Dot Product Pass 1 Pipeline Layout"), // Updated label
        bind_group_layouts: &[&layout_pass1],
        push_constant_ranges: &[],
    });
    let pipeline_pass1 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Internal Dot Product Pass 1 Pipeline"), // Updated label
        layout: Some(&pipeline_layout_pass1),
        module: &module_pass1,
        entry_point: Some("main"),
        cache: None,
        compilation_options: Default::default(),
    });
    let bind_group_pass1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Internal Dot Product Pass 1 Bind Group"), // Updated label
        layout: &layout_pass1,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer_pass1.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: x.as_entire_binding(),
            }, // Use GpuVector method
            wgpu::BindGroupEntry {
                binding: 2,
                resource: y.as_entire_binding(),
            }, // Use GpuVector method
            wgpu::BindGroupEntry {
                binding: 3,
                resource: partial_results_buffer.as_entire_binding(),
            },
        ],
    });

    // --- Pass 2 ---
    // Updated shader path
    let shader_pass2 = include_str!("./shaders/dot_product_pass2.wgsl");
    let module_pass2 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Internal Dot Product Pass 2 Shader"), // Updated label
        source: wgpu::ShaderSource::Wgsl(shader_pass2.into()),
    });
    let final_result_buffer = context.create_empty_buffer(
        "Internal Dot Product Final Result Buffer", // Updated label
        mem::size_of::<f32>() as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        false,
    );
    let params_pass2 = DotParams {
        size: num_workgroups,
        _padding: [0; 3],
    };
    let params_buffer_pass2 = context.create_gpu_buffer_with_data(
        "Internal Dot Product Pass 2 Params Buffer", // Updated label
        bytemuck::bytes_of(&params_pass2),
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    let layout_pass2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Internal Dot Product Pass 2 Layout"), // Updated label
        entries: &[
            wgpu::BindGroupLayoutEntry {
                // params
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                // partial_results
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                // final_result
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout_pass2 = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Internal Dot Product Pass 2 Pipeline Layout"), // Updated label
        bind_group_layouts: &[&layout_pass2],
        push_constant_ranges: &[],
    });
    let pipeline_pass2 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Internal Dot Product Pass 2 Pipeline"), // Updated label
        layout: Some(&pipeline_layout_pass2),
        module: &module_pass2,
        entry_point: Some("main"),
        cache: None,
        compilation_options: Default::default(),
    });
    let bind_group_pass2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Internal Dot Product Pass 2 Bind Group"), // Updated label
        layout: &layout_pass2,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer_pass2.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: partial_results_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: final_result_buffer.as_entire_binding(),
            },
        ],
    });

    // --- Command Encoding ---
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Internal Dot Product Encoder"), // Updated label
    });
    {
        // Pass 1
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Internal Dot Product Pass 1"), // Updated label
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline_pass1);
        compute_pass.set_bind_group(0, &bind_group_pass1, &[]);
        compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
    }
    if num_workgroups > 1 {
        // Pass 2 (if needed)
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Internal Dot Product Pass 2"), // Updated label
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline_pass2);
        compute_pass.set_bind_group(0, &bind_group_pass2, &[]);
        compute_pass.dispatch_workgroups(1, 1, 1);
    } else {
        // Copy if only one workgroup
        encoder.copy_buffer_to_buffer(
            &partial_results_buffer,
            0,
            &final_result_buffer,
            0,
            mem::size_of::<f32>() as u64,
        );
    }

    // --- Submit and Readback ---
    queue.submit(std::iter::once(encoder.finish()));
    // Readback is handled by the context helper, which includes polling/awaiting
    let result_vec = context
        .read_buffer_to_cpu::<f32>(&final_result_buffer, 1) // Read 1 f32 element
        .await?;

    result_vec.first().copied().ok_or_else(|| {
        LcaCoreError::Internal("Dot product readback returned empty vector".to_string())
    })
}

/// Internal implementation for y = A * x on the GPU using CSR format.
pub(crate) async fn internal_spmv(
    // Renamed and changed visibility
    context: &GpuContext,
    matrix: &SparseMatrixGpu,
    x: &GpuVector,     // Updated type
    y: &mut GpuVector, // Updated type
) -> Result<(), LcaCoreError> {
    let device = &context.device;
    let queue = &context.queue;
    let rows = matrix.rows() as u32;
    let cols = matrix.cols() as u32;
    let nnz = matrix.nnz() as u32;

    // Dimension check
    if matrix.cols() != x.size() {
        return Err(LcaCoreError::InvalidDimensions(format!(
            "Matrix cols ({}) do not match vector x size ({}) for SpMV",
            matrix.cols(),
            x.size()
        )));
    }
    if matrix.rows() != y.size() {
        return Err(LcaCoreError::InvalidDimensions(format!(
            "Matrix rows ({}) do not match vector y size ({}) for SpMV",
            matrix.rows(),
            y.size()
        )));
    }

    // Create an intermediate output buffer. Atomics are no longer needed as the shader writes directly.
    let y_output_buffer = context.create_empty_buffer(
        "Internal SpMV Output Buffer", // New label
        y.size_bytes(),                // Size for f32 elements
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, // Writable storage
        false,
    );

    // Updated shader path
    let shader_source = include_str!("./shaders/spmv_csr.wgsl");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Internal SpMV CSR Shader"), // Updated label
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let params = SpmvParams {
        rows,
        cols,
        nnz,
        _padding: 0,
    };
    let params_buffer = context.create_gpu_buffer_with_data(
        "Internal SpMV Params Buffer", // Updated label
        bytemuck::bytes_of(&params),
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );

    // Layout for the main SpMV calculation (Initialization layout removed)
    let bind_group_layout_main =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Internal SpMV CSR Main Bind Group Layout"), // Updated label
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    // params
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // row_pointers
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // col_indices
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // values
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // x
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // y_output (binding 5)
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    // Pipeline layout for the main SpMV calculation (Initialization layout removed)
    let pipeline_layout_main = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Internal SpMV CSR Main Pipeline Layout"), // Updated label
        bind_group_layouts: &[&bind_group_layout_main],
        push_constant_ranges: &[],
    });

    // Pipeline for the main SpMV calculation (Initialization pipeline removed)
    let pipeline_main = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Internal SpMV CSR Main Pipeline"), // Updated label
        layout: Some(&pipeline_layout_main),
        module: &shader_module,
        cache: None,
        entry_point: Some("main_spmv_csr_per_row"),
        compilation_options: Default::default(),
    });

    // Bind group for the main SpMV calculation (Initialization bind group removed)
    let bind_group_main = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Internal SpMV CSR Main Bind Group"), // Updated label
        layout: &bind_group_layout_main,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: matrix.row_pointers_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: matrix.col_indices_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: matrix.values_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: x.as_entire_binding(),
            }, // Use GpuVector method
            wgpu::BindGroupEntry {
                binding: 5,
                resource: y_output_buffer.as_entire_binding(),
            }, // Use y_output_buffer
        ],
    });

    // --- Command Encoding ---
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Internal SpMV Encoder"), // Updated label
    });
    let workgroup_size = 256; // TODO: Configurable?
                              // Initialization pass removed
    {
        // Main SpMV calculation
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Internal SpMV CSR Main Pass"), // Updated label
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline_main);
        compute_pass.set_bind_group(0, &bind_group_main, &[]);
        // Dispatch one workgroup per row for this shader implementation
        let workgroup_count_rows = (rows + workgroup_size - 1) / workgroup_size; // Assuming workgroup size matches shader
        compute_pass.dispatch_workgroups(workgroup_count_rows, 1, 1);
    }
    // Copy result back from the intermediate output buffer to the final output buffer y
    encoder.copy_buffer_to_buffer(
        &y_output_buffer,
        0u64,
        y.inner(),
        0u64,
        y.size_bytes() as u64,
    ); // Explicitly cast offsets/size

    // --- Submit ---
    queue.submit(std::iter::once(encoder.finish()));

    // Polling might be needed here too, similar consideration as axpy
    cfg_if! { if #[cfg(not(target_arch = "wasm32"))] {
        // context.device.poll(wgpu::Maintain::Wait);
    }}
    Ok(())
}
