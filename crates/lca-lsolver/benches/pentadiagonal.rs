use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
#[cfg(all(feature = "pardiso", not(target_arch = "wasm32")))]
use lca_core::devices::CpuDevice;
use lca_core::{device::GpuDevice, sparse_matrix::Triplete};
#[cfg(all(feature = "pardiso", not(target_arch = "wasm32")))]
use lca_lsolver::algorithms::pardiso_direct::{PardisoConfig, PardisoDirect, PardisoMatrixType};
use lca_lsolver::{
    SparseMatrix,
    algorithms::{BiCGSTAB, ConjugateGradient, SolveAlgorithm},
};

// Utilities copied/adapted from examples/pentadiagonal_solve.rs
fn create_pentadiagonal_matrix(n: usize) -> SparseMatrix {
    let mut triplets = Vec::new();

    for i in 0..n {
        let row = i as usize;
        if i >= 2 {
            triplets.push(Triplete::new(row, i - 2, -0.5));
        }
        if i >= 1 {
            triplets.push(Triplete::new(row, i - 1, -1.0));
        }
        triplets.push(Triplete::new(row, i, 4.0));
        if i + 1 < n {
            triplets.push(Triplete::new(row, i + 1, -1.0));
        }
        if i + 2 < n {
            triplets.push(Triplete::new(row, i + 2, -0.5));
        }
    }

    SparseMatrix::from_triplets(n, n, triplets).expect("Failed to create sparse matrix from COO")
}

fn create_sin_vector(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64 / n as f64).sin()).collect()
}

trait SolveStats {
    fn iterations(&self) -> usize;
    fn residual_norm(&self) -> f64;
}

impl SolveStats for lca_lsolver::algorithms::BiCGSTABMetadata {
    fn iterations(&self) -> usize {
        self.iterations
    }
    fn residual_norm(&self) -> f64 {
        self.residual_norm
    }
}

use lca_lsolver::algorithms::gpu_sparse_cg_checked::ConjugateGradientMetadata;

impl SolveStats for ConjugateGradientMetadata {
    fn iterations(&self) -> usize {
        self.iterations
    }
    fn residual_norm(&self) -> f64 {
        self.residual_norm
    }
}

async fn bench_one_algorithm<A>(
    device: &GpuDevice,
    a_gpu: &lca_core::SparseMatrixGpu,
    b: &[f64],
    mut make_algo: impl FnMut() -> A,
) -> (usize, f64)
where
    A: SolveAlgorithm<GpuDevice, lca_core::SparseMatrixGpu, Value = f64>,
    A::Metadata: SolveStats,
{
    let algo = make_algo();
    let result = algo.solve(device, a_gpu, b).await.expect("solver failed");
    (
        result.metadata.iterations(),
        result.metadata.residual_norm(),
    )
}

fn do_benches(c: &mut Criterion) {
    // Initialize logger once per process (noop if RUST_LOG not set)
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .try_init();

    // Problem sizes
    let sizes = [256usize, 512, 1024, 2048];

    let mut group = c.benchmark_group("pentadiagonal_solve");

    for &n in &sizes {
        // Prepare problem on CPU
        let a_cpu = create_pentadiagonal_matrix(n);
        let b = create_sin_vector(n);

        // Create device and upload matrix once per size
        let (device, a_gpu) = pollster::block_on(async {
            let device = GpuDevice::new().await.expect("gpu device");
            let a_gpu = device.create_sparse_matrix(&a_cpu).expect("upload matrix");
            (device, a_gpu)
        });

        // We'll reuse the same b slice each iteration; solver uploads as needed
        group.throughput(Throughput::Elements(n as u64));

        // BiCGSTAB without preconditioner
        group.bench_with_input(
            BenchmarkId::new("BiCGSTAB-no-precond", n),
            &n,
            |bencher, &_n| {
                bencher.iter(|| {
                    // reset transfer stats for fairness
                    device.reset_transfer_stats();
                    pollster::block_on(async {
                        let (_iters, _res) = bench_one_algorithm(&device, &a_gpu, &b, || {
                            BiCGSTAB::with_params(1e-3, n * 5, false)
                        })
                        .await;
                    });
                });
            },
        );

        // BiCGSTAB with Jacobi preconditioner
        group.bench_with_input(
            BenchmarkId::new("BiCGSTAB-jacobi", n),
            &n,
            |bencher, &_n| {
                bencher.iter(|| {
                    device.reset_transfer_stats();
                    pollster::block_on(async {
                        let (_iters, _res) = bench_one_algorithm(&device, &a_gpu, &b, || {
                            BiCGSTAB::with_params(1e-3, n * 5, true)
                        })
                        .await;
                    });
                });
            },
        );

        // Conjugate Gradient (assumes SPD)
        group.bench_with_input(
            BenchmarkId::new("ConjugateGradient", n),
            &n,
            |bencher, &_n| {
                bencher.iter(|| {
                    device.reset_transfer_stats();
                    pollster::block_on(async {
                        let (_iters, _res) = bench_one_algorithm(&device, &a_gpu, &b, || {
                            ConjugateGradient::with_params(1e-3, n * 5)
                        })
                        .await;
                    });
                });
            },
        );

        // CPU PARDISO Direct solver (native only), matrix is SPD so use Spd type.
        // Enable by setting env LCA_BENCH_PARDISO=1 to avoid CI/local crashes when MKL is not configured.
        #[cfg(all(feature = "pardiso", not(target_arch = "wasm32")))]
        if std::env::var("LCA_BENCH_PARDISO").ok().as_deref() == Some("1") {
            group.bench_with_input(BenchmarkId::new("PARDISO-direct", n), &n, |bencher, &_n| {
                bencher.iter(|| {
                    let cfg = PardisoConfig {
                        matrix_type: PardisoMatrixType::Spd,
                        max_threads: None,
                    };
                    let solver = PardisoDirect::new(cfg);
                    let cpu = CpuDevice::default();
                    let _ = pollster::block_on(async { solver.solve(&cpu, &a_cpu, &b).await })
                        .expect("PARDISO solve failed");
                });
            });
        }
    }

    group.finish();
}

criterion_group!(benches, do_benches);
criterion_main!(benches);
