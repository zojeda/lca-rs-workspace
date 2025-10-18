#![cfg(all(feature = "pardiso", not(target_arch = "wasm32")))]
use lca_rs::EvalLCASystem;
use lca_rs::core::{DemandItem, LcaSystem};
use std::error::Error;

mod shared;
// use paths directly via `shared::...` to avoid unused import warnings

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .filter_module("wgpu", log::LevelFilter::Off)
        .filter_module("naga", log::LevelFilter::Off)
        .init();

    let water_bottle_lca_system = shared::create_water_botle_lca_system()?;
    let ecoinvent_system = shared::load_ecoinvent_lca_system()?;
    let lca_system = LcaSystem::combine(vec![ecoinvent_system, water_bottle_lca_system])?;

    let demand = vec![DemandItem::new(
        "Water bottle LCA::Drinking water from a bottle|Drinking 1 liter from a water bottle"
            .to_string(),
        1.0,
    )];

    let eval_system: EvalLCASystem = lca_system.try_into()?;
    // Use CPU PARDISO backend; default matrix type General; user can control threads via MKL env
    let eval_system = eval_system
        .with_demand(demand)
        .with_evaluation_methods(vec![
            "ecoinvent-3.11::EF v3.1|climate change|global warming potential (GWP100)".to_string(),
            "ecoinvent-3.11::EF v3.1|climate change: biogenic|global warming potential (GWP100)".to_string(),
            "ecoinvent-3.11::EF v3.1|climate change: fossil|global warming potential (GWP100)".to_string(),
            "ecoinvent-3.11::EF v3.1|climate change: land use and land use change|global warming potential (GWP100)".to_string(),
        ])
        .with_cpu_pardiso(None, None);

    let start_time = std::time::Instant::now();
    let lca_result = eval_system.evaluate().await?;
    let elapsed_time = start_time.elapsed();
    println!("Elapsed time (PARDISO CPU solve): {:?}", elapsed_time);
    println!("Result length: {}", lca_result.len());
    println!("Result: {:?}", lca_result);
    Ok(())
}
