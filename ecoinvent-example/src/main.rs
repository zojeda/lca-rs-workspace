use lca_rs::EvalLCASystem;
use lca_rs::core::{DemandItem, LcaSystem};
use std::{error::Error, vec};
mod shared;
use shared::{create_water_botle_lca_system, load_ecoinvent_lca_system};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .filter_module("wgpu", log::LevelFilter::Off)
        .filter_module("naga", log::LevelFilter::Off)
        .init();

    // Load the LCA system
    let water_bottle_lca_system = create_water_botle_lca_system()?;
    water_bottle_lca_system.a_links().iter().for_each(|link| {
        println!("A link: {:?}", link);
    });
    println!("Water bottle LCA system loaded");

    let ecoinvent_system = load_ecoinvent_lca_system()?;
    println!("EcoInvent system loaded");

    let lca_system = LcaSystem::combine(vec![ecoinvent_system, water_bottle_lca_system])?;

    println!(
        "Combined A matrix: {:?}",
        lca_system.a_matrix().matrix().dims()
    );
    println!(
        "Combined B matrix: {:?}",
        lca_system.b_matrix().matrix().dims()
    );
    println!(
        "Combined C matrix: {:?}",
        lca_system.c_matrix().matrix().dims()
    );

    for triplete in lca_system.a_matrix().matrix().iter() {
        if triplete.row() >= 25412 {
            println!("A matrix triplet: {:?}", triplete);
        }
    }
    println!(
        "B matrix row id 562: {:?}",
        lca_system.b_matrix().row_id(562)
    );

    let demand = vec![DemandItem::new(
        "Water bottle LCA::Drinking water from a bottle|Drinking 1 liter from a water bottle"
            .to_string(),
        1.0,
    )];
    let start_time = std::time::Instant::now();
    let eval_system: EvalLCASystem = lca_system.try_into()?;
    let eval_system = eval_system
      .with_demand(demand)
      .with_evaluation_methods(vec![
        "ecoinvent-3.11::EF v3.1|climate change|global warming potential (GWP100)".to_string(),
        "ecoinvent-3.11::EF v3.1|climate change: biogenic|global warming potential (GWP100)".to_string(),
        "ecoinvent-3.11::EF v3.1|climate change: fossil|global warming potential (GWP100)".to_string(),
        "ecoinvent-3.11::EF v3.1|climate change: land use and land use change|global warming potential (GWP100)".to_string(),
      ]);

    let lca_result = eval_system.evaluate().await?;
    let elapsed_time = start_time.elapsed();
    println!("Elapsed time: {:?}", elapsed_time);
    println!("result length: {}", lca_result.len());
    println!("Result: {:?}", lca_result);
    // If a GPU device was created internally, transfers aren't directly accessible here

    Ok(())
}

// helpers moved to shared.rs
