mod eval_lca_system;

pub mod error;
pub mod model;
pub use eval_lca_system::EvalLCASystem;
pub use error::LcaError;


#[cfg(feature = "wasm")]
use console_error_panic_hook;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

// --- WASM Setup ---
// Initialize logging and panic hook for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen(start)]
pub fn wasm_init() {
    // Only run this initialization once.
    static START: std::sync::Once = std::sync::Once::new();
    START.call_once(|| {
        // Use `wasm_logger` for Rust logs -> console.log
        wasm_logger::init(wasm_logger::Config::new(log::Level::Info));
        // Redirect panics to console.error
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        log::info!("LCA WASM module initialized.");
    });
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub fn human_size(size: u64) -> String {
    let mut size = size as f64;
    let mut unit = "B";
    let units = ["B", "KB", "MB", "GB", "TB"];
    for &u in &units[1..] {
        size /= 1024.0;
        if size < 1024.0 {
            unit = u;
            break;
        }
    }
    format!("{:.2} {}", size, unit)
}
