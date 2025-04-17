// ecospold-parser/src/main.rs
mod model; // Declare the model module

use model::EcoSpold; // Import the root struct
use std::env;
use std::fs;
use std::process;

fn main() {
    // --- Argument Parsing ---
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <path_to_ecospold_file.spold>", args[0]);
        process::exit(1);
    }
    let file_path = &args[1];

    // --- File Reading ---
    println!("Reading file: {}", file_path);
    let xml_content = match fs::read_to_string(file_path) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("Error reading file '{}': {}", file_path, e);
            process::exit(1);
        }
    };

    // --- XML Parsing ---
    println!("Parsing XML content...");
    // quick-xml needs BOM handling for files starting with one (like the example)
    // The UTF-8 BOM ('\u{FEFF}') is 3 bytes long (EF BB BF).
    let content_to_parse = if xml_content.starts_with('\u{FEFF}') {
        &xml_content[3..] // Skip the 3-byte BOM
    } else {
        &xml_content
    };

    let parsed_data: EcoSpold = match quick_xml::de::from_str(content_to_parse) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error parsing XML from '{}': {}", file_path, e);
            // Optionally print more details for debugging
            eprintln!("Error details: {:?}", e);
            process::exit(1);
        }
    };

    // --- Success & Output (Optional) ---
    println!("Successfully parsed EcoSpold file!");

    // Example: Print the activity name
    let activity_name_value = &parsed_data
        .child_activity_dataset
        .activity_description
        .activity
        .activity_name
        .value;
    println!("Activity Name: {}", activity_name_value);

    let activity_description = &parsed_data
        .child_activity_dataset
        .activity_description
        .activity;
    println!("Activity Description: {:#?}", activity_description);

    // Example: Print actual intermediate exchanges
    let intermediate_exchanges: Vec<_> = parsed_data
        .child_activity_dataset
        .flow_data
        .exchanges
        .iter()
        .filter(|ex| matches!(ex, model::Exchange::IntermediateExchange(_)))
        .collect();
    println!("Intermediate Exchanges:");
    for exchange in intermediate_exchanges {
        println!("\t{:#?}", exchange);
    }
    let elementary_exchanges: Vec<_> = parsed_data
        .child_activity_dataset
        .flow_data
        .exchanges
        .iter()
        .filter(|ex| matches!(ex, model::Exchange::ElementaryExchange(_)))
        .collect();
    println!("Elementary Exchanges:");
    // for exchange in elementary_exchanges {
    //     // println!("\t{:#?}", exchange);
    // }

    // For more detailed output, you could use:
    // println!("Parsed Data: {:#?}", parsed_data);
}
