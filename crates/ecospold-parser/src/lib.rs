pub mod error;
pub mod masters;
pub mod model;

use error::Result;

pub fn parse_ecospold(xml_content: &str) -> Result<model::EcoSpold> {
    let content_to_parse = if xml_content.starts_with('\u{FEFF}') {
        &xml_content[3..] // Skip the 3-byte BOM
    } else {
        xml_content
    };

    let parsed_data = quick_xml::de::from_str(content_to_parse)?;
    Ok(parsed_data)
}

pub fn parse_master<'de, T>(xml_content: &'de str) -> Result<T>
where
    T: serde::de::Deserialize<'de>,
{
    let content_to_parse = if xml_content.starts_with('\u{FEFF}') {
        &xml_content[3..] // Skip the 3-byte BOM
    } else {
        xml_content
    };

    let parsed_data = quick_xml::de::from_str(content_to_parse)?;
    Ok(parsed_data)
}
