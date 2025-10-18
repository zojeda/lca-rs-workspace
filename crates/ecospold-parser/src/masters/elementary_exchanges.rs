use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(rename = "validElementaryExchanges")]
#[serde(rename_all = "camelCase")]
pub struct ValidElementaryExchanges {
    #[serde(rename = "elementaryExchange")]
    pub elementary_exchanges: Vec<ElementaryExchange>,

    #[serde(rename = "@majorRelease")]
    pub major_release: u8,

    #[serde(rename = "@minorRelease")]
    pub minor_release: u8,

    #[serde(rename = "@majorRevision")]
    pub major_revision: u16,

    #[serde(rename = "@minorRevision")]
    pub minor_revision: u32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ElementaryExchange {
    #[serde(rename = "@id")]
    pub id: String,

    #[serde(rename = "@unitId")]
    pub unit_id: String,

    #[serde(rename = "@formula")]
    pub formula: Option<String>,

    #[serde(rename = "@casNumber")]
    pub cas_number: Option<String>,

    pub name: LocalizedText,

    pub unit_name: LocalizedText,

    pub compartment: CompartmentInfo,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LocalizedText {
    #[serde(rename = "$value")]
    pub value: String,

    #[serde(rename = "@xml:lang")]
    pub lang: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompartmentInfo {
    #[serde(rename = "@subcompartmentId")]
    pub subcompartment_id: String,

    pub compartment: LocalizedText,
    pub subcompartment: LocalizedText,
}
