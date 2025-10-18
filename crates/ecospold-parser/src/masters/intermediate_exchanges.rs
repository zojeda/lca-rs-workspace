use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(rename = "validIntermediateExchanges", rename_all = "camelCase")]
pub struct ValidIntermediateExchanges {
    #[serde(rename = "@majorRelease")]
    pub major_release: u8,

    #[serde(rename = "@minorRelease")]
    pub minor_release: u8,

    #[serde(rename = "@majorRevision")]
    pub major_revision: u8,

    #[serde(rename = "@minorRevision")]
    pub minor_revision: u32,

    #[serde(rename = "@contextId")]
    pub context_id: String,

    pub intermediate_exchange: Vec<IntermediateExchange>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IntermediateExchange {
    #[serde(rename = "@id")]
    pub id: String,

    #[serde(rename = "@unitId")]
    pub unit_id: String,

    pub name: LocalizedText,
    pub unit_name: LocalizedText,

    #[serde(default)]
    pub classification: Vec<Classification>,

    #[serde(default)]
    pub comment: Option<LocalizedText>,

    #[serde(default)]
    pub property: Vec<Property>,

    #[serde(default)]
    pub product_information: Option<ProductInformation>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Classification {
    #[serde(rename = "@classificationId")]
    pub classification_id: String,

    pub classification_system: LocalizedText,
    pub classification_value: LocalizedText,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Property {
    #[serde(rename = "@propertyId")]
    pub property_id: String,

    #[serde(rename = "@amount")]
    pub amount: f64,

    #[serde(default)]
    pub comment: Option<LocalizedText>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProductInformation {
    pub text: Vec<IndexedLocalizedText>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexedLocalizedText {
    #[serde(rename = "$value")]
    pub value: Option<String>,

    #[serde(rename = "@xml:lang")]
    pub lang: Option<String>,

    #[serde(rename = "@index")]
    pub index: usize,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LocalizedText {
    #[serde(rename = "$value")]
    pub value: Option<String>,

    #[serde(rename = "@xml:lang")]
    pub lang: Option<String>,
}
