use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(rename = "validClassificationSystems", rename_all = "camelCase")]
pub struct ValidClassificationSystems {
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

    pub context_name: LocalizedText,

    pub classification_system: Vec<ClassificationSystem>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClassificationSystem {
    #[serde(rename = "@id")]
    pub id: String,

    #[serde(rename = "@type")]
    pub system_type: u8,

    pub name: LocalizedText,

    #[serde(default)]
    pub classification_value: Vec<ClassificationValue>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClassificationValue {
    #[serde(rename = "@id")]
    pub id: String,

    pub name: LocalizedText,

    #[serde(default)]
    pub comment: Option<LocalizedText>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LocalizedText {
    #[serde(rename = "$value")]
    pub value: String,

    #[serde(rename = "@xml:lang")]
    pub lang: Option<String>,
}
