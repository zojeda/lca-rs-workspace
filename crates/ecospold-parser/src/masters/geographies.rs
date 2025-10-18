use serde::Deserialize;
#[derive(Debug, Deserialize)]
#[serde(rename = "validGeographies", rename_all = "camelCase")]
pub struct ValidGeographies {
    #[serde(rename = "@contextId")]
    pub context_id: String,

    pub context_name: LocalizedText,

    #[serde(default)]
    pub geography: Vec<Geography>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Geography {
    #[serde(rename = "@id")]
    pub id: String,

    #[serde(rename = "@longitude")]
    pub longitude: Option<f64>,

    #[serde(rename = "@latitude")]
    pub latitude: Option<f64>,

    #[serde(rename = "@uNCode")]
    pub un_code: Option<u32>,

    #[serde(rename = "@uNRegionCode")]
    pub un_region_code: Option<u32>,

    #[serde(rename = "@uNSubregionCode")]
    pub un_subregion_code: Option<u32>,

    pub name: String,

    #[serde(rename = "shortname")]
    pub short_name: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LocalizedText {
    #[serde(rename = "$value")]
    pub value: String,

    #[serde(rename = "@xml:lang")]
    pub lang: Option<String>,
}
