use derive_more::Display;

use surrealdb::{sql::Datetime, RecordId, sql::Geometry};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Activity {
    pub id: RecordId,
    pub name: String,
    pub database: RecordId,
    pub classifications: Vec<RecordId>,
    pub geography: RecordId

}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ActivityOutput {
    pub id: RecordId,
    #[serde(rename = "in")]
    pub activity: RecordId,
    #[serde(rename = "out")]
    pub intermediate_exchange: RecordId,

    pub production_volume_amount: Option<f64>,
    pub production_volume_mathematical_relation: Option<String>,
    pub production_volume_comment: Option<String>,
    pub amount: f64,
    pub uom: String,
    pub cas_number: Option<String>,
    pub comments: Option<String>,
    pub output_type: OutputType,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TechnosphereFlow {
    pub id: RecordId,
    #[serde(rename = "in")]
    pub input: RecordId,
    #[serde(rename = "out")]
    pub output: RecordId,
    pub intermediate_exchange: RecordId,
    pub amount: f64,
    pub uom: String,
    pub cas_number: Option<String>,
    pub input_type: InputType,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BiosphereFlow {
    pub id: RecordId,
    #[serde(rename = "in")]
    pub activity: RecordId,
    #[serde(rename = "out")]
    pub elementary_exchange: RecordId,
    pub amount: f64,
    pub uom: String,
}


#[derive(Debug, Serialize, Deserialize)]
pub struct IntermediateExchange {
    pub id: RecordId,
    pub database: RecordId,
    pub name: String,
    pub uom: String,
    pub classifications: Vec<RecordId>
}

#[derive(Display, Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum IOType {
  Input(InputType),
  Output(OutputType)
}

#[derive(Display, Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum InputType {
  MaterialsFuels,
  ElectricityHeat,
  Services,
  FromEnvironment,
  FromTechnosphere,
}

#[derive(Display, Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum OutputType {
  ReferenceProduct,
  ByProduct,
  MaterialForTreatment,
  ToEnvironment,
  StockAddition,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ElementaryExchange {
    pub id: RecordId,
    pub name: String,
    pub uom: String,
    pub comparment: String,
    pub subcompartment: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Database {
    pub id: RecordId,
    pub name: String,
    pub provider: String,
    pub provider_url: Option<String>,
    pub comment: Option<String>,
    pub version: SemVer,
    pub created: Datetime,
    pub updated: Datetime,
    pub depends_on: Vec<RecordId>,

}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SemVer {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}
impl SemVer {
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        SemVer { major, minor, patch }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Geography {
    pub id: RecordId,
    pub short_name: String,
    pub name: String,
    pub location: Option<Geometry>,
    pub un_code: Option<u32>,
    pub un_region_code: Option<u32>,
    pub un_subregion_code: Option<u32>
}


#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ClassificationSystem {
    pub id: RecordId,
    pub name: String,
    pub database: RecordId,
}


#[derive(Debug, Serialize, Deserialize)]
pub struct Classification {
    pub id: RecordId,
    pub classification_system: RecordId,
    pub name: String,
    pub comments: Option<String>
}
