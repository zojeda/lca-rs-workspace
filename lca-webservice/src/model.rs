use serde::{Deserialize, Serialize};
use serde_json::json; // Added for json! macro in schema example
use utoipa::ToSchema;
use validator::Validate;

// --- Request Models ---

// Mirroring lca_rs::model::LcaModel for the request body.
// We'll use this as LcaRequest to avoid direct dependency on lca_rs types in the API signature if possible,
// or we can directly use lca_rs::model::LcaModel if it has Serialize/Deserialize and ToSchema.
// For now, let's define it explicitly for clarity in the web service layer.

#[derive(Clone, Debug, Serialize, Deserialize, Validate, ToSchema)]
#[schema(example = json!({
  "database_name": "Bike Example DB",
  "case_name": "Bike Example",
  "case_description": "Bike Example Description",
  "imported_dbs": [],
  "processes": [
    {
      "name": "Bike production DK",
      "products": [
        {
          "product": {
            "name": "Bike"
          },
          "amount": {
            "Literal": 1.0
          },
          "unit": "bike"
        }
      ],
      "inputs": [
        {
          "product": {
            "Product": "Carbon fibre DE|Carbon fibre DE"
          },
          "amount": {
            "Literal": 2.5
          },
          "unit": "kg"
        }
      ],
      "emissions": [],
      "resources": []
    },
    {
      "name": "Carbon fibre DE",
      "products": [
        {
          "product": {
            "name": "Carbon fibre DE"
          },
          "amount": {
            "Literal": 1.0
          },
          "unit": "kg"
        }
      ],
      "inputs": [
        {
          "product": {
            "Product": "Natural gas NO|Natural gas NO"
          },
          "amount": {
            "Literal": 237.0
          },
          "unit": "MJ"
        }
      ],
      "emissions": [
        {
          "substance": {
            "Substance": "CO2"
          },
          "amount": {
            "Literal": 26.6
          },
          "unit": "kg",
          "compartment": null
        }
      ],
      "resources": []
    },
    {
      "name": "Natural gas NO",
      "products": [
        {
          "product": {
            "name": "Natural gas NO"
          },
          "amount": {
            "Literal": 1.0
          },
          "unit": "MJ"
        }
      ],
      "inputs": [],
      "emissions": [],
      "resources": []
    }
  ],
  "substances": [
    {
      "name": "CO2",
      "type": "Emission",
      "compartment": null,
      "sub_compartment": null,
      "reference_unit": "kg",
      "impacts": [
        {
          "name": "GWP100",
          "amount": {
            "Literal": 1.0
          },
          "unit": "kg CO2 eq"
        }
      ]
    }
  ],
  "evaluation_demands": [
    [
      {
        "product": "Bike production DK|Bike",
        "amount": 1.0
      }
    ]
  ],
  "evaluation_impacts": [
    {
      "Impact": "GWP100"
    }
  ]
}))]
pub struct LcaRequest {
    #[validate(length(min = 1))]
    pub database_name: String,
    #[validate(length(min = 1))]
    pub case_name: String,
    pub case_description: String,
    pub imported_dbs: Vec<DbImportRequest>,
    #[validate(length(min = 1))]
    pub processes: Vec<ProcessRequest>,
    pub substances: Vec<SubstanceRequest>, // Can be empty if all substances are external/implicit
    #[validate(length(min = 1))]
    pub evaluation_demands: Vec<Vec<DemandItemRequest>>,
    #[validate(length(min = 1))]
    pub evaluation_impacts: Vec<ImpactRefRequest>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Validate, ToSchema)]
pub struct DbImportRequest {
    #[validate(length(min = 1))]
    pub name: String,
    #[validate(length(min = 1))]
    pub alias: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, Validate, ToSchema)]
pub struct ProcessRequest {
    #[validate(length(min =1))]
    pub name: String,
    #[validate(length(min = 1))] // Ensures products is not empty. Exact count of 1 is validated by lca-rs.
    pub products: Vec<OutputItemRequest>,
    pub inputs: Vec<InputItemRequest>,
    pub emissions: Vec<BiosphereItemRequest>,
    pub resources: Vec<BiosphereItemRequest>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Validate, ToSchema)]
pub struct OutputItemRequest {
    pub product: ProductRequest,
    pub amount: AmountRequest,
    #[validate(length(min = 1))]
    pub unit: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, Validate, ToSchema)]
pub struct BiosphereItemRequest {
    pub substance: SubstanceRefRequest,
    pub amount: AmountRequest,
    #[validate(length(min = 1))]
    pub unit: String,
    pub compartment: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Validate, ToSchema)]
pub struct InputItemRequest {
    pub product: ProductRefRequest,
    pub amount: AmountRequest,
    #[validate(length(min = 1))]
    pub unit: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
pub enum ImpactRefRequest {
    Impact(String),
    External(ExternalRefRequest),
}

#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
pub enum ProductRefRequest {
    Product(String),
    External(ExternalRefRequest),
}

#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
pub enum SubstanceRefRequest {
    Substance(String),
    External(ExternalRefRequest),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Validate, ToSchema)]
pub struct ExternalRefRequest {
    #[validate(length(min = 1))]
    pub alias: String,
    #[validate(length(min = 1))]
    pub name: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
pub enum AmountRequest {
    Literal(f64),
    // Potentially extend with formulas later, matching lca_rs::model::Amount
}

#[derive(Clone, Debug, Serialize, Deserialize, Validate, ToSchema)]
pub struct ProductRequest {
    #[validate(length(min = 1))]
    pub name: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, Validate, ToSchema)]
pub struct SubstanceRequest {
    #[validate(length(min = 1))]
    pub name: String,
    pub r#type: SubstanceTypeRequest,
    pub compartment: Option<String>,
    pub sub_compartment: Option<String>,
    #[validate(length(min = 1))]
    pub reference_unit: String,
    pub impacts: Vec<ImpactRequest>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Validate, ToSchema)]
pub struct ImpactRequest {
    #[validate(length(min = 1))]
    pub name: String,
    pub amount: AmountRequest,
    #[validate(length(min = 1))]
    pub unit: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, ToSchema)]
pub enum SubstanceTypeRequest {
    Emission,
    Resource,
}

#[derive(Clone, Debug, Serialize, Deserialize, Validate, ToSchema)]
pub struct DemandItemRequest {
    #[validate(length(min = 1))]
    pub product: String, // e.g., "ProcessName|ProductName"
    pub amount: f64,
}

// --- Response Models ---

#[derive(Serialize, Deserialize, ToSchema, Debug)]
pub struct CalculationResponse {
    pub message: String,
    pub results: Option<Vec<f64>>,
    pub errors: Option<Vec<String>>,
}

#[derive(Serialize, Deserialize, ToSchema, Debug)]
pub enum ProgressUpdateType {
    Info,
    Error,
    Result,
    Status, // General status update
    Compilation,
    EvaluationStep,
}


#[derive(Serialize, Deserialize, ToSchema, Debug)]
pub struct ProgressUpdate {
    pub event_type: ProgressUpdateType,
    pub message: String,
    pub details: Option<serde_json::Value>, // For richer error details or structured progress
    pub result_data: Option<Vec<f64>>, // Only for final result event
}


#[derive(Serialize, Deserialize, ToSchema, Debug)]
pub struct ErrorResponse {
    pub status_code: u16,
    pub error: String,
    pub message: String,
    pub details: Option<Vec<String>>,
}

// Helper to convert LcaRequest to lca_rs::model::LcaModel
// This involves mapping each field and sub-struct.
// This is a bit verbose but ensures decoupling if web service models diverge slightly
// or if lca_rs models don't have all the derives we need for the web layer (like Validate).
impl From<LcaRequest> for lca_rs::model::LcaModel { // LcaModel is in lca_rs::model
    fn from(req: LcaRequest) -> Self {
        lca_rs::model::LcaModel {
            database_name: req.database_name,
            case_name: req.case_name,
            case_description: req.case_description,
            imported_dbs: req.imported_dbs.into_iter().map(|db| db.into()).collect(),
            processes: req.processes.into_iter().map(|p| p.into()).collect(),
            substances: req.substances.into_iter().map(|s| s.into()).collect(),
            evaluation_demands: req
                .evaluation_demands
                .into_iter()
                .map(|demands| demands.into_iter().map(|d| d.into()).collect())
                .collect(),
            evaluation_impacts: req
                .evaluation_impacts
                .into_iter()
                .map(|impact_ref_req| impact_ref_req.into()).collect(),                
        }
    }
}

impl From<DbImportRequest> for lca_rs::model::DbImport {
    fn from(req: DbImportRequest) -> Self {
        lca_rs::model::DbImport { name: req.name, alias: req.alias }
    }
}

impl From<ProcessRequest> for lca_rs::model::Process {
    fn from(req: ProcessRequest) -> Self {
        lca_rs::model::Process {
            name: req.name,
            products: req.products.into_iter().map(|p| p.into()).collect(),
            inputs: req.inputs.into_iter().map(|i| i.into()).collect(),
            emissions: req.emissions.into_iter().map(|e| e.into()).collect(),
            resources: req.resources.into_iter().map(|r| r.into()).collect(),
        }
    }
}

impl From<OutputItemRequest> for lca_rs::model::OutputItem {
    fn from(req: OutputItemRequest) -> Self {
        lca_rs::model::OutputItem {
            product: req.product.into(),
            amount: req.amount.into(),
            unit: req.unit,
        }
    }
}

impl From<BiosphereItemRequest> for lca_rs::model::BiosphereItem {
    fn from(req: BiosphereItemRequest) -> Self {
        lca_rs::model::BiosphereItem {
            substance: req.substance.into(),
            amount: req.amount.into(),
            unit: req.unit,
            compartment: req.compartment,
        }
    }
}

impl From<InputItemRequest> for lca_rs::model::InputItem {
    fn from(req: InputItemRequest) -> Self {
        lca_rs::model::InputItem {
            product: req.product.into(),
            amount: req.amount.into(),
            unit: req.unit,
        }
    }
}

impl From<ProductRefRequest> for lca_rs::model::ProductRef {
    fn from(req: ProductRefRequest) -> Self {
        match req {
            ProductRefRequest::Product(s) => lca_rs::model::ProductRef::Product(s),
            ProductRefRequest::External(e) => lca_rs::model::ProductRef::External(e.into()),
        }
    }
}

impl From<ImpactRefRequest> for lca_rs::model::ImpactRef {
  fn from(req: ImpactRefRequest) -> Self {
      match req {
        ImpactRefRequest::Impact(s) => lca_rs::model::ImpactRef::Impact(s),
        ImpactRefRequest::External(e) => lca_rs::model::ImpactRef::External(e.into()),
      }
  }
}

impl From<SubstanceRefRequest> for lca_rs::model::SubstanceRef {
    fn from(req: SubstanceRefRequest) -> Self {
        match req {
            SubstanceRefRequest::Substance(s) => lca_rs::model::SubstanceRef::Substance(s),
            SubstanceRefRequest::External(e) => lca_rs::model::SubstanceRef::External(e.into()),
        }
    }
}

impl From<ExternalRefRequest> for lca_rs::model::ExternalRef {
    fn from(req: ExternalRefRequest) -> Self {
        lca_rs::model::ExternalRef { alias: req.alias, name: req.name }
    }
}

impl From<AmountRequest> for lca_rs::model::Amount {
    fn from(req: AmountRequest) -> Self {
        match req {
            AmountRequest::Literal(val) => lca_rs::model::Amount::Literal(val),
        }
    }
}

impl From<ProductRequest> for lca_rs::model::Product {
    fn from(req: ProductRequest) -> Self {
        lca_rs::model::Product { name: req.name }
    }
}

impl From<SubstanceRequest> for lca_rs::model::Substance {
    fn from(req: SubstanceRequest) -> Self {
        lca_rs::model::Substance {
            name: req.name,
            r#type: req.r#type.into(),
            compartment: req.compartment,
            sub_compartment: req.sub_compartment,
            reference_unit: req.reference_unit,
            impacts: req.impacts.into_iter().map(|i| i.into()).collect(),
        }
    }
}

impl From<ImpactRequest> for lca_rs::model::Impact {
    fn from(req: ImpactRequest) -> Self {
        lca_rs::model::Impact {
            name: req.name,
            amount: req.amount.into(),
            unit: req.unit,
        }
    }
}

impl From<SubstanceTypeRequest> for lca_rs::model::SubstanceType {
    fn from(req: SubstanceTypeRequest) -> Self {
        match req {
            SubstanceTypeRequest::Emission => lca_rs::model::SubstanceType::Emission,
            SubstanceTypeRequest::Resource => lca_rs::model::SubstanceType::Resource,
        }
    }
}

impl From<DemandItemRequest> for lca_rs::DemandItem { // DemandItem is pub use lca_rs::DemandItem;
    fn from(req: DemandItemRequest) -> Self {
        lca_rs::DemandItem::new(req.product, req.amount)
    }
}
