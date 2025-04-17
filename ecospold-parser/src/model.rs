// ecospold-parser/src/model.rs
use serde::Deserialize;
// serde_json::Value import removed

// Using quick_xml::de attributes for XML mappin:g
// Renaming attributes and elements to match Rust conventions (snake_case)
// Using Option<T> for optional attributes/elements

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct EcoSpold {
    pub child_activity_dataset: ChildActivityDataset,
    // Add xmlns attribute if needed, though often handled by parser context
    // #[serde(rename = "@xmlns")]
    // pub xmlns: String,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ChildActivityDataset {
    pub activity_description: ActivityDescription,
    pub flow_data: FlowData,
    pub modelling_and_validation: ModellingAndValidation,
    pub administrative_information: AdministrativeInformation,
}

// --- ActivityDescription ---

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ActivityDescription {
    pub activity: Activity,
    #[serde(
        default,
        rename = "classification",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub classifications: Vec<Classification>,
    pub geography: Geography,
    pub technology: Technology,
    pub time_period: TimePeriod,
    pub macro_economic_scenario: MacroEconomicScenario,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Activity {
    #[serde(rename = "@id")]
    pub id: String,
    #[serde(rename = "@activityNameId")]
    pub activity_name_id: String,
    #[serde(
        rename = "@parentActivityId",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub parent_activity_id: Option<String>,
    #[serde(rename = "@inheritanceDepth")]
    pub inheritance_depth: i32,
    #[serde(rename = "@type")]
    pub type_attr: i32, // 'type' is a reserved keyword in Rust
    #[serde(rename = "@specialActivityType")]
    pub special_activity_type: i32,
    #[serde(rename = "@energyValues")]
    pub energy_values: i32,
    pub activity_name: TextLang,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub synonym: Vec<TextLang>,
    pub included_activities_start: TextLang,
    pub included_activities_end: TextLang,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub general_comment: Option<GeneralComment>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tag: Vec<String>,
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct TextLang {
    #[serde(rename = "@xml:lang", default)] // Make lang optional by adding default
    pub lang: Option<String>,
    #[serde(rename = "$text", default)] // Add default for potentially empty text nodes
    pub value: String,
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct TextLangIndex {
    #[serde(rename = "@xml:lang", default)] // Make lang optional by adding default
    pub lang: Option<String>,
    #[serde(rename = "@index")]
    pub index: i32,
    #[serde(rename = "$text", default)] // Add default for potentially empty text nodes
    pub value: String,
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct GeneralComment {
    #[serde(rename = "text", default, skip_serializing_if = "Vec::is_empty")]
    pub texts: Vec<TextLangIndex>,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Classification {
    #[serde(rename = "@classificationId")]
    pub classification_id: String,
    pub classification_system: TextLang,
    pub classification_value: TextLang,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Geography {
    #[serde(rename = "@geographyId")]
    pub geography_id: String,
    pub shortname: TextLang,
    #[serde(default, rename = "comment", skip_serializing_if = "Vec::is_empty")]
    pub comments: Vec<Comment>, // Reverted back to Vec<Comment>
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Technology {
    #[serde(rename = "@technologyLevel")]
    pub technology_level: i32,
    #[serde(default, rename = "comment", skip_serializing_if = "Vec::is_empty")]
    pub comments: Vec<Comment>, // Reverted back to Vec<Comment>
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TimePeriod {
    #[serde(rename = "@startDate")]
    pub start_date: String, // Consider using a date type like chrono::NaiveDate
    #[serde(rename = "@endDate")]
    pub end_date: String, // Consider using a date type like chrono::NaiveDate
    #[serde(rename = "@isDataValidForEntirePeriod")]
    pub is_data_valid_for_entire_period: bool,
    #[serde(default, rename = "comment", skip_serializing_if = "Vec::is_empty")]
    pub comments: Vec<Comment>, // Reverted back to Vec<Comment>
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MacroEconomicScenario {
    #[serde(rename = "@macroEconomicScenarioId")]
    pub macro_economic_scenario_id: String,
    pub name: TextLang,
}

// Re-introduced Comment struct
#[derive(Debug, Deserialize, PartialEq)]
pub struct Comment {
    #[serde(rename = "text", default, skip_serializing_if = "Vec::is_empty")]
    pub texts: Vec<TextLangIndex>,
    #[serde(rename = "imageUrl", default, skip_serializing_if = "Vec::is_empty")]
    pub image_urls: Vec<ImageUrl>,
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct ImageUrl {
    #[serde(rename = "@index")]
    pub index: i32,
    #[serde(rename = "$text")]
    pub value: String,
}

// --- FlowData ---

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct FlowData {
    // Use an enum to represent different exchange types
    // Need to handle multiple types directly under flowData
    #[serde(rename = "$value", default, skip_serializing_if = "Vec::is_empty")]
    pub exchanges: Vec<Exchange>,
}

// Using an enum to handle different types of exchanges within flowData
#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum Exchange {
    IntermediateExchange(IntermediateExchange),
    ElementaryExchange(ElementaryExchange),
    Parameter(Parameter), // Added Parameter based on the XML
                          // Add other potential exchange types if they exist in the schema
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct IntermediateExchange {
    #[serde(rename = "@id")]
    pub id: String,
    #[serde(rename = "@unitId")]
    pub unit_id: String,
    #[serde(rename = "@amount")]
    pub amount: f32, // Use f32 for numerical amounts
    #[serde(rename = "@intermediateExchangeId")]
    pub intermediate_exchange_id: String,
    #[serde(
        rename = "@activityLinkId",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub activity_link_id: Option<String>,
    #[serde(
        rename = "@casNumber",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub cas_number: Option<String>,
    #[serde(
        rename = "@productionVolumeAmount",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub production_volume_amount: Option<f32>,
    #[serde(rename = "@sourceId", default, skip_serializing_if = "Option::is_none")]
    pub source_id: Option<String>,
    #[serde(
        rename = "@sourceYear",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub source_year: Option<String>, // Could be i32 if always a year
    #[serde(
        rename = "@sourceFirstAuthor",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub source_first_author: Option<String>,

    pub name: TextLang,
    pub unit_name: TextLang,
    #[serde(default, rename = "comment", skip_serializing_if = "Vec::is_empty")]
    pub comments: Vec<TextLang>, // Reverted to Vec<TextLang>
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uncertainty: Option<Uncertainty>,
    #[serde(default, rename = "property", skip_serializing_if = "Vec::is_empty")]
    pub properties: Vec<Property>,
    #[serde(
        default,
        rename = "classification",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub classifications: Vec<Classification>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_group: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_group: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub production_volume_comment: Option<TextLang>,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ElementaryExchange {
    #[serde(rename = "@id")]
    pub id: String,
    #[serde(rename = "@unitId")]
    pub unit_id: String,
    #[serde(rename = "@amount")]
    pub amount: f32,
    #[serde(rename = "@elementaryExchangeId")]
    pub elementary_exchange_id: String,
    #[serde(
        rename = "@casNumber",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub cas_number: Option<String>,
    #[serde(rename = "@formula", default, skip_serializing_if = "Option::is_none")]
    pub formula: Option<String>,
    #[serde(
        rename = "@variableName",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub variable_name: Option<String>,
    #[serde(rename = "@isCalculatedAmount", default)]
    pub is_calculated_amount: bool,
    #[serde(
        rename = "@mathematicalRelation",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub mathematical_relation: Option<String>,
    #[serde(rename = "@sourceId", default, skip_serializing_if = "Option::is_none")]
    pub source_id: Option<String>,
    #[serde(
        rename = "@sourceYear",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub source_year: Option<String>, // Could be i32 if always a year
    #[serde(
        rename = "@sourceFirstAuthor",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub source_first_author: Option<String>,

    pub name: TextLang,
    pub unit_name: TextLang,
    #[serde(default, rename = "comment", skip_serializing_if = "Vec::is_empty")]
    pub comments: Vec<TextLang>, // Reverted to Vec<TextLang>
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uncertainty: Option<Uncertainty>,
    #[serde(default, rename = "synonym", skip_serializing_if = "Vec::is_empty")]
    pub synonyms: Vec<TextLang>,
    #[serde(default, rename = "property", skip_serializing_if = "Vec::is_empty")]
    pub properties: Vec<Property>,
    pub compartment: Compartment,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_group: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_group: Option<i32>,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Parameter {
    #[serde(rename = "@parameterId")]
    pub parameter_id: String,
    #[serde(rename = "@variableName")]
    pub variable_name: String,
    #[serde(rename = "@amount")]
    pub amount: f32,
    #[serde(
        rename = "@mathematicalRelation",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub mathematical_relation: Option<String>,
    #[serde(rename = "@isCalculatedAmount", default)]
    pub is_calculated_amount: bool,

    pub name: TextLang,
    pub unit_name: TextLang,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uncertainty: Option<Uncertainty>,
    #[serde(default, rename = "comment", skip_serializing_if = "Vec::is_empty")]
    pub comments: Vec<TextLang>, // Reverted to Vec<TextLang>
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Uncertainty {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lognormal: Option<Lognormal>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub undefined: Option<UndefinedUncertainty>, // Added based on XML
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pedigree_matrix: Option<PedigreeMatrix>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub comment: Option<TextLang>,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Lognormal {
    #[serde(rename = "@meanValue")]
    pub mean_value: f32,
    #[serde(rename = "@mu")]
    pub mu: f32,
    #[serde(rename = "@variance")]
    pub variance: f32,
    #[serde(rename = "@varianceWithPedigreeUncertainty")]
    pub variance_with_pedigree_uncertainty: f32,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UndefinedUncertainty {
    #[serde(rename = "@minValue")]
    pub min_value: f32,
    #[serde(rename = "@maxValue")]
    pub max_value: f32,
    #[serde(rename = "@standardDeviation95")]
    pub standard_deviation95: f32,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PedigreeMatrix {
    #[serde(rename = "@reliability")]
    pub reliability: i32,
    #[serde(rename = "@completeness")]
    pub completeness: i32,
    #[serde(rename = "@temporalCorrelation")]
    pub temporal_correlation: i32,
    #[serde(rename = "@geographicalCorrelation")]
    pub geographical_correlation: i32,
    #[serde(rename = "@furtherTechnologyCorrelation")]
    pub further_technology_correlation: i32,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Property {
    #[serde(rename = "@propertyId")]
    pub property_id: String,
    #[serde(rename = "@amount")]
    pub amount: f32, // Sometimes this is a string (e.g., mathematicalRelation), handle appropriately if needed
    #[serde(rename = "@unitId")]
    pub unit_id: String,
    #[serde(
        rename = "@mathematicalRelation",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub mathematical_relation: Option<String>,
    #[serde(rename = "@isDefiningValue", default)] // Default to false if not present
    pub is_defining_value: bool,
    #[serde(
        rename = "@variableName",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub variable_name: Option<String>,

    pub name: TextLang,
    pub unit_name: TextLang,
    #[serde(default, rename = "comment", skip_serializing_if = "Vec::is_empty")]
    pub comments: Vec<TextLang>, // Reverted to Vec<TextLang>
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Compartment {
    #[serde(rename = "@subcompartmentId")]
    pub subcompartment_id: String,
    pub compartment: TextLang,
    pub subcompartment: TextLang,
}

// --- ModellingAndValidation ---

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ModellingAndValidation {
    pub representativeness: Representativeness,
    #[serde(default, rename = "review", skip_serializing_if = "Vec::is_empty")]
    pub reviews: Vec<Review>,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Representativeness {
    #[serde(rename = "@percent", default, skip_serializing_if = "Option::is_none")]
    pub percent: Option<f32>,
    #[serde(rename = "@systemModelId")]
    pub system_model_id: String,
    pub system_model_name: TextLang,
    pub sampling_procedure: TextLang,
    pub extrapolations: TextLang,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Review {
    #[serde(rename = "@reviewerId")]
    pub reviewer_id: String,
    #[serde(rename = "@reviewerName")]
    pub reviewer_name: String,
    #[serde(rename = "@reviewerEmail")]
    pub reviewer_email: String,
    #[serde(rename = "@reviewDate")]
    pub review_date: String, // Consider chrono::NaiveDate
    #[serde(rename = "@reviewedMajorRelease")]
    pub reviewed_major_release: i32,
    #[serde(rename = "@reviewedMinorRelease")]
    pub reviewed_minor_release: i32,
    #[serde(rename = "@reviewedMajorRevision")]
    pub reviewed_major_revision: i32,
    #[serde(rename = "@reviewedMinorRevision")]
    pub reviewed_minor_revision: i32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<ReviewDetails>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub other_details: Option<TextLang>,
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct ReviewDetails {
    #[serde(rename = "text", default, skip_serializing_if = "Vec::is_empty")]
    pub texts: Vec<TextLangIndex>,
}

// --- AdministrativeInformation ---

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct AdministrativeInformation {
    pub data_entry_by: PersonInfo,
    pub data_generator_and_publication: DataGeneratorAndPublication,
    pub file_attributes: FileAttributes,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PersonInfo {
    #[serde(rename = "@personId")]
    pub person_id: String,
    #[serde(rename = "@isActiveAuthor", default)]
    pub is_active_author: bool,
    #[serde(rename = "@personName")]
    pub person_name: String,
    #[serde(rename = "@personEmail")]
    pub person_email: String,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct DataGeneratorAndPublication {
    #[serde(rename = "@personId")]
    pub person_id: String,
    #[serde(rename = "@personName")]
    pub person_name: String,
    #[serde(rename = "@personEmail")]
    pub person_email: String,
    #[serde(rename = "@dataPublishedIn")]
    pub data_published_in: i32,
    #[serde(rename = "@publishedSourceId")]
    pub published_source_id: String,
    #[serde(rename = "@publishedSourceYear")]
    pub published_source_year: i32,
    #[serde(rename = "@publishedSourceFirstAuthor")]
    pub published_source_first_author: String,
    #[serde(rename = "@isCopyrightProtected")]
    pub is_copyright_protected: bool,
    #[serde(
        rename = "@pageNumbers",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub page_numbers: Option<String>,
    #[serde(rename = "@accessRestrictedTo")]
    pub access_restricted_to: i32,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct FileAttributes {
    #[serde(rename = "@majorRelease")]
    pub major_release: i32,
    #[serde(rename = "@minorRelease")]
    pub minor_release: i32,
    #[serde(rename = "@majorRevision")]
    pub major_revision: i32,
    #[serde(rename = "@minorRevision")]
    pub minor_revision: i32,
    #[serde(rename = "@internalSchemaVersion")]
    pub internal_schema_version: String,
    #[serde(rename = "@defaultLanguage")]
    pub default_language: String,
    #[serde(rename = "@creationTimestamp")]
    pub creation_timestamp: String, // Consider chrono::DateTime<Utc>
    #[serde(rename = "@lastEditTimestamp")]
    pub last_edit_timestamp: String, // Consider chrono::DateTime<Utc>
    #[serde(rename = "@fileGenerator")]
    pub file_generator: String,
    #[serde(rename = "@fileTimestamp")]
    pub file_timestamp: String, // Consider chrono::DateTime<Utc>
    #[serde(rename = "@contextId")]
    pub context_id: String,
    pub context_name: TextLang,
}
