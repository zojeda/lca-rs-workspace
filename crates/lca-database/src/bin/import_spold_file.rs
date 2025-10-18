use chrono::Utc;
use geo::Point;
use indicatif::ProgressBar;
use itertools::Itertools;
use lca_database::{
    Error, LcaDatabase, RefRecord, Result, SurrealLcaDb,
    model::{self, Database, IOType, SemVer},
};

// Use structs and Error from lca-database
use std::{path::PathBuf, str::FromStr as _};
use surrealdb::{RecordId, sql::Datetime}; // Import Uuid from surrealdb
// use uuid::Uuid; // Import Uuid for unique identifiers
// Import necessary types from ecospold-parser
use ecospold_parser::masters::{
    classifications::ValidClassificationSystems, elementary_exchanges::ValidElementaryExchanges,
    geographies::ValidGeographies, intermediate_exchanges::ValidIntermediateExchanges,
};

#[tokio::main]
async fn main() -> Result<()> {
    let db: SurrealLcaDb = SurrealLcaDb::new("127.0.0.1:8080").await?;
    import_ecoinvent_database(&db, "../ecoinvent/ecoinvent 3.11_cutoff_ecoSpold02").await?;
    Ok(())
}

async fn import_ecoinvent_database(
    db: &SurrealLcaDb,
    path: impl Into<PathBuf> + Clone,
) -> Result<()> {
    let database = db.create_database(model::Database {
    id: RecordId::from_str("database:⟨de659012-50c4-4e96-b54a-fc781bf987ab⟩")?,
    name: "ecoinvent 3.11_cutoff_ecoSpold02".to_string(),
    provider: "ecoinvent".to_string(),
    provider_url: Some("https://ecoinvent.org/".to_string()),
    version: SemVer::new(3, 11, 0),
    created: Datetime::from(Utc::now()),
    updated: Datetime::from(Utc::now()),
    comment: Some("This is the default context of the ecoinvent database. It has no dependencies to other contexts.".to_string()),
    depends_on: vec![],
  }).await?;

    // import_classifications(db, path.clone(), database.clone()).await?;
    import_geographies(db, path.clone(), database.clone()).await?;
    // import_elementary_exchanges(db, path.clone()).await?;
    // import_intermediate_exchanges(db,path.clone(), database.clone()).await?;
    // import_activities(db, path.clone(), database.clone()).await?;

    Ok(())
}

async fn import_classifications(
    db: &SurrealLcaDb,
    base_path: impl Into<PathBuf> + Clone,
    database: RefRecord<Database>,
) -> Result<()> {
    let path = base_path.into().join("MasterData/Classifications.xml");

    let xml_content = std::fs::read_to_string(path).expect("Failed to read file");

    let parsed_data = ecospold_parser::parse_master::<ValidClassificationSystems>(&xml_content)
        .expect("Failed to parse XML");
    let classification_systems = parsed_data.classification_system;
    println!(
        "Importing {} classification systems",
        classification_systems.len()
    );
    let total_classification_items = classification_systems
        .iter()
        .map(|cs| cs.classification_value.len())
        .sum::<usize>();
    println!("Total classification items: {}", total_classification_items);
    let bar = ProgressBar::new(total_classification_items as u64);
    for parsed_classification_system in classification_systems {
        let parsed_classification_system_id =
            RecordId::from_table_key("classification_system", parsed_classification_system.id);
        let classification_system = model::ClassificationSystem {
            id: parsed_classification_system_id,
            name: parsed_classification_system.name.value,
            database: database.id.clone(),
        };
        let classification_system_record = db
            .create_classification_system(classification_system)
            .await?;
        for parsed_classification_value in parsed_classification_system.classification_value {
            let parsed_classification_system_id =
                RecordId::from_table_key("classification", parsed_classification_value.id);
            let classification_value = model::Classification {
                id: parsed_classification_system_id,
                name: parsed_classification_value.name.value,
                classification_system: classification_system_record.id.clone(),
                comments: parsed_classification_value.comment.map(|c| c.value),
            };
            db.create_classification(classification_value).await?;
            bar.inc(1);
        }
    }
    bar.finish();
    Ok(())
}

async fn import_geographies(
    db: &SurrealLcaDb,
    base_path: impl Into<PathBuf> + Clone,
    database: RefRecord<Database>,
) -> Result<()> {
    let path = base_path.into().join("MasterData/Geographies.xml");

    let xml_content = std::fs::read_to_string(path).expect("Failed to read file");

    let parsed_data = ecospold_parser::parse_master::<ValidGeographies>(&xml_content)
        .expect("Failed to parse XML");
    let geographies = parsed_data.geography;
    println!("Importing {} geographies", geographies.len());
    let total_geographies = geographies.len();
    let bar = ProgressBar::new(total_geographies as u64);
    for parsed_geography in geographies {
        let location =
            if let (Some(x), Some(y)) = (parsed_geography.longitude, parsed_geography.latitude) {
                let point = surrealdb::sql::Geometry::Point(Point::new(x, y));
                Some(point)
            } else {
                None
            };

        let parsed_geography_id = RecordId::from_table_key("geography", parsed_geography.id);
        let geography = model::Geography {
            id: parsed_geography_id,
            name: parsed_geography.name,
            short_name: parsed_geography.short_name,
            un_code: parsed_geography.un_code,
            un_region_code: parsed_geography.un_region_code,
            location,
            un_subregion_code: parsed_geography.un_subregion_code,
        };
        db.create_geography(geography).await?;
        bar.inc(1);
    }
    bar.finish();
    Ok(())
}

async fn import_elementary_exchanges(
    db: &SurrealLcaDb,
    base_path: impl Into<PathBuf> + Clone,
) -> Result<()> {
    let path = base_path.into().join("MasterData/ElementaryExchanges.xml");

    let xml_content = std::fs::read_to_string(path).expect("Failed to read file");
    let parsed_data = ecospold_parser::parse_master::<ValidElementaryExchanges>(&xml_content)
        .expect("Failed to parse XML");
    let elementary_exchanges = parsed_data.elementary_exchanges;
    println!(
        "Importing {} elementary exchanges",
        elementary_exchanges.len()
    );
    let bar = ProgressBar::new(elementary_exchanges.len() as u64);
    for exchange in elementary_exchanges {
        bar.inc(1);
        let elementary_exchange = model::ElementaryExchange {
            id: RecordId::from_table_key("elementary_exchange", exchange.id),
            name: exchange.name.value,
            uom: exchange.unit_name.value,
            comparment: exchange.compartment.compartment.value,
            subcompartment: exchange.compartment.subcompartment.value,
        };
        db.create_elementary_exchange(elementary_exchange)
            .await
            .unwrap();
    }
    bar.finish();

    Ok(())
}

async fn import_intermediate_exchanges(
    db: &SurrealLcaDb,
    base_path: impl Into<PathBuf> + Clone,
    database: RefRecord<Database>,
) -> Result<()> {
    let path = base_path
        .into()
        .join("MasterData/IntermediateExchanges.xml");
    let xml_content = std::fs::read_to_string(path).expect("Failed to read file");

    let parsed_data = ecospold_parser::parse_master::<ValidIntermediateExchanges>(&xml_content)
        .expect("Failed to parse XML");
    let parsed_exchanges = parsed_data.intermediate_exchange;
    println!(
        "Importing {} intermediate exchanges",
        parsed_exchanges.len()
    );
    let bar = ProgressBar::new(parsed_exchanges.len() as u64);
    for parsed_exchange in parsed_exchanges {
        bar.inc(1);
        let classifications = parsed_exchange
            .classification
            .iter()
            .map(|pc| RecordId::from_table_key("classification", pc.classification_id.clone()))
            .collect();
        let exchange = model::IntermediateExchange {
            id: RecordId::from_table_key("intermediate_exchange", parsed_exchange.id),
            name: parsed_exchange.name.value.unwrap_or_default(),
            uom: parsed_exchange.unit_name.value.unwrap_or_default(),
            classifications,
            database: database.clone().into(),
        };
        db.create_intermediate_exchange(exchange).await.unwrap();
    }
    bar.finish();

    Ok(())
}

async fn import_activities(
    db: &SurrealLcaDb,
    base_path: impl Into<PathBuf> + Clone,
    database: RefRecord<Database>,
) -> Result<()> {
    let path = base_path.into().join("datasets");
    let datasets = std::fs::read_dir(path)
        .expect("Failed to read directory")
        .filter_map(|entry| {
            entry.ok().and_then(|dir_entry| {
                dir_entry.path().extension().and_then(|e| {
                    if e == "spold" {
                        Some(dir_entry.path())
                    } else {
                        None
                    }
                })
            })
        })
        .collect::<Vec<_>>();
    let activities_count = datasets.len();
    println!("Importing {} activities", activities_count);
    let bar = ProgressBar::new(activities_count as u64);
    for dataset in datasets {
        bar.inc(1);
        let xml_content = std::fs::read_to_string(&dataset).expect("Failed to read file");

        match ecospold_parser::parse_ecospold(&xml_content) {
            Ok(parsed_data) => {
                let parsed_activity = parsed_data
                    .child_activity_dataset
                    .or(parsed_data.activity_dataset)
                    .ok_or(Error::NoActivityDatasetFound(dataset))?;

                let classifications = parsed_activity
                    .activity_description
                    .classifications
                    .iter()
                    .map(|pc| {
                        RecordId::from_table_key("classification", pc.classification_id.clone())
                    })
                    .collect::<Vec<_>>();

                let activity_id = RecordId::from_table_key(
                    "activity",
                    parsed_activity.activity_description.activity.id,
                );
                let bio_flows = parsed_activity
                    .flow_data
                    .exchanges
                    .iter()
                    .filter_map(|f| match f {
                        ecospold_parser::model::Exchange::ElementaryExchange(ee) => {
                            Some(model::BiosphereFlow {
                                id: RecordId::from_table_key("biosphere_flow", ee.id.clone()),
                                activity: activity_id.clone(),
                                elementary_exchange: RecordId::from_table_key(
                                    "elementary_exchange",
                                    ee.elementary_exchange_id.clone(),
                                ),
                                amount: ee.amount,
                                uom: ee.unit_id.clone(),
                            })
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let outputs = parsed_activity
                    .flow_data
                    .exchanges
                    .iter()
                    .filter_map(|f| match f {
                        ecospold_parser::model::Exchange::IntermediateExchange(ie) => {
                            if let Ok(IOType::Output(o_type)) =
                                get_io_type(ie.input_group, ie.output_group)
                            {
                                Some(model::ActivityOutput {
                                    id: RecordId::from_table_key("activity_output", ie.id.clone()),
                                    activity: activity_id.clone(),
                                    intermediate_exchange: RecordId::from_table_key(
                                        "intermediate_exchange",
                                        ie.intermediate_exchange_id.clone(),
                                    ),
                                    production_volume_amount: ie.production_volume_amount,
                                    production_volume_comment: ie
                                        .production_volume_comment
                                        .as_ref()
                                        .map(|c| c.value.clone()),
                                    production_volume_mathematical_relation: ie
                                        .production_volume_comment
                                        .as_ref()
                                        .map(|c| c.value.clone()),
                                    amount: ie.amount,
                                    output_type: o_type,
                                    uom: ie.unit_name.value.clone(),
                                    cas_number: ie.cas_number.clone(),
                                    comments: Some(
                                        ie.comments.iter().map(|c| c.value.clone()).join("\n"),
                                    ),
                                })
                            } else {
                                None
                            }
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let tech_flows = parsed_activity
                    .flow_data
                    .exchanges
                    .iter()
                    .filter_map(|f| match f {
                        ecospold_parser::model::Exchange::IntermediateExchange(ie) => {
                            if let Ok(IOType::Input(i_grp)) =
                                get_io_type(ie.input_group, ie.output_group)
                            {
                                Some(model::TechnosphereFlow {
                                    id: RecordId::from_table_key(
                                        "technosphere_flow",
                                        ie.id.clone(),
                                    ),
                                    input: RecordId::from_table_key(
                                        "activity",
                                        ie.activity_link_id.as_ref().unwrap().clone(),
                                    ),
                                    output: activity_id.clone(),
                                    intermediate_exchange: RecordId::from_table_key(
                                        "intermediate_exchange",
                                        ie.intermediate_exchange_id.clone(),
                                    ),

                                    amount: ie.amount,
                                    input_type: i_grp,
                                    uom: ie.unit_name.value.clone(),
                                    cas_number: ie.cas_number.clone(),
                                })
                            } else {
                                None
                            }
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                // let bio_flow_ids = bio_flows.iter().map(|bf| bf.id.clone()).collect::<Vec<_>>();

                let activity = model::Activity {
                    id: activity_id.clone(),
                    name: parsed_activity
                        .activity_description
                        .activity
                        .activity_name
                        .value,
                    classifications,
                    geography: RecordId::from_table_key(
                        "geography",
                        parsed_activity.activity_description.geography.geography_id,
                    ),
                    database: database.id.clone(),
                };
                // let json = serde_json::to_string_pretty(&tech_flows.iter().map(|tf| tf.clone()).collect::<Vec<_>>()).unwrap();
                // println!("{}", json);
                db.create_activity(activity, outputs, bio_flows, tech_flows)
                    .await
                    .unwrap();
            }
            Err(e) => println!("failed to parse ecospold file {}: {}", dataset.display(), e),
        };
    }
    bar.finish();

    Ok(())
}

fn get_io_type(input_group: Option<i32>, output_group: Option<i32>) -> Result<IOType> {
    if let Some(igrp) = input_group {
        match igrp {
            1 => Ok(IOType::Input(model::InputType::MaterialsFuels)),
            2 => Ok(IOType::Input(model::InputType::ElectricityHeat)),
            3 => Ok(IOType::Input(model::InputType::Services)),
            4 => Ok(IOType::Input(model::InputType::FromEnvironment)),
            5 => Ok(IOType::Input(model::InputType::FromTechnosphere)),
            _ => Err(Error::DatabaseImportError("inputGroup invalid".to_string())),
        }
    } else if let Some(ogrp) = output_group {
        match ogrp {
            0 => Ok(IOType::Output(model::OutputType::ReferenceProduct)),
            2 => Ok(IOType::Output(model::OutputType::ByProduct)),
            3 => Ok(IOType::Output(model::OutputType::MaterialForTreatment)),
            4 => Ok(IOType::Output(model::OutputType::ToEnvironment)),
            5 => Ok(IOType::Output(model::OutputType::StockAddition)),
            _ => Err(Error::DatabaseImportError(
                "outputGroup invalid".to_string(),
            )),
        }
    } else {
        Err(Error::DatabaseImportError(
            "inputGroup or outputGroup required".to_string(),
        ))
    }
}
