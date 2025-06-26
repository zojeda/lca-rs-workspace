use std::marker::PhantomData;

use crate::model::{ActivityOutput, BiosphereFlow, Classification, ClassificationSystem, TechnosphereFlow};
use crate::{error::*, RecordId, RefRecord};
use crate::{model::Activity, LcaDatabase};

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use surrealdb::opt::auth::Root;
use surrealdb::opt::PatchOp;
use surrealdb::{engine::remote::ws::{Ws, Client as WsClient}, Surreal, sql::{Thing, Id}};

type SurrealClient = Surreal<WsClient>;
pub struct SurrealLcaDb {
  db: SurrealClient
}

impl SurrealLcaDb {
  pub async fn new(connection_string: &str) -> Result<Self> {
    let db = Surreal::new::<Ws>(connection_string).await?;
    
    db.signin(Root {
      username: "root",
      password: "root",
    })
    .await?;

    db.use_ns("lci")
      .use_db("lci_db")
      .await?;
    Ok(SurrealLcaDb { 
      db
    })
  }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SavedRecord {
  id: surrealdb::RecordId
}

impl SavedRecord {
  fn to_ref_record<T: Serialize + DeserializeOwned>(&self) -> RefRecord<T> {
    RefRecord {
      id: self.id.clone(),
      _type: PhantomData
    }
  }
}

impl From<RefRecord<crate::model::Database>> for RecordId {
  fn from(val: RefRecord<crate::model::Database>) -> Self {
    val.id
  }
}
impl From<RefRecord<crate::model::ClassificationSystem>> for RecordId {
  fn from(val: RefRecord<crate::model::ClassificationSystem>) -> Self {
    val.id
  }
}

impl LcaDatabase for SurrealLcaDb {
  async fn create_database(&self, database: crate::model::Database) -> Result<RefRecord<crate::model::Database>> {
    let id = database.id.clone();
    let record: Option<SavedRecord> = self.db.upsert(id.clone())
      .content(database).await?;
    record.map(|r| r.to_ref_record()).ok_or_else(|| Error::DatabaseImportError(format!("Failed to create database with id: {}", id)))
  }
  async fn create_geography(&self, database: crate::model::Geography) -> Result<RefRecord<crate::model::Geography>> {
    let id = database.id.clone();
    let record: Option<SavedRecord> = self.db.upsert(id.clone())
      .content(database).await?;
    record.map(|r| r.to_ref_record()).ok_or_else(|| Error::DatabaseImportError(format!("Failed to create geoagraphy with id: {}", id)))
  }
  async fn create_elementary_exchange(&self, elementary_exchange: crate::model::ElementaryExchange) -> Result<RefRecord<crate::model::ElementaryExchange>> {
    let id = elementary_exchange.id.clone();
    let record: Option<SavedRecord> = self.db.upsert(id.clone())
      .content(elementary_exchange).await?;
    record.map(|r| r.to_ref_record()).ok_or_else(|| Error::DatabaseImportError(format!("Failed to create elementary_exchange with id: {}", id)))
  }
  async fn create_intermediate_exchange(&self, intermediate_exchange: crate::model::IntermediateExchange) -> Result<RefRecord<crate::model::IntermediateExchange>> {
    let id = intermediate_exchange.id.clone();
    let record: Option<SavedRecord> = self.db.upsert(id.clone())
      .content(intermediate_exchange).await?;
    record.map(|r| r.to_ref_record()).ok_or_else(|| Error::DatabaseImportError(format!("Failed to create intermediate_exchange with id: {}", id)))
  }  
  async fn create_activity(&self, activity: Activity, outputs: Vec<ActivityOutput>, biosphere_flows: Vec<BiosphereFlow>, technosphere_flows: Vec<TechnosphereFlow>) -> Result<RefRecord<crate::model::Activity>> {
    let id = activity.id.clone();
    let record: Option<SavedRecord> = self.db.upsert(id.clone()).content(activity).await?;
    let result: Vec<ActivityOutput> = self.db.insert("activity_output").relation(outputs).await?;
    let result: Vec<BiosphereFlow> = self.db.insert("biosphere_flow").relation(biosphere_flows).await?;
    let result: Vec<TechnosphereFlow> = self.db.insert("technosphere_flow").relation(technosphere_flows).await?;
    record.map(|r| r.to_ref_record()).ok_or_else(|| Error::DatabaseImportError(format!("Failed to create activity with id: {}", id)))
  }

  async fn create_classification_system(&self, classification_system: ClassificationSystem) -> Result<RefRecord<ClassificationSystem>> {
    let id = classification_system.id.clone();
    let record: Option<SavedRecord> = self.db
      .upsert(id.clone())
      .content(classification_system).await?;
    record.map(|r| r.to_ref_record()).ok_or_else(|| Error::DatabaseImportError(format!("Failed to create classification_system with id: {}", id)))
  }

  async fn create_classification(&self, classification: Classification) -> Result<RefRecord<crate::model::Classification>> {
    let id = classification.id.clone();
    let record: Option<SavedRecord> = self.db
      .upsert(id.clone())
      .content(classification).await?;
    record.map(|r| r.to_ref_record()).ok_or_else(|| Error::DatabaseImportError(format!("Failed to create classification with id: {}", id)))
  }  

}