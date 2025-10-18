mod error;
pub mod model;
mod surreal_lca_db;

use std::marker::PhantomData;

pub use error::*;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
pub use surreal_lca_db::SurrealLcaDb;
use surrealdb::RecordId;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RefRecord<T: Serialize + DeserializeOwned> {
    pub id: RecordId,
    _type: PhantomData<T>,
}

pub trait LcaDatabase {
    fn create_database(
        &self,
        database: model::Database,
    ) -> impl std::future::Future<Output = Result<RefRecord<model::Database>>> + Send;
    fn create_geography(
        &self,
        geography: model::Geography,
    ) -> impl std::future::Future<Output = Result<RefRecord<model::Geography>>> + Send;
    fn create_intermediate_exchange(
        &self,
        exchange: model::IntermediateExchange,
    ) -> impl std::future::Future<Output = Result<RefRecord<model::IntermediateExchange>>> + Send;
    fn create_elementary_exchange(
        &self,
        exchange: model::ElementaryExchange,
    ) -> impl std::future::Future<Output = Result<RefRecord<model::ElementaryExchange>>> + Send;
    fn create_activity(
        &self,
        activity: model::Activity,
        outputs: Vec<model::ActivityOutput>,
        biosphere_flows: Vec<model::BiosphereFlow>,
        technosphere_flows: Vec<model::TechnosphereFlow>,
    ) -> impl std::future::Future<Output = Result<RefRecord<model::Activity>>> + Send;
    fn create_classification_system(
        &self,
        classification_system: model::ClassificationSystem,
    ) -> impl std::future::Future<Output = Result<RefRecord<model::ClassificationSystem>>> + Send;
    fn create_classification(
        &self,
        classification: model::Classification,
    ) -> impl std::future::Future<Output = Result<RefRecord<model::Classification>>> + Send;
    // fn get_activity(&self, id: Uuid) -> Result<Activity>;
    // fn update_activity(
    //   &self,
    //   activity: Activity) -> Result<(), String>;
    // fn delete_activity(&self, id: Uuid) -> Result<(), String>;
    // fn search_activities(&self) -> Result<Vec<Activity>, String>;
    // fn search_intermediate_exchanges(&self) -> Result<Vec<IntermediateExchange>, String>;
}
