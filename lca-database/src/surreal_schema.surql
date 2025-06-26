// Table for Database
DEFINE TABLE database SCHEMAFULL;
DEFINE FIELD id ON database TYPE string;
DEFINE FIELD name ON database TYPE string;
DEFINE FIELD provider ON database TYPE string;
DEFINE FIELD provider_url ON database TYPE string;
DEFINE FIELD created ON database TYPE datetime;
DEFINE FIELD updated ON database TYPE datetime;
DEFINE FIELD version ON database TYPE { major: int, minor: int, patch: int};
DEFINE FIELD depends_on ON database TYPE array<record<database>>;

// Table for Geography
DEFINE TABLE geography SCHEMAFULL;
DEFINE FIELD id ON geography TYPE string;
DEFINE FIELD code ON geography TYPE string;
DEFINE FIELD name ON geography TYPE string;

// Table for ElementaryExchange
DEFINE TABLE elementary_exchange SCHEMAFULL;
DEFINE FIELD id ON elementary_exchange TYPE string;
DEFINE FIELD database ON elementary_exchange TYPE record<database>;
DEFINE FIELD name ON elementary_exchange TYPE string;
DEFINE FIELD uom ON elementary_exchange TYPE string;
DEFINE FIELD comparment ON elementary_exchange TYPE string;
DEFINE FIELD subcompartment ON elementary_exchange TYPE string;


// Table for ClassificationSystem
DEFINE TABLE classification_system SCHEMAFULL;
DEFINE FIELD id ON classification_system TYPE string;
DEFINE FIELD database ON classification_system TYPE record<database>;
DEFINE FIELD name ON classification_system TYPE string;

// Table for Classification
DEFINE TABLE classification SCHEMAFULL;
DEFINE FIELD id ON classification TYPE string;
DEFINE FIELD classification_system ON classification TYPE record<classification_system>;
DEFINE FIELD name ON classification_system TYPE string;

// Table for IntermediateExchange
DEFINE TABLE intermediate_exchange SCHEMAFULL;
DEFINE FIELD id ON intermediate_exchange TYPE string;
DEFINE FIELD database ON intermediate_exchange TYPE record<database>;
DEFINE FIELD name ON intermediate_exchange TYPE string;
DEFINE FIELD uom ON intermediate_exchange TYPE string;
DEFINE FIELD classifications ON intermediate_exchange TYPE array<record<classification>>;


// Table for Activity
DEFINE TABLE activity SCHEMAFULL;
DEFINE FIELD classifications ON activity TYPE array<record<classification>>;
DEFINE FIELD id ON activity TYPE string;
DEFINE FIELD name ON activity TYPE string;
DEFINE FIELD geography ON activity TYPE {
    id: record<geography>,
    short_name: string,
    comment: string
};
DEFINE FIELD intermediate_exchanges ON TABLE activity VALUE <future> { 
    RETURN SELECT VALUE ->technosphere_flow.{amount, uom, name: out.name, intermediate_exchange: out} FROM ONLY $this 
};

DEFINE FIELD database ON activity TYPE record<database>;

-- DEFINE FIELD geography ON activity TYPE record<geography>;

DEFINE TABLE IF NOT EXISTS technosphere_flow
TYPE RELATION FROM activity TO intermediate_exchange;
DEFINE FIELD amount ON technosphere_flow TYPE number;
DEFINE FIELD uom ON technosphere_flow TYPE string;

DEFINE TABLE IF NOT EXISTS biosphere_flow
TYPE RELATION FROM activity TO elementary_exchange;
DEFINE FIELD id ON biosphere_flow TYPE string;
DEFINE FIELD amount ON biosphere_flow TYPE number;
DEFINE FIELD uom ON biosphere_flow TYPE string;
