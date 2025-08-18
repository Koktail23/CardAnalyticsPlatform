-- Create database if not exists
CREATE DATABASE IF NOT EXISTS card_analytics;

USE card_analytics;

-- 1. Main transactions table with optimized types
CREATE TABLE IF NOT EXISTS transactions_main
(
    -- Transaction identifiers
    transaction_code String,
    rday Date,
    sttl_date Date,
    hour UInt8,
    minute UInt8,

    -- Card information
    hpan String,
    card_product_type String,
    card_type String,
    expire_date Date,
    issue_date Date,
    reissuing_flag UInt8,

    -- Client information
    pinfl String,
    gender Enum8('M' = 1, 'F' = 2, 'U' = 0),
    birth_year UInt16,
    age UInt8,
    age_group String,
    emitent_region String,

    -- Transaction amounts
    amount_uzs Decimal(20, 2),
    reqamt Decimal(20, 2),
    conamt Decimal(20, 2),

    -- Transaction details
    credit_debit Enum8('credit' = 1, 'debit' = 2),
    reversal_flag UInt8,
    respcode String,
    refnum String,
    fe_refnum String,
    fe_stan String,
    fe_trace String,

    -- Merchant information
    mcc UInt16,
    merchant_name String,
    merchant_type String,
    oked String,
    terminal_type String,
    terminal_id String,

    -- Bank information
    emitent_bank String,
    acquirer_bank String,

    -- P2P transfer flags
    p2p_flag UInt8,
    p2p_type String,
    sender_hpan String,
    sender_bank String,
    receiver_hpan String,
    receiver_bank String,

    -- Technical fields
    processing_date DateTime DEFAULT now(),
    data_quality_score Float32 DEFAULT 1.0
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(rday)
ORDER BY (rday, hpan, transaction_code)
SETTINGS index_granularity = 8192;

-- 2. Merchants dictionary table
CREATE TABLE IF NOT EXISTS merchants_dictionary
(
    merchant_id String,
    merchant_name String,
    merchant_type String,
    mcc UInt16,
    oked String,
    risk_score Float32 DEFAULT 0.0,
    avg_transaction_amount Decimal(20, 2),
    total_transactions UInt64,
    last_updated DateTime DEFAULT now(),

    PRIMARY KEY (merchant_id)
)
ENGINE = ReplacingMergeTree(last_updated)
ORDER BY merchant_id;

-- 3. Cards dictionary table
CREATE TABLE IF NOT EXISTS cards_dictionary
(
    hpan String,
    card_product_type String,
    card_type String,
    issue_date Date,
    expire_date Date,
    reissuing_count UInt8 DEFAULT 0,
    pinfl String,
    emitent_bank String,
    emitent_region String,
    is_active UInt8 DEFAULT 1,
    fraud_score Float32 DEFAULT 0.0,
    last_transaction_date Date,
    total_transactions UInt64 DEFAULT 0,
    total_amount Decimal(20, 2) DEFAULT 0,
    last_updated DateTime DEFAULT now(),

    PRIMARY KEY (hpan)
)
ENGINE = ReplacingMergeTree(last_updated)
ORDER BY hpan;

-- 4. P2P transfers dedicated table
CREATE TABLE IF NOT EXISTS p2p_transfers
(
    transfer_id String,
    transaction_code String,
    transfer_date Date,
    transfer_time DateTime,
    sender_hpan String,
    sender_bank String,
    sender_pinfl String,
    receiver_hpan String,
    receiver_bank String,
    receiver_pinfl String,
    amount Decimal(20, 2),
    p2p_type String,
    respcode String,
    reversal_flag UInt8,
    network_cluster_id UInt32 DEFAULT 0,

    PRIMARY KEY (transfer_id)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(transfer_date)
ORDER BY (transfer_date, sender_hpan, receiver_hpan);

-- 5. Daily aggregates materialized view
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_aggregates_mv
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, mcc, card_type, emitent_region)
AS SELECT
    toDate(rday) as date,
    mcc,
    card_type,
    emitent_region,
    credit_debit,
    count() as transaction_count,
    sum(amount_uzs) as total_amount,
    avg(amount_uzs) as avg_amount,
    min(amount_uzs) as min_amount,
    max(amount_uzs) as max_amount,
    countDistinct(hpan) as unique_cards,
    countDistinct(merchant_name) as unique_merchants,
    sum(reversal_flag) as reversal_count,
    sum(p2p_flag) as p2p_count
FROM transactions_main
GROUP BY date, mcc, card_type, emitent_region, credit_debit;

-- 6. Hourly traffic patterns
CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_traffic_mv
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, hour)
AS SELECT
    toDate(rday) as date,
    hour,
    count() as transaction_count,
    sum(amount_uzs) as total_amount,
    avg(amount_uzs) as avg_amount,
    countDistinct(hpan) as unique_cards,
    sum(p2p_flag) as p2p_count,
    sum(reversal_flag) as reversal_count
FROM transactions_main
GROUP BY date, hour;

-- 7. Customer behavior aggregates
CREATE TABLE IF NOT EXISTS customer_behavior_features
(
    pinfl String,
    calc_date Date,

    -- Transaction counts
    txn_count_7d UInt32,
    txn_count_30d UInt32,
    txn_count_90d UInt32,

    -- Amount statistics
    avg_amount_7d Decimal(20, 2),
    avg_amount_30d Decimal(20, 2),
    avg_amount_90d Decimal(20, 2),
    total_amount_7d Decimal(20, 2),
    total_amount_30d Decimal(20, 2),
    total_amount_90d Decimal(20, 2),

    -- MCC preferences
    top_mcc_1 UInt16,
    top_mcc_2 UInt16,
    top_mcc_3 UInt16,
    mcc_diversity_score Float32,

    -- P2P behavior
    p2p_sent_count_30d UInt32,
    p2p_received_count_30d UInt32,
    p2p_unique_contacts_30d UInt32,

    -- Risk indicators
    night_txn_ratio Float32,
    weekend_txn_ratio Float32,
    foreign_txn_ratio Float32,
    high_risk_mcc_ratio Float32,
    velocity_score Float32,

    -- Activity patterns
    days_since_last_txn UInt16,
    active_days_30d UInt8,
    preferred_hour UInt8,
    preferred_day_of_week UInt8,

    last_updated DateTime DEFAULT now(),

    PRIMARY KEY (pinfl, calc_date)
)
ENGINE = ReplacingMergeTree(last_updated)
PARTITION BY toYYYYMM(calc_date)
ORDER BY (pinfl, calc_date);

-- 8. Fraud detection results
CREATE TABLE IF NOT EXISTS fraud_detection_results
(
    transaction_code String,
    detection_timestamp DateTime,
    fraud_probability Float32,
    fraud_label UInt8,
    model_version String,
    detection_reason String,
    feature_importance String,  -- JSON string with SHAP values

    PRIMARY KEY (transaction_code)
)
ENGINE = ReplacingMergeTree(detection_timestamp)
ORDER BY transaction_code;

-- 9. System monitoring table
CREATE TABLE IF NOT EXISTS system_monitoring
(
    timestamp DateTime,
    metric_name String,
    metric_value Float64,
    metric_type String,

    PRIMARY KEY (timestamp, metric_name)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, metric_name)
TTL timestamp + INTERVAL 90 DAY;

-- Create indexes for better query performance
ALTER TABLE transactions_main ADD INDEX idx_pinfl pinfl TYPE bloom_filter GRANULARITY 1;
ALTER TABLE transactions_main ADD INDEX idx_mcc mcc TYPE minmax GRANULARITY 1;
ALTER TABLE transactions_main ADD INDEX idx_merchant merchant_name TYPE bloom_filter GRANULARITY 1;
ALTER TABLE transactions_main ADD INDEX idx_amount amount_uzs TYPE minmax GRANULARITY 1;

-- Create projection for fast card analytics
ALTER TABLE transactions_main ADD PROJECTION card_analytics_projection
(
    SELECT
        hpan,
        rday,
        sum(amount_uzs) as daily_amount,
        count() as daily_count
    GROUP BY hpan, rday
);

-- Create projection for merchant analytics
ALTER TABLE transactions_main ADD PROJECTION merchant_analytics_projection
(
    SELECT
        merchant_name,
        mcc,
        rday,
        sum(amount_uzs) as daily_amount,
        count() as daily_count,
        countDistinct(hpan) as unique_cards
    GROUP BY merchant_name, mcc, rday
);

-- Grant permissions
GRANT ALL ON card_analytics.* TO analyst;