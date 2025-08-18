-- Создание основной таблицы транзакций с оптимальной структурой
CREATE TABLE IF NOT EXISTS transactions
(
    -- Основные поля
    transaction_id String,
    company_id String,
    card_masked String,
    amount Decimal64(2),
    currency FixedString(3) DEFAULT 'USD',

    -- MCC категоризация
    mcc_code FixedString(4),
    mcc_description String,

    -- Мерчант
    merchant_name String,
    merchant_id String,
    merchant_country FixedString(2),
    merchant_city String,

    -- Временные метки
    transaction_date Date,
    transaction_time DateTime,
    processing_time DateTime DEFAULT now(),

    -- Статусы и коды
    authorization_code String,
    response_code FixedString(2),
    status Enum8('pending' = 1, 'completed' = 2, 'failed' = 3, 'reversed' = 4),
    channel Enum8('pos' = 1, 'online' = 2, 'atm' = 3, 'mobile' = 4, 'other' = 5),

    -- Fraud detection
    is_fraud UInt8 DEFAULT 0,
    fraud_score Float32 DEFAULT 0,

    -- Дополнительные поля
    card_type Enum8('debit' = 1, 'credit' = 2, 'prepaid' = 3, 'other' = 4) DEFAULT 'other',
    issuer_country FixedString(2),
    pos_entry_mode String,

    -- Метаданные
    batch_id String,
    source_file String,
    loaded_at DateTime DEFAULT now()
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(transaction_date)
ORDER BY (company_id, transaction_date, transaction_time)
PRIMARY KEY (company_id, transaction_date)
SETTINGS index_granularity = 8192;

-- Таблица компаний
CREATE TABLE IF NOT EXISTS companies
(
    company_id String,
    company_name String,
    company_type String,
    country FixedString(2),
    is_active UInt8 DEFAULT 1,
    created_at DateTime DEFAULT now(),
    updated_at DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY company_id;

-- Таблица для агрегированных метрик (материализованное представление)
CREATE TABLE IF NOT EXISTS daily_metrics
(
    metric_date Date,
    company_id String,

    -- Основные метрики
    total_transactions UInt64,
    total_volume Decimal128(2),
    unique_cards UInt64,
    unique_merchants UInt64,

    -- Средние значения
    avg_transaction_amount Decimal64(2),
    median_transaction_amount Decimal64(2),

    -- Fraud метрики
    fraud_transactions UInt64,
    fraud_volume Decimal128(2),
    fraud_rate Float32,

    -- Распределение по каналам
    pos_count UInt64,
    online_count UInt64,
    atm_count UInt64,
    mobile_count UInt64,

    -- Топ категории
    top_mcc_code FixedString(4),
    top_merchant String,

    -- Временные метрики
    peak_hour UInt8,
    min_amount Decimal64(2),
    max_amount Decimal64(2)
)
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(metric_date)
ORDER BY (metric_date, company_id);

-- Материализованное представление для real-time метрик
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_realtime_metrics
ENGINE = SummingMergeTree()
ORDER BY (window_start, company_id)
AS
SELECT
    toStartOfMinute(transaction_time) AS window_start,
    company_id,
    count() AS transaction_count,
    sum(amount) AS total_volume,
    avg(amount) AS avg_amount,
    max(amount) AS max_amount,
    sumIf(1, is_fraud = 1) AS fraud_count
FROM transactions
GROUP BY window_start, company_id;

-- Таблица для хранения ML features
CREATE TABLE IF NOT EXISTS ml_features
(
    card_masked String,
    feature_date Date,

    -- Behavioral features
    daily_transaction_count UInt32,
    daily_total_amount Decimal64(2),
    avg_transaction_amount Decimal64(2),
    std_transaction_amount Float32,

    -- Temporal features
    night_transaction_ratio Float32,
    weekend_transaction_ratio Float32,

    -- Merchant diversity
    unique_merchants_count UInt32,
    unique_mcc_count UInt32,
    entropy_mcc Float32,

    -- Risk indicators
    high_risk_mcc_ratio Float32,
    foreign_transaction_ratio Float32,

    -- Velocity features
    transactions_last_hour UInt32,
    transactions_last_day UInt32,
    amount_last_hour Decimal64(2),
    amount_last_day Decimal64(2)
)
ENGINE = ReplacingMergeTree()
ORDER BY (card_masked, feature_date);

-- Создание индексов для оптимизации
ALTER TABLE transactions ADD INDEX idx_amount amount TYPE minmax GRANULARITY 4;
ALTER TABLE transactions ADD INDEX idx_fraud is_fraud TYPE set(2) GRANULARITY 4;
ALTER TABLE transactions ADD INDEX idx_merchant merchant_name TYPE bloom_filter(0.01) GRANULARITY 4;

-- Создание словаря для MCC кодов (для быстрого lookup)
CREATE DICTIONARY IF NOT EXISTS mcc_dictionary
(
    mcc_code String,
    description String,
    category String,
    risk_level UInt8
)
PRIMARY KEY mcc_code
SOURCE(FILE(path '/data/dictionaries/mcc_codes.csv' format 'CSV'))
LIFETIME(MIN 0 MAX 0)
LAYOUT(HASHED());