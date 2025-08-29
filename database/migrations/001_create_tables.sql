-- Создание основной таблицы transactions_main для транзакционных данных
CREATE TABLE IF NOT EXISTS card_analytics.transactions_main (
    hpan String,
    transaction_code String,
    rday UInt32,
    day_type String,
    issue_method String,
    card_product_type String,
    emitent_filial String,
    product_category String,
    emitent_region String,
    reissuing_flag UInt8,
    expire_date String,  -- Можно изменить на Date, если формат всегда DD.MM.YYYY
    issue_date String,   -- Аналогично
    card_type String,
    product_type String,
    card_bo_table String,
    pinfl String,
    pinfl_flag Float32,
    gender String,
    birth_year UInt16,
    age UInt8,
    age_group String,
    iss_flag UInt8,
    emitent_net String,
    emitent_bank String,
    acq_flag Boolean,
    acquirer_net String,
    acquirer_bank String,
    mcc UInt16,
    acquirer_mfo UInt32,
    merchant_name String,
    acquirer_branch String,
    acquirer_region String,
    merchant_type String,
    ip String,
    oked String,
    login_category String,
    login_group String,
    login String,
    amount_uzs UInt64,
    record_state String,
    reqamt UInt64,
    conamt UInt64,
    match_num UInt64,
    reversal_flag Boolean,
    fe_trace UInt64,
    refnum UInt64,
    sttl_date UInt32,
    sttl_hour UInt8,
    sttl_minute UInt8,
    hour_num UInt8,
    minute_num UInt8,
    udatetime_month UInt32,
    merchant String,
    respcode String,
    terminal_id String,
    address_name String,
    address_country String,
    currency String,
    merch_id String,
    terminal_type String,
    credit_debit String,
    inst_id String,
    inst_id2 String,
    term_id_key String,
    bo_table String,
    data_flag String,
    trans_type_by_day_key String,
    emission_country String,
    sender_hpan String,
    sender_bank String,
    receiver_hpan String,
    receiver_bank String,
    p2p_flag Boolean,
    p2p_type String
) ENGINE = MergeTree()
PARTITION BY rday  -- Партиционирование по дню для быстрых запросов
ORDER BY (rday, hpan);  -- Сортировка для оптимизации

-- Создание справочника merchants_dictionary (на основе merchant данных)
CREATE TABLE IF NOT EXISTS card_analytics.merchants_dictionary (
    merchant_name String,
    merchant_type String,
    mcc UInt16,
    acquirer_region String,
    oked String,
    address_name String,
    address_country String
) ENGINE = MergeTree()
ORDER BY merchant_name;

-- Создание таблицы для P2P переводов
CREATE TABLE IF NOT EXISTS card_analytics.p2p_transfers (
    sender_hpan String,
    sender_bank String,
    receiver_hpan String,
    receiver_bank String,
    p2p_flag Boolean,
    p2p_type String,
    amount_uzs UInt64,
    rday UInt32
) ENGINE = MergeTree()
PARTITION BY rday
ORDER BY (rday, sender_hpan);

-- Материализованное представление для ежедневных агрегаций
CREATE MATERIALIZED VIEW IF NOT EXISTS card_analytics.daily_aggregates
ENGINE = AggregatingMergeTree()
ORDER BY (rday, mcc)
POPULATE AS
SELECT
    rday,
    mcc,
    sum(amount_uzs) AS total_amount,
    count() AS transaction_count,
    avg(amount_uzs) AS avg_amount
FROM card_analytics.transactions_main
GROUP BY rday, mcc;