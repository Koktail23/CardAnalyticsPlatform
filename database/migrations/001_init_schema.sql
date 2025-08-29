-- database/migrations/001_init_schema.sql
-- Инициализация схемы для Card Analytics Platform

-- Создаем базу данных
CREATE DATABASE IF NOT EXISTS card_analytics;

-- Используем созданную БД
USE card_analytics;

-- Основная таблица транзакций (74 колонки из датасета)
CREATE TABLE IF NOT EXISTS transactions_main
(
    -- Идентификаторы транзакции
    transaction_code String,
    rday UInt32,
    day_type String,

    -- Информация о карте
    hpan Float64,
    card_product_type String,
    card_type String,
    product_type String,
    product_category String,
    card_bo_table String,
    issue_method String,
    issue_date Date,
    expire_date Date,
    reissuing_flag String,

    -- Информация о клиенте
    pinfl String,
    pinfl_flag Nullable(Float32),
    gender String,
    birth_year String,
    age String,
    age_group String,

    -- Эмитент
    iss_flag UInt8,
    emitent_filial String,
    emitent_region String,
    emitent_net String,
    emitent_bank String,
    emission_country String,

    -- Эквайер
    acq_flag UInt8,
    acquirer_net String,
    acquirer_bank String,
    acquirer_mfo String,
    acquirer_branch String,
    acquirer_region String,

    -- Мерчант
    mcc UInt16,
    merchant_name String,
    merchant_type String,
    merchant UInt32,
    merch_id UInt32,
    oked Nullable(Float32),
    terminal_id String,
    terminal_type String,
    term_id_key String,
    address_name String,
    address_country String,

    -- IP и логин
    ip String,
    login_category String,
    login_group String,
    login String,

    -- Суммы
    amount_uzs UInt64,
    reqamt UInt64,
    conamt UInt64,
    currency UInt16,

    -- Статусы
    record_state String,
    match_num UInt32,
    reversal_flag UInt8,
    respcode Nullable(Float32),
    credit_debit String,
    data_flag String,
    trans_type_by_day_key String,

    -- Временные метки
    fe_trace UInt32,
    refnum UInt64,
    sttl_date UInt32,
    sttl_hour UInt8,
    sttl_minute UInt8,
    hour_num UInt8,
    minute_num UInt8,
    udatetime_month UInt8,

    -- Инстанс данные
    inst_id UInt32,
    inst_id2 UInt32,
    bo_table String,

    -- P2P данные
    p2p_flag UInt8,
    p2p_type String,
    sender_hpan String,
    sender_bank String,
    receiver_hpan String,
    receiver_bank String,

    -- Системные поля
    inserted_at DateTime DEFAULT now()
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(toDate(rday))
ORDER BY (rday, transaction_code, hpan)
SETTINGS index_granularity = 8192;