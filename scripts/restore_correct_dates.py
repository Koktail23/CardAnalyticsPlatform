#!/usr/bin/env python3
"""
Восстановление правильных дат в transactions_optimized
База должна быть 1900-01-01 для корректных дат 2025 года
"""

from clickhouse_driver import Client
from datetime import datetime, timedelta


def restore_correct_dates():
    """Восстанавливает правильные даты с базой 1900-01-01"""

    print("\n" + "=" * 60)
    print("ВОССТАНОВЛЕНИЕ ПРАВИЛЬНЫХ ДАТ")
    print("=" * 60)

    client = Client(
        host='localhost',
        port=9000,
        user='analyst',
        password='admin123',
        database='card_analytics'
    )

    # 1. Проверяем что есть исходные данные
    print("\n1. Проверка доступных таблиц...")

    tables = client.execute("""
        SELECT name, total_rows 
        FROM system.tables 
        WHERE database = 'card_analytics' 
        AND name LIKE 'transactions%'
    """)

    print("Найденные таблицы:")
    source_table = None
    for name, rows in tables:
        print(f"  • {name}: {rows:,} записей")
        if rows > 0 and 'simple' in name:
            source_table = name

    if not source_table:
        print("❌ Не найдена таблица с данными!")
        return

    print(f"\n✅ Будем использовать таблицу: {source_table}")

    # 2. Проверяем rday в исходной таблице
    print("\n2. Анализ rday в исходных данных...")

    rday_stats = client.execute(f"""
        SELECT 
            min(toUInt32OrNull(rday)) as min_rday,
            max(toUInt32OrNull(rday)) as max_rday
        FROM {source_table}
        WHERE rday != ''
    """)[0]

    print(f"  • Min rday: {rday_stats[0]}")
    print(f"  • Max rday: {rday_stats[1]}")

    # Проверяем с базой 1900-01-01
    base_date = datetime(1900, 1, 1)
    min_date = base_date + timedelta(days=rday_stats[0])
    max_date = base_date + timedelta(days=rday_stats[1])

    print(f"  • С базой 1900-01-01: {min_date.date()} - {max_date.date()}")

    if min_date.year < 2024 or max_date.year > 2026:
        print("⚠️  Даты выглядят некорректно даже с базой 1900!")
        return

    # 3. Пересоздаем таблицу с правильными датами
    print("\n3. Создание новой таблицы transactions_optimized...")

    client.execute("DROP TABLE IF EXISTS transactions_optimized_backup")
    client.execute("DROP TABLE IF EXISTS transactions_optimized_new")

    # Создаем резервную копию если есть старая таблица
    try:
        client.execute("RENAME TABLE transactions_optimized TO transactions_optimized_backup")
        print("  • Старая таблица сохранена как transactions_optimized_backup")
    except:
        pass

    # Создаем новую таблицу
    create_table = """
    CREATE TABLE transactions_optimized_new
    (
        -- Основные идентификаторы
        hpan String,
        transaction_code String,
        rday UInt32,
        transaction_date Date MATERIALIZED toDate('1900-01-01') + rday,  -- База 1900!

        -- Суммы
        amount_uzs Float64,
        reqamt Float64,
        conamt Float64,

        -- Категории
        mcc UInt16,
        merchant_name String,
        merchant_type String,
        merchant UInt32,

        -- P2P
        p2p_flag UInt8,
        p2p_type String,
        sender_hpan String,
        receiver_hpan String,
        sender_bank String,
        receiver_bank String,

        -- Клиент
        pinfl String,
        gender String,
        age String,
        age_group String,

        -- Банк и регион
        emitent_bank String,
        emitent_region String,
        acquirer_bank String,
        acquirer_region String,

        -- Карта
        card_type String,
        card_product_type String,

        -- Временные метки
        hour_num UInt8,
        minute_num UInt8,

        -- Дополнительные поля
        day_type String,
        terminal_type String,
        terminal_id String,
        credit_debit String,
        reversal_flag String,
        respcode String,
        ip String,
        login String
    )
    ENGINE = MergeTree()
    PARTITION BY toYYYYMM(transaction_date)
    ORDER BY (transaction_date, hpan, transaction_code)
    """

    client.execute(create_table)
    print("  ✅ Таблица создана с правильной базой 1900-01-01")

    # 4. Копируем данные из transactions_simple
    print("\n4. Копирование данных с преобразованием типов...")

    insert_query = f"""
    INSERT INTO transactions_optimized_new
    SELECT
        hpan,
        transaction_code,
        toUInt32OrDefault(rday, toUInt32(0)),
        toFloat64OrDefault(amount_uzs, toFloat64(0)),
        toFloat64OrDefault(reqamt, toFloat64(0)),
        toFloat64OrDefault(conamt, toFloat64(0)),
        toUInt16OrDefault(mcc, toUInt16(0)),
        merchant_name,
        merchant_type,
        toUInt32OrDefault(merchant, toUInt32(0)),
        -- P2P флаг: конвертируем True/False в 1/0
        CASE 
            WHEN p2p_flag = 'True' THEN toUInt8(1)
            WHEN p2p_flag = 'False' THEN toUInt8(0)
            ELSE toUInt8OrDefault(p2p_flag, toUInt8(0))
        END,
        p2p_type,
        sender_hpan,
        receiver_hpan,
        sender_bank,
        receiver_bank,
        pinfl,
        gender,
        age,
        age_group,
        emitent_bank,
        emitent_region,
        acquirer_bank,
        acquirer_region,
        card_type,
        card_product_type,
        toUInt8OrDefault(hour_num, toUInt8(0)),
        toUInt8OrDefault(minute_num, toUInt8(0)),
        day_type,
        terminal_type,
        terminal_id,
        credit_debit,
        reversal_flag,
        respcode,
        ip,
        login
    FROM {source_table}
    WHERE toUInt32OrNull(rday) IS NOT NULL
    """

    client.execute(insert_query)

    # 5. Проверяем результат
    print("\n5. Проверка результатов...")

    stats = client.execute("""
        SELECT 
            count() as total,
            min(transaction_date) as min_date,
            max(transaction_date) as max_date,
            avg(amount_uzs) as avg_amount
        FROM transactions_optimized_new
    """)[0]

    print(f"  • Записей загружено: {stats[0]:,}")
    print(f"  • Период: {stats[1]} - {stats[2]}")
    print(f"  • Средняя сумма: {stats[3]:,.0f} UZS")

    # 6. Переименовываем в финальную таблицу
    client.execute("DROP TABLE IF EXISTS transactions_optimized")
    client.execute("RENAME TABLE transactions_optimized_new TO transactions_optimized")

    print("\n" + "=" * 60)
    print("✅ УСПЕШНО ВОССТАНОВЛЕНО!")
    print("=" * 60)
    print(f"Таблица transactions_optimized готова к использованию")
    print(f"Период данных: {stats[1]} - {stats[2]}")
    print(f"База для rday: 1900-01-01")


if __name__ == "__main__":
    restore_correct_dates()