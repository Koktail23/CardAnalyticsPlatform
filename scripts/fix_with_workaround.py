#!/usr/bin/env python3
"""
Обходное решение для проблемы с датами в ClickHouse
Используем вычитание вместо сложения
"""

from clickhouse_driver import Client


def fix_with_workaround():
    client = Client(
        host='localhost',
        port=9000,
        user='analyst',
        password='admin123',
        database='card_analytics'
    )

    print("\n" + "=" * 60)
    print("ОБХОДНОЕ РЕШЕНИЕ ДЛЯ ДАТ")
    print("=" * 60)

    # Правильный расчет: нужно вычесть из rday определенное количество дней
    # rday 45658 должен дать 2025-01-03
    # Если от 2095-01-03 отнять 70 лет (25550 дней), получим 2025-01-03

    print("\n1. Анализ смещения...")

    # Проверяем смещение
    result = client.execute("""
        SELECT 
            45658 as rday,
            toDate('1970-01-01') + 45658 as wrong_date,
            toDate('1970-01-01') + 45658 - toIntervalYear(70) as corrected_date,
            toDate('1970-01-01') + (45658 - 25550) as alternative
    """)[0]

    print(f"   rday: {result[0]}")
    print(f"   Неправильная дата: {result[1]}")
    print(f"   Исправленная (минус 70 лет): {result[2]}")
    print(f"   Альтернатива (rday - 25550): {result[3]}")

    # Создаем исправленную таблицу
    print("\n2. Создание исправленной таблицы...")

    client.execute("DROP TABLE IF EXISTS transactions_fixed")

    client.execute("""
    CREATE TABLE transactions_fixed
    (
        hpan String,
        transaction_code String,
        rday UInt32,
        transaction_date Date,
        amount_uzs Float64,
        reqamt Float64,
        conamt Float64,
        mcc UInt16,
        merchant_name String,
        merchant_type String,
        merchant UInt32,
        p2p_flag UInt8,
        p2p_type String,
        sender_hpan String,
        receiver_hpan String,
        sender_bank String,
        receiver_bank String,
        pinfl String,
        gender String,
        age String,
        age_group String,
        emitent_bank String,
        emitent_region String,
        acquirer_bank String,
        acquirer_region String,
        card_type String,
        card_product_type String,
        hour_num UInt8,
        minute_num UInt8,
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
    """)

    print("   ✅ Таблица создана")

    # Копируем данные с корректным расчетом даты
    print("\n3. Копирование данных с исправлением дат...")

    # ОБХОДНОЕ РЕШЕНИЕ: вычитаем 70 лет из неправильной даты
    client.execute("""
    INSERT INTO transactions_fixed
    SELECT
        hpan,
        transaction_code,
        rday,
        -- Вычитаем 70 лет (25550 дней) чтобы получить правильную дату
        toDate('1970-01-01') + (rday - 25550) as transaction_date,
        amount_uzs,
        reqamt,
        conamt,
        mcc,
        merchant_name,
        merchant_type,
        merchant,
        p2p_flag,
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
        hour_num,
        minute_num,
        day_type,
        terminal_type,
        terminal_id,
        credit_debit,
        reversal_flag,
        respcode,
        ip,
        login
    FROM transactions_optimized
    """)

    # Проверяем результат
    print("\n4. Проверка результатов...")

    stats = client.execute("""
        SELECT 
            count() as total,
            min(transaction_date) as min_date,
            max(transaction_date) as max_date,
            min(rday) as min_rday,
            max(rday) as max_rday
        FROM transactions_fixed
    """)[0]

    print(f"   Записей: {stats[0]:,}")
    print(f"   Даты: {stats[1]} - {stats[2]}")
    print(f"   rday: {stats[3]} - {stats[4]}")

    if stats[2].year == 2025:
        print("\n✅ Даты исправлены успешно!")

        # Заменяем старую таблицу
        print("\n5. Замена таблицы...")
        client.execute("DROP TABLE IF EXISTS transactions_optimized_old")
        client.execute("RENAME TABLE transactions_optimized TO transactions_optimized_old")
        client.execute("RENAME TABLE transactions_fixed TO transactions_optimized")

        print("   ✅ Таблица transactions_optimized обновлена")

        print("\n" + "=" * 60)
        print("✅ ГОТОВО!")
        print("=" * 60)
        print("Теперь можно запускать feature_engineering.py")
    else:
        print(f"\n❌ Проблема не решена, год: {stats[2].year}")


if __name__ == "__main__":
    fix_with_workaround()