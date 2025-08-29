#!/usr/bin/env python3
"""
Прямое исправление дат в transactions_optimized
Проблема: даты показывают 2095 вместо 2025
"""

from clickhouse_driver import Client


def fix_optimized_directly():
    print("\n" + "=" * 60)
    print("ИСПРАВЛЕНИЕ ДАТ В transactions_optimized")
    print("=" * 60)

    client = Client(
        host='localhost',
        port=9000,
        user='analyst',
        password='admin123',
        database='card_analytics'
    )

    # 1. Проверяем текущее состояние
    print("\n1. Текущее состояние transactions_optimized...")
    stats = client.execute("""
        SELECT 
            count() as total,
            min(transaction_date) as min_date,
            max(transaction_date) as max_date,
            min(rday) as min_rday,
            max(rday) as max_rday
        FROM transactions_optimized
    """)[0]

    print(f"   Записей: {stats[0]:,}")
    print(f"   Даты сейчас: {stats[1]} - {stats[2]}")
    print(f"   rday: {stats[3]} - {stats[4]}")

    if stats[2].year != 2025:
        print(f"\n⚠️ Даты некорректны (год {stats[2].year}), исправляем...")

        # 2. Пересоздаем таблицу с правильной формулой
        print("\n2. Создание исправленной таблицы...")

        client.execute("DROP TABLE IF EXISTS transactions_correct")

        # Создаем новую таблицу БЕЗ MATERIALIZED для transaction_date
        client.execute("""
        CREATE TABLE transactions_correct
        (
            hpan String,
            transaction_code String,
            rday UInt32,
            transaction_date Date,  -- Обычное поле, не MATERIALIZED
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

        print("   ✅ Новая таблица создана")

        # 3. Копируем данные с правильным расчетом даты
        print("\n3. Копирование данных с исправлением дат...")

        # ВАЖНО: используем базу 1900-01-01 для правильных дат 2025
        client.execute("""
        INSERT INTO transactions_correct
        SELECT
            hpan,
            transaction_code,
            rday,
            toDate('1900-01-01') + rday as transaction_date,  -- База 1900!
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

        # 4. Проверяем результат
        print("\n4. Проверка исправленных дат...")
        new_stats = client.execute("""
            SELECT 
                count() as total,
                min(transaction_date) as min_date,
                max(transaction_date) as max_date
            FROM transactions_correct
        """)[0]

        print(f"   Записей: {new_stats[0]:,}")
        print(f"   Новые даты: {new_stats[1]} - {new_stats[2]}")

        if new_stats[2].year == 2025:
            print("\n✅ Даты исправлены успешно!")

            # 5. Заменяем старую таблицу
            print("\n5. Замена таблицы...")
            client.execute("DROP TABLE IF EXISTS transactions_optimized_broken")
            client.execute("RENAME TABLE transactions_optimized TO transactions_optimized_broken")
            client.execute("RENAME TABLE transactions_correct TO transactions_optimized")

            print("   ✅ Таблица transactions_optimized обновлена")

            print("\n" + "=" * 60)
            print("✅ ГОТОВО!")
            print("=" * 60)
            print("Теперь можно запускать feature_engineering.py")

        else:
            print(f"\n❌ Даты все еще некорректны: {new_stats[2].year}")
    else:
        print("\n✅ Даты уже корректны (2025 год)")


if __name__ == "__main__":
    fix_optimized_directly()