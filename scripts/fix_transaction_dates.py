#!/usr/bin/env python3
"""
Скрипт для исправления дат в таблице transactions_optimized
Проблема: даты рассчитаны неправильно и показывают 2095 год
"""

from clickhouse_driver import Client
from datetime import datetime, timedelta


def check_and_fix_dates():
    """Проверяет и исправляет даты в таблице"""

    print("\n" + "=" * 60)
    print("ПРОВЕРКА И ИСПРАВЛЕНИЕ ДАТ")
    print("=" * 60)

    # Подключение
    client = Client(
        host='localhost',
        port=9000,
        user='analyst',
        password='admin123',
        database='card_analytics'
    )

    # 1. Проверяем текущие даты
    print("\n1. Проверка текущих дат в transactions_optimized...")

    date_check = client.execute("""
        SELECT 
            min(transaction_date) as min_date,
            max(transaction_date) as max_date,
            min(rday) as min_rday,
            max(rday) as max_rday
        FROM transactions_optimized
    """)[0]

    print(f"   Минимальная дата: {date_check[0]}")
    print(f"   Максимальная дата: {date_check[1]}")
    print(f"   Мин rday: {date_check[2]}")
    print(f"   Макс rday: {date_check[3]}")

    # Проверяем, нужно ли исправление
    if date_check[1].year > 2030:
        print("\n⚠️  Обнаружены некорректные даты! Требуется исправление.")

        # 2. Создаем новую таблицу с правильными датами
        print("\n2. Создание исправленной таблицы...")

        client.execute("DROP TABLE IF EXISTS transactions_fixed")

        create_table = """
        CREATE TABLE transactions_fixed
        (
            -- Основные идентификаторы
            hpan String,
            transaction_code String,
            rday UInt32,
            transaction_date Date,  -- Будем вычислять правильно

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
        print("   ✅ Таблица transactions_fixed создана")

        # 3. Копируем данные с правильными датами
        print("\n3. Перенос данных с корректными датами...")

        # Используем базу 1900-01-01 для правильной конвертации!
        # rday 45658 / 365 ≈ 125 лет от 1900 = 2025 год
        insert_query = """
        INSERT INTO transactions_fixed
        SELECT
            hpan,
            transaction_code,
            rday,
            toDate('1900-01-01') + rday as transaction_date,  -- База 1900-01-01!
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
        FROM transactions_optimized_old  -- Берем из старой таблицы если она есть
        """

        client.execute(insert_query)

        # 4. Проверяем новые даты
        print("\n4. Проверка исправленных дат...")

        new_dates = client.execute("""
            SELECT 
                min(transaction_date) as min_date,
                max(transaction_date) as max_date,
                count() as total
            FROM transactions_fixed
        """)[0]

        print(f"   Новые даты: {new_dates[0]} - {new_dates[1]}")
        print(f"   Всего записей: {new_dates[2]:,}")

        # 5. Переименовываем таблицы
        print("\n5. Замена таблиц...")

        client.execute("DROP TABLE IF EXISTS transactions_optimized_old")
        client.execute("RENAME TABLE transactions_optimized TO transactions_optimized_old")
        client.execute("RENAME TABLE transactions_fixed TO transactions_optimized")

        print("   ✅ Таблица transactions_optimized обновлена")

        print("\n" + "=" * 60)
        print("✅ ДАТЫ УСПЕШНО ИСПРАВЛЕНЫ!")
        print("=" * 60)
        print(f"\nТеперь период данных: {new_dates[0]} - {new_dates[1]}")

    else:
        print("\n✅ Даты корректны, исправление не требуется")
        print(f"   Период: {date_check[0]} - {date_check[1]}")


if __name__ == "__main__":
    check_and_fix_dates()