# fix_table_final.py
"""
Финальное исправление таблицы с правильными типами
"""

from clickhouse_driver import Client
from datetime import datetime, timedelta


def fix_table_final():
    """Финальное исправление с правильными типами"""

    print("\n" + "=" * 60)
    print("🔧 ФИНАЛЬНОЕ ИСПРАВЛЕНИЕ ТАБЛИЦЫ")
    print("=" * 60)

    # Подключение
    client = Client(
        host='localhost',
        port=9000,
        user='analyst',
        password='admin123'
    )

    # Удаляем старую таблицу
    print("\n🗑️ Удаление старой оптимизированной таблицы...")
    client.execute('DROP TABLE IF EXISTS card_analytics.transactions_optimized')

    # Создаем новую таблицу
    print("\n📝 Создание новой оптимизированной таблицы...")

    create_table = """
    CREATE TABLE card_analytics.transactions_optimized
    (
        -- Основные идентификаторы
        hpan String,
        transaction_code String,
        rday UInt32,
        transaction_date Date MATERIALIZED toDate('2000-01-01') + rday,

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
    SETTINGS index_granularity = 8192
    """

    try:
        client.execute(create_table)
        print("✅ Таблица создана")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return

    # Переносим данные с ПРАВИЛЬНЫМИ типами для default
    print("\n📤 Перенос данных с преобразованием типов...")

    # ВАЖНО: Используем правильные типы для default значений!
    insert_query = """
    INSERT INTO card_analytics.transactions_optimized
    SELECT
        hpan,
        transaction_code,
        toUInt32OrDefault(rday, toUInt32(0)),  -- Явно указываем тип
        toFloat64OrDefault(amount_uzs, toFloat64(0)),  -- Float64
        toFloat64OrDefault(reqamt, toFloat64(0)),
        toFloat64OrDefault(conamt, toFloat64(0)),
        toUInt16OrDefault(mcc, toUInt16(0)),
        merchant_name,
        merchant_type,
        toUInt32OrDefault(merchant, toUInt32(0)),
        toUInt8OrDefault(p2p_flag, toUInt8(0)),
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
    FROM card_analytics.transactions_simple
    """

    try:
        client.execute(insert_query)

        # Проверяем результат
        count = client.execute('SELECT count() FROM card_analytics.transactions_optimized')[0][0]
        print(f"✅ Перенесено {count:,} записей")

        if count == 0:
            print("⚠️ Записи не перенеслись, пробуем альтернативный метод...")

            # Альтернативный метод - без OrDefault функций
            insert_simple = """
            INSERT INTO card_analytics.transactions_optimized
            SELECT
                hpan,
                transaction_code,
                CAST(assumeNotNull(toUInt32OrNull(rday)) AS UInt32),
                CAST(assumeNotNull(toFloat64OrNull(amount_uzs)) AS Float64),
                CAST(assumeNotNull(toFloat64OrNull(reqamt)) AS Float64),
                CAST(assumeNotNull(toFloat64OrNull(conamt)) AS Float64),
                CAST(assumeNotNull(toUInt16OrNull(mcc)) AS UInt16),
                merchant_name,
                merchant_type,
                CAST(assumeNotNull(toUInt32OrNull(merchant)) AS UInt32),
                CAST(assumeNotNull(toUInt8OrNull(p2p_flag)) AS UInt8),
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
                CAST(assumeNotNull(toUInt8OrNull(hour_num)) AS UInt8),
                CAST(assumeNotNull(toUInt8OrNull(minute_num)) AS UInt8),
                day_type,
                terminal_type,
                terminal_id,
                credit_debit,
                reversal_flag,
                respcode,
                ip,
                login
            FROM card_analytics.transactions_simple
            """

            client.execute(insert_simple)
            count = client.execute('SELECT count() FROM card_analytics.transactions_optimized')[0][0]
            print(f"✅ Альтернативный метод: перенесено {count:,} записей")

    except Exception as e:
        print(f"❌ Ошибка при переносе: {e}")

        # Последняя попытка - создаем все поля Nullable
        print("\n🔄 Пробуем создать таблицу с Nullable полями...")

        client.execute('DROP TABLE IF EXISTS card_analytics.transactions_optimized')

        create_nullable = """
        CREATE TABLE card_analytics.transactions_optimized
        (
            hpan String,
            transaction_code String,
            rday Nullable(UInt32),
            transaction_date Date MATERIALIZED toDate('2000-01-01') + assumeNotNull(rday),
            amount_uzs Nullable(Float64),
            reqamt Nullable(Float64),
            conamt Nullable(Float64),
            mcc Nullable(UInt16),
            merchant_name String,
            merchant_type String,
            merchant Nullable(UInt32),
            p2p_flag Nullable(UInt8),
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
            hour_num Nullable(UInt8),
            minute_num Nullable(UInt8),
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
        ORDER BY (hpan, transaction_code)
        """

        client.execute(create_nullable)

        # Простая вставка
        insert_nullable = """
        INSERT INTO card_analytics.transactions_optimized
        SELECT
            hpan,
            transaction_code,
            toUInt32OrNull(rday),
            toFloat64OrNull(amount_uzs),
            toFloat64OrNull(reqamt),
            toFloat64OrNull(conamt),
            toUInt16OrNull(mcc),
            merchant_name,
            merchant_type,
            toUInt32OrNull(merchant),
            toUInt8OrNull(p2p_flag),
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
            toUInt8OrNull(hour_num),
            toUInt8OrNull(minute_num),
            day_type,
            terminal_type,
            terminal_id,
            credit_debit,
            reversal_flag,
            respcode,
            ip,
            login
        FROM card_analytics.transactions_simple
        """

        client.execute(insert_nullable)
        count = client.execute('SELECT count() FROM card_analytics.transactions_optimized')[0][0]
        print(f"✅ С Nullable полями: перенесено {count:,} записей")

    # Показываем статистику
    if count > 0:
        print("\n" + "=" * 60)
        print("📊 СТАТИСТИКА ОПТИМИЗИРОВАННОЙ ТАБЛИЦЫ")
        print("=" * 60)

        # Проверка дат
        date_check = client.execute("""
            SELECT 
                min(transaction_date) as min_date,
                max(transaction_date) as max_date,
                count() as total
            FROM card_analytics.transactions_optimized
            WHERE transaction_date IS NOT NULL
        """)[0]

        print(f"\n📅 Период транзакций:")
        print(f"  • С {date_check[0]} по {date_check[1]}")
        print(f"  • Всего записей: {date_check[2]:,}")

        # MCC статистика
        print(f"\n🏪 Топ-10 MCC категорий:")
        mcc_stats = client.execute("""
            SELECT 
                mcc,
                count() as cnt,
                sum(amount_uzs) as volume
            FROM card_analytics.transactions_optimized
            WHERE mcc > 0
            GROUP BY mcc
            ORDER BY cnt DESC
            LIMIT 10
        """)

        if mcc_stats:
            for mcc, cnt, volume in mcc_stats:
                print(f"  • MCC {mcc}: {cnt:,} транзакций ({volume:,.0f} UZS)")
        else:
            print("  ⚠️ MCC коды отсутствуют или нулевые")

            # Проверим что в исходных данных
            mcc_check = client.execute("""
                SELECT 
                    mcc,
                    count() as cnt
                FROM card_analytics.transactions_simple
                WHERE mcc != '' AND mcc != '0'
                GROUP BY mcc
                ORDER BY cnt DESC
                LIMIT 5
            """)

            if mcc_check:
                print("\n  Найдены MCC в исходной таблице:")
                for mcc, cnt in mcc_check:
                    print(f"    • {mcc}: {cnt:,}")

        # P2P статистика
        p2p_stats = client.execute("""
            SELECT 
                p2p_flag,
                count() as cnt
            FROM card_analytics.transactions_optimized
            GROUP BY p2p_flag
        """)

        print(f"\n💸 P2P статистика:")
        for flag, cnt in p2p_stats:
            p2p_type = "P2P" if flag == 1 else "Обычные"
            print(f"  • {p2p_type}: {cnt:,} ({cnt / count * 100:.1f}%)")

        print("\n" + "=" * 60)
        print("✅ ТАБЛИЦА ГОТОВА К ИСПОЛЬЗОВАНИЮ!")
        print("=" * 60)

        print("\n📊 Проверьте в ClickHouse UI:")
        print("  http://localhost:8123/play")
        print("""
SELECT 
    transaction_date,
    count() as cnt,
    sum(amount_uzs) as volume
FROM card_analytics.transactions_optimized
WHERE transaction_date IS NOT NULL
GROUP BY transaction_date
ORDER BY transaction_date DESC
LIMIT 30;
        """)


if __name__ == "__main__":
    fix_table_final()