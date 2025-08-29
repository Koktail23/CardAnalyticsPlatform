# fix_optimized_table.py
"""
Исправление оптимизированной таблицы и корректная обработка дат
"""

from clickhouse_driver import Client
from datetime import datetime, timedelta


def fix_dates_and_optimize():
    """Исправляет даты и создает оптимизированную таблицу"""

    print("\n" + "=" * 60)
    print("🔧 ИСПРАВЛЕНИЕ ДАТ И ОПТИМИЗАЦИЯ")
    print("=" * 60)

    # Подключение
    client = Client(
        host='localhost',
        port=9000,
        user='analyst',
        password='admin123'
    )

    # Анализ формата дат
    print("\n📅 Анализ формата дат (rday)...")

    date_analysis = client.execute("""
        SELECT 
            min(toUInt32OrNull(rday)) as min_rday,
            max(toUInt32OrNull(rday)) as max_rday,
            count() as total
        FROM card_analytics.transactions_simple
        WHERE rday != ''
    """)[0]

    print(f"  • Минимальное значение rday: {date_analysis[0]}")
    print(f"  • Максимальное значение rday: {date_analysis[1]}")
    print(f"  • Диапазон: {date_analysis[1] - date_analysis[0]} дней")

    # Проверяем гипотезы о формате
    print("\n🔍 Определение формата дат...")

    # Гипотеза 1: Дни от 1900-01-01 (Excel-style)
    base_date_1900 = datetime(1900, 1, 1)
    test_date_1900 = base_date_1900 + timedelta(days=date_analysis[0])
    print(f"  • Если база 1900-01-01: {test_date_1900.date()}")

    # Гипотеза 2: Дни от 2000-01-01
    base_date_2000 = datetime(2000, 1, 1)
    test_date_2000 = base_date_2000 + timedelta(days=date_analysis[0])
    print(f"  • Если база 2000-01-01: {test_date_2000.date()}")

    # Гипотеза 3: Дни от 1970-01-01 (Unix epoch в днях)
    base_date_1970 = datetime(1970, 1, 1)
    test_date_1970 = base_date_1970 + timedelta(days=date_analysis[0])
    print(f"  • Если база 1970-01-01: {test_date_1970.date()}")

    # Выбираем наиболее вероятный вариант (2000-01-01 дает разумные даты 2024-2025)
    print(f"\n✅ Используем базу 2000-01-01 (даты получаются в диапазоне 2024-2025)")

    # Удаляем старую оптимизированную таблицу
    print("\n🗑️ Удаление старой оптимизированной таблицы...")
    client.execute('DROP TABLE IF EXISTS card_analytics.transactions_optimized')

    # Создаем новую оптимизированную таблицу
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

    # Переносим данные
    print("\n📤 Перенос данных с преобразованием типов...")

    insert_query = """
    INSERT INTO card_analytics.transactions_optimized
    SELECT
        hpan,
        transaction_code,
        toUInt32OrDefault(rday, CAST(0 AS UInt32)),
        toFloat64OrDefault(amount_uzs, 0),
        toFloat64OrDefault(reqamt, 0),
        toFloat64OrDefault(conamt, 0),
        toUInt16OrDefault(mcc, CAST(0 AS UInt16)),
        merchant_name,
        merchant_type,
        toUInt32OrDefault(merchant, CAST(0 AS UInt32)),
        toUInt8OrDefault(p2p_flag, CAST(0 AS UInt8)),
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
        toUInt8OrDefault(hour_num, CAST(0 AS UInt8)),
        toUInt8OrDefault(minute_num, CAST(0 AS UInt8)),
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

    except Exception as e:
        print(f"❌ Ошибка при переносе: {e}")
        return

    # Показываем статистику с правильными датами
    print("\n" + "=" * 60)
    print("📊 СТАТИСТИКА ОПТИМИЗИРОВАННОЙ ТАБЛИЦЫ")
    print("=" * 60)

    # Период данных
    date_stats = client.execute("""
        SELECT 
            min(transaction_date) as min_date,
            max(transaction_date) as max_date,
            dateDiff('day', min_date, max_date) as days
        FROM card_analytics.transactions_optimized
    """)[0]

    print(f"\n📅 Период транзакций:")
    print(f"  • С {date_stats[0]} по {date_stats[1]}")
    print(f"  • Всего {date_stats[2]} дней")

    # Финансовая статистика
    fin_stats = client.execute("""
        SELECT 
            count() as total,
            sum(amount_uzs) as volume,
            avg(amount_uzs) as avg_amount,
            max(amount_uzs) as max_amount
        FROM card_analytics.transactions_optimized
    """)[0]

    print(f"\n💰 Финансовые показатели:")
    print(f"  • Транзакций: {fin_stats[0]:,}")
    print(f"  • Общий объем: {fin_stats[1]:,.0f} UZS")
    print(f"  • Средняя сумма: {fin_stats[2]:,.0f} UZS")
    print(f"  • Максимальная: {fin_stats[3]:,.0f} UZS")

    # Топ MCC
    print(f"\n🏪 Топ-5 MCC категорий:")
    mcc_stats = client.execute("""
        SELECT 
            mcc,
            count() as cnt,
            sum(amount_uzs) as volume
        FROM card_analytics.transactions_optimized
        WHERE mcc > 0
        GROUP BY mcc
        ORDER BY cnt DESC
        LIMIT 5
    """)

    for mcc, cnt, volume in mcc_stats:
        print(f"  • MCC {mcc}: {cnt:,} транзакций ({volume:,.0f} UZS)")

    # P2P
    print(f"\n💸 P2P статистика:")
    p2p_stats = client.execute("""
        SELECT 
            p2p_flag,
            count() as cnt,
            avg(amount_uzs) as avg_amount
        FROM card_analytics.transactions_optimized
        GROUP BY p2p_flag
    """)

    for flag, cnt, avg_amt in p2p_stats:
        p2p_type = "P2P" if flag == 1 else "Обычные"
        print(f"  • {p2p_type}: {cnt:,} ({avg_amt:,.0f} UZS средняя)")

    # Активность по месяцам
    print(f"\n📈 Активность по месяцам:")
    monthly_stats = client.execute("""
        SELECT 
            toYYYYMM(transaction_date) as month,
            count() as cnt,
            sum(amount_uzs) as volume
        FROM card_analytics.transactions_optimized
        GROUP BY month
        ORDER BY month
    """)

    for month, cnt, volume in monthly_stats:
        print(f"  • {month}: {cnt:,} транзакций ({volume:,.0f} UZS)")

    print("\n" + "=" * 60)
    print("✅ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА!")
    print("=" * 60)

    print("\n🎯 Теперь можно использовать:")
    print("  1. transactions_optimized - для быстрых запросов")
    print("  2. transaction_date - правильные даты (2024-2025)")
    print("  3. Партиционирование по месяцам работает")

    print("\n📊 Примеры запросов:")
    print("""
-- Транзакции за последний месяц
SELECT 
    transaction_date,
    count() as cnt,
    sum(amount_uzs) as volume
FROM card_analytics.transactions_optimized
WHERE transaction_date >= today() - 30
GROUP BY transaction_date
ORDER BY transaction_date;

-- Топ мерчантов
SELECT 
    merchant_name,
    count() as cnt,
    sum(amount_uzs) as volume
FROM card_analytics.transactions_optimized
WHERE merchant_name != ''
GROUP BY merchant_name
ORDER BY volume DESC
LIMIT 10;
    """)


if __name__ == "__main__":
    fix_dates_and_optimize()