# analyze_loaded_data.py
"""
Анализ загруженных данных и создание оптимизированной таблицы
"""

from clickhouse_driver import Client
import pandas as pd


def analyze_and_optimize():
    """Анализ данных и создание оптимизированной структуры"""

    print("\n" + "=" * 60)
    print("📊 АНАЛИЗ ЗАГРУЖЕННЫХ ДАННЫХ")
    print("=" * 60)

    # Подключение
    client = Client(
        host='localhost',
        port=9000,
        user='analyst',
        password='admin123'
    )

    # Базовая статистика
    print("\n✅ Данные успешно загружены в таблицу transactions_simple")
    count = client.execute('SELECT count() FROM card_analytics.transactions_simple')[0][0]
    print(f"📊 Всего записей: {count:,}")

    # Анализ ключевых полей
    print("\n" + "-" * 60)
    print("🔍 АНАЛИЗ КЛЮЧЕВЫХ ПОЛЕЙ:")
    print("-" * 60)

    # 1. Период данных (rday)
    print("\n📅 Период транзакций:")
    date_stats = client.execute("""
        SELECT 
            toUInt32(min(rday)) as min_rday,
            toUInt32(max(rday)) as max_rday,
            toDate(min(toUInt32(rday))) as min_date,
            toDate(max(toUInt32(rday))) as max_date
        FROM card_analytics.transactions_simple
        WHERE rday != ''
    """)[0]

    print(f"  • Минимальный timestamp: {date_stats[0]}")
    print(f"  • Максимальный timestamp: {date_stats[1]}")
    print(f"  • Период: с {date_stats[2]} по {date_stats[3]}")

    # 2. Суммы транзакций
    print("\n💰 Суммы транзакций (amount_uzs):")
    amount_stats = client.execute("""
        SELECT 
            count() as total,
            countIf(amount_uzs = '') as empty,
            countIf(amount_uzs != '' AND toFloat64OrNull(amount_uzs) IS NULL) as non_numeric,
            min(toFloat64OrNull(amount_uzs)) as min_amount,
            max(toFloat64OrNull(amount_uzs)) as max_amount,
            avg(toFloat64OrNull(amount_uzs)) as avg_amount,
            sum(toFloat64OrNull(amount_uzs)) as total_amount
        FROM card_analytics.transactions_simple
    """)[0]

    print(f"  • Всего записей: {amount_stats[0]:,}")
    print(f"  • Пустых значений: {amount_stats[1]:,}")
    print(f"  • Нечисловых значений: {amount_stats[2]:,}")
    if amount_stats[3]:
        print(f"  • Минимальная сумма: {amount_stats[3]:,.0f} UZS")
        print(f"  • Максимальная сумма: {amount_stats[4]:,.0f} UZS")
        print(f"  • Средняя сумма: {amount_stats[5]:,.0f} UZS")
        print(f"  • Общий объем: {amount_stats[6]:,.0f} UZS")

    # 3. MCC коды
    print("\n🏪 MCC категории:")
    mcc_stats = client.execute("""
        SELECT 
            mcc,
            count() as cnt,
            sum(toFloat64OrNull(amount_uzs)) as volume
        FROM card_analytics.transactions_simple
        WHERE mcc != '' AND toUInt16OrNull(mcc) IS NOT NULL
        GROUP BY mcc
        ORDER BY cnt DESC
        LIMIT 10
    """)

    if mcc_stats:
        print("  Топ-10 MCC по количеству транзакций:")
        for mcc, cnt, volume in mcc_stats:
            volume_str = f"{volume:,.0f}" if volume else "0"
            print(f"    • MCC {mcc}: {cnt:,} транзакций ({volume_str} UZS)")

    # 4. P2P статистика
    print("\n💸 P2P переводы:")
    p2p_stats = client.execute("""
        SELECT 
            p2p_flag,
            count() as cnt,
            avg(toFloat64OrNull(amount_uzs)) as avg_amount,
            sum(toFloat64OrNull(amount_uzs)) as total_amount
        FROM card_analytics.transactions_simple
        WHERE p2p_flag IN ('0', '1', 'True', 'False')
        GROUP BY p2p_flag
    """)

    total_txn = 0
    for flag, cnt, avg_amt, total_amt in p2p_stats:
        total_txn += cnt
        flag_type = "P2P" if flag in ('1', 'True') else "Обычные"
        avg_str = f"{avg_amt:,.0f}" if avg_amt else "0"
        total_str = f"{total_amt:,.0f}" if total_amt else "0"
        print(f"  • {flag_type}: {cnt:,} ({cnt / count * 100:.1f}%) - средняя {avg_str} UZS")

    # 5. Уникальные значения
    print("\n👥 Уникальные сущности:")

    # Карты
    unique_cards = client.execute("""
        SELECT uniq(hpan) 
        FROM card_analytics.transactions_simple 
        WHERE hpan != ''
    """)[0][0]
    print(f"  • Уникальных карт (hpan): {unique_cards:,}")

    # Клиенты
    unique_clients = client.execute("""
        SELECT uniq(pinfl) 
        FROM card_analytics.transactions_simple 
        WHERE pinfl != ''
    """)[0][0]
    print(f"  • Уникальных клиентов (pinfl): {unique_clients:,}")

    # Мерчанты
    unique_merchants = client.execute("""
        SELECT uniq(merchant_name) 
        FROM card_analytics.transactions_simple 
        WHERE merchant_name != ''
    """)[0][0]
    print(f"  • Уникальных мерчантов: {unique_merchants:,}")

    # 6. Банки
    print("\n🏦 Топ-5 банков-эмитентов:")
    bank_stats = client.execute("""
        SELECT 
            emitent_bank,
            count() as cnt
        FROM card_analytics.transactions_simple
        WHERE emitent_bank != ''
        GROUP BY emitent_bank
        ORDER BY cnt DESC
        LIMIT 5
    """)

    for bank, cnt in bank_stats:
        print(f"  • {bank}: {cnt:,} транзакций")

    # 7. Регионы
    print("\n📍 Топ-5 регионов:")
    region_stats = client.execute("""
        SELECT 
            emitent_region,
            count() as cnt
        FROM card_analytics.transactions_simple
        WHERE emitent_region != ''
        GROUP BY emitent_region
        ORDER BY cnt DESC
        LIMIT 5
    """)

    for region, cnt in region_stats:
        print(f"  • {region}: {cnt:,} транзакций")

    # Создание оптимизированной таблицы
    print("\n" + "=" * 60)
    print("🔧 СОЗДАНИЕ ОПТИМИЗИРОВАННОЙ ТАБЛИЦЫ")
    print("=" * 60)

    create_optimized = """
    CREATE TABLE IF NOT EXISTS card_analytics.transactions_optimized
    (
        -- Основные идентификаторы
        hpan String,
        transaction_code String,
        rday UInt32,

        -- Суммы (с проверкой на NULL)
        amount_uzs Float64 DEFAULT 0,
        reqamt Float64 DEFAULT 0,
        conamt Float64 DEFAULT 0,

        -- Категории
        mcc UInt16 DEFAULT 0,
        merchant_name String,
        merchant_type String,

        -- P2P
        p2p_flag UInt8 DEFAULT 0,
        p2p_type String,
        sender_hpan String,
        receiver_hpan String,

        -- Клиент
        pinfl String,
        gender String,
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
        hour_num UInt8 DEFAULT 0,
        minute_num UInt8 DEFAULT 0,

        -- Дополнительные поля как String
        day_type String,
        terminal_type String,
        credit_debit String,
        reversal_flag String,
        respcode String
    )
    ENGINE = MergeTree()
    PARTITION BY toYYYYMM(toDate(rday))
    ORDER BY (rday, hpan, transaction_code)
    SETTINGS index_granularity = 8192
    """

    try:
        client.execute(create_optimized)
        print("✅ Оптимизированная таблица создана")

        # Переливаем данные с преобразованием типов
        print("\n📤 Перенос данных в оптимизированную таблицу...")

        insert_optimized = """
        INSERT INTO card_analytics.transactions_optimized
        SELECT
            hpan,
            transaction_code,
            toUInt32OrDefault(rday, 0) as rday,
            toFloat64OrDefault(amount_uzs, 0) as amount_uzs,
            toFloat64OrDefault(reqamt, 0) as reqamt,
            toFloat64OrDefault(conamt, 0) as conamt,
            toUInt16OrDefault(mcc, 0) as mcc,
            merchant_name,
            merchant_type,
            toUInt8OrDefault(p2p_flag, 0) as p2p_flag,
            p2p_type,
            sender_hpan,
            receiver_hpan,
            pinfl,
            gender,
            age_group,
            emitent_bank,
            emitent_region,
            acquirer_bank,
            acquirer_region,
            card_type,
            card_product_type,
            toUInt8OrDefault(hour_num, 0) as hour_num,
            toUInt8OrDefault(minute_num, 0) as minute_num,
            day_type,
            terminal_type,
            credit_debit,
            reversal_flag,
            respcode
        FROM card_analytics.transactions_simple
        """

        client.execute(insert_optimized)

        # Проверяем результат
        opt_count = client.execute('SELECT count() FROM card_analytics.transactions_optimized')[0][0]
        print(f"✅ Перенесено {opt_count:,} записей")

        # Добавляем индексы
        print("\n📑 Создание индексов...")

        try:
            client.execute("""
                ALTER TABLE card_analytics.transactions_optimized 
                ADD INDEX idx_mcc mcc TYPE minmax GRANULARITY 4
            """)
            print("  ✅ Индекс по MCC")
        except:
            pass

        try:
            client.execute("""
                ALTER TABLE card_analytics.transactions_optimized 
                ADD INDEX idx_amount amount_uzs TYPE minmax GRANULARITY 4
            """)
            print("  ✅ Индекс по суммам")
        except:
            pass

        print("\n" + "=" * 60)
        print("✅ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА!")
        print("=" * 60)

        print("\n📊 Теперь у вас есть две таблицы:")
        print("  1. transactions_simple - все данные как String (100,000 записей)")
        print("  2. transactions_optimized - с правильными типами и индексами")

        print("\n🚀 Следующие шаги:")
        print("  1. Запустить дашборд:")
        print("     streamlit run run_app.py")
        print()
        print("  2. Выполнить анализ в ClickHouse:")
        print("     http://localhost:8123/play")
        print()
        print("  3. Использовать оптимизированную таблицу для запросов:")
        print("     SELECT * FROM card_analytics.transactions_optimized LIMIT 10")

    except Exception as e:
        print(f"⚠️ Ошибка при создании оптимизированной таблицы: {e}")


if __name__ == "__main__":
    analyze_and_optimize()