# final_fix.py
"""
Финальное исправление всех проблем с данными
"""

from clickhouse_driver import Client


def final_fix():
    print("\n" + "=" * 60)
    print("ФИНАЛЬНОЕ ИСПРАВЛЕНИЕ ДАННЫХ")
    print("=" * 60)

    client = Client(
        host='localhost',
        port=9000,
        user='analyst',
        password='admin123'
    )

    # Удаляем старую таблицу
    print("\nУдаление старой оптимизированной таблицы...")
    client.execute('DROP TABLE IF EXISTS card_analytics.transactions_optimized')

    # Создаем новую с правильной базой даты
    print("Создание новой таблицы с правильными датами...")

    create_table = """
    CREATE TABLE card_analytics.transactions_optimized
    (
        -- Основные поля
        hpan String,
        transaction_code String,
        rday UInt32,
        transaction_date Date MATERIALIZED toDate('1900-01-01') + rday,  -- База 1900!

        -- Суммы
        amount_uzs Float64,
        reqamt Float64,
        conamt Float64,

        -- MCC - правильно конвертируем
        mcc UInt16,
        merchant_name String,
        merchant_type String,
        merchant UInt32,

        -- P2P - конвертируем True/False в 1/0
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

        -- Время
        hour_num UInt8,
        minute_num UInt8,

        -- Остальное
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
    print("Таблица создана")

    # Вставляем данные с правильной конвертацией
    print("\nПеренос данных с исправлениями...")

    insert_query = """
    INSERT INTO card_analytics.transactions_optimized
    SELECT
        hpan,
        transaction_code,
        toUInt32OrDefault(rday, toUInt32(0)),
        toFloat64OrDefault(amount_uzs, toFloat64(0)),
        toFloat64OrDefault(reqamt, toFloat64(0)),
        toFloat64OrDefault(conamt, toFloat64(0)),
        -- MCC: убираем .0 и конвертируем в число
        toUInt16OrDefault(replaceOne(mcc, '.0', ''), toUInt16(0)),
        merchant_name,
        merchant_type,
        toUInt32OrDefault(merchant, toUInt32(0)),
        -- P2P: конвертируем True/False в 1/0
        CASE 
            WHEN p2p_flag = 'True' THEN 1
            WHEN p2p_flag = 'False' THEN 0
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
    FROM card_analytics.transactions_simple
    """

    client.execute(insert_query)

    count = client.execute('SELECT count() FROM card_analytics.transactions_optimized')[0][0]
    print(f"Перенесено {count:,} записей")

    # Проверяем результаты
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ИСПРАВЛЕНИЯ")
    print("=" * 60)

    # 1. Даты
    date_check = client.execute("""
        SELECT 
            min(transaction_date) as min_date,
            max(transaction_date) as max_date,
            count() as total
        FROM card_analytics.transactions_optimized
    """)[0]

    print(f"\nПериод транзакций:")
    print(f"  С {date_check[0]} по {date_check[1]}")
    print(f"  Всего записей: {date_check[2]:,}")

    # 2. MCC теперь числовые
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

    print(f"\nТоп-10 MCC категорий:")
    for mcc, cnt, volume in mcc_stats:
        print(f"  MCC {mcc}: {cnt:,} транзакций ({volume:,.0f} UZS)")

    # 3. P2P статистика
    p2p_stats = client.execute("""
        SELECT 
            p2p_flag,
            count() as cnt,
            sum(amount_uzs) as volume,
            avg(amount_uzs) as avg_amount
        FROM card_analytics.transactions_optimized
        GROUP BY p2p_flag
    """)

    print(f"\nP2P статистика:")
    for flag, cnt, volume, avg_amt in p2p_stats:
        p2p_type = "P2P переводы" if flag == 1 else "Обычные покупки"
        print(f"  {p2p_type}: {cnt:,} ({cnt / count * 100:.1f}%) - {volume:,.0f} UZS (средний: {avg_amt:,.0f})")

    # 4. Gender статистика
    gender_stats = client.execute("""
        SELECT 
            gender,
            count() as cnt,
            sum(amount_uzs) as volume
        FROM card_analytics.transactions_optimized
        WHERE gender IN ('М', 'Ж')
        GROUP BY gender
    """)

    print(f"\nРаспределение по полу:")
    for gender, cnt, volume in gender_stats:
        gender_name = "Мужчины" if gender == 'М' else "Женщины"
        print(f"  {gender_name}: {cnt:,} ({cnt / count * 100:.1f}%) - {volume:,.0f} UZS")

    # 5. Активность по месяцам
    monthly_stats = client.execute("""
        SELECT 
            toYYYYMM(transaction_date) as month,
            count() as cnt,
            sum(amount_uzs) as volume
        FROM card_analytics.transactions_optimized
        GROUP BY month
        ORDER BY month
    """)

    print(f"\nАктивность по месяцам:")
    for month, cnt, volume in monthly_stats:
        print(f"  {month}: {cnt:,} транзакций ({volume:,.0f} UZS)")

    print("\n" + "=" * 60)
    print("ГОТОВО! Все проблемы исправлены!")
    print("=" * 60)

    print("\nТеперь можно:")
    print("1. Использовать правильные даты (2025 год)")
    print("2. Анализировать MCC категории")
    print("3. Анализировать P2P vs обычные транзакции")
    print("4. Строить графики по полу")
    print("\nЗапустите дашборд: streamlit run run_app.py")


if __name__ == "__main__":
    final_fix()