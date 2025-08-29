# diagnose_data.py
"""
Диагностика проблем с данными - почему MCC и gender пустые
"""

from clickhouse_driver import Client


def diagnose_data():
    """Детальная диагностика данных"""

    print("\n" + "=" * 60)
    print("🔍 ДИАГНОСТИКА ДАННЫХ")
    print("=" * 60)

    # Подключение
    client = Client(
        host='localhost',
        port=9000,
        user='analyst',
        password='admin123'
    )

    # 1. Проверка MCC
    print("\n📊 АНАЛИЗ MCC КОДОВ:")
    print("-" * 40)

    # Уникальные значения MCC
    mcc_values = client.execute("""
        SELECT 
            mcc,
            count() as cnt
        FROM card_analytics.transactions_simple
        GROUP BY mcc
        ORDER BY cnt DESC
        LIMIT 20
    """)

    print("Топ-20 значений MCC (как есть в данных):")
    for mcc, cnt in mcc_values:
        print(f"  '{mcc}': {cnt:,} записей")

    # Проверка на числовые MCC
    mcc_numeric = client.execute("""
        SELECT 
            countIf(mcc = '') as empty,
            countIf(mcc = '0') as zero,
            countIf(toUInt16OrNull(mcc) IS NULL AND mcc != '') as non_numeric,
            countIf(toUInt16OrNull(mcc) > 0) as valid_mcc
        FROM card_analytics.transactions_simple
    """)[0]

    print(f"\nСтатистика MCC:")
    print(f"  • Пустые: {mcc_numeric[0]:,}")
    print(f"  • Нулевые: {mcc_numeric[1]:,}")
    print(f"  • Не числовые: {mcc_numeric[2]:,}")
    print(f"  • Валидные (>0): {mcc_numeric[3]:,}")

    # 2. Проверка gender
    print("\n👥 АНАЛИЗ ПОЛЯ GENDER:")
    print("-" * 40)

    gender_values = client.execute("""
        SELECT 
            gender,
            count() as cnt
        FROM card_analytics.transactions_simple
        GROUP BY gender
        ORDER BY cnt DESC
    """)

    print("Все значения gender:")
    for gender, cnt in gender_values:
        print(f"  '{gender}': {cnt:,} записей")

    # 3. Проверка P2P
    print("\n💸 АНАЛИЗ P2P_FLAG:")
    print("-" * 40)

    p2p_values = client.execute("""
        SELECT 
            p2p_flag,
            count() as cnt
        FROM card_analytics.transactions_simple
        GROUP BY p2p_flag
        ORDER BY cnt DESC
    """)

    print("Все значения p2p_flag:")
    for p2p, cnt in p2p_values:
        print(f"  '{p2p}': {cnt:,} записей")

    # 4. Проверка дат (rday)
    print("\n📅 АНАЛИЗ ДАТ (rday):")
    print("-" * 40)

    date_stats = client.execute("""
        SELECT 
            min(toUInt32OrNull(rday)) as min_rday,
            max(toUInt32OrNull(rday)) as max_rday,
            countIf(rday = '') as empty,
            countIf(toUInt32OrNull(rday) IS NULL AND rday != '') as non_numeric
        FROM card_analytics.transactions_simple
    """)[0]

    print(f"  • Минимальное rday: {date_stats[0]}")
    print(f"  • Максимальное rday: {date_stats[1]}")
    print(f"  • Диапазон: {date_stats[1] - date_stats[0] if date_stats[0] and date_stats[1] else 'N/A'} дней")
    print(f"  • Пустых: {date_stats[2]:,}")
    print(f"  • Не числовых: {date_stats[3]:,}")

    # Конвертация дат
    if date_stats[0]:
        from datetime import datetime, timedelta

        # Разные базовые даты
        bases = {
            '1900-01-01': datetime(1900, 1, 1),
            '1970-01-01': datetime(1970, 1, 1),
            '2000-01-01': datetime(2000, 1, 1),
            '2020-01-01': datetime(2020, 1, 1)
        }

        print("\nВозможные интерпретации дат:")
        for base_name, base_date in bases.items():
            min_date = base_date + timedelta(days=date_stats[0])
            max_date = base_date + timedelta(days=date_stats[1])
            print(f"  • База {base_name}: {min_date.date()} - {max_date.date()}")

    # 5. Проверка сумм
    print("\n💰 АНАЛИЗ СУММ (amount_uzs):")
    print("-" * 40)

    amount_stats = client.execute("""
        SELECT 
            countIf(amount_uzs = '') as empty,
            countIf(amount_uzs = '0') as zero,
            countIf(toFloat64OrNull(amount_uzs) IS NULL AND amount_uzs != '') as non_numeric,
            min(toFloat64OrNull(amount_uzs)) as min_amount,
            max(toFloat64OrNull(amount_uzs)) as max_amount,
            avg(toFloat64OrNull(amount_uzs)) as avg_amount
        FROM card_analytics.transactions_simple
    """)[0]

    print(f"  • Пустых: {amount_stats[0]:,}")
    print(f"  • Нулевых: {amount_stats[1]:,}")
    print(f"  • Не числовых: {amount_stats[2]:,}")
    print(f"  • Минимальная: {amount_stats[3]:,.0f} UZS" if amount_stats[3] else "N/A")
    print(f"  • Максимальная: {amount_stats[4]:,.0f} UZS" if amount_stats[4] else "N/A")
    print(f"  • Средняя: {amount_stats[5]:,.0f} UZS" if amount_stats[5] else "N/A")

    # 6. Примеры записей
    print("\n📋 ПРИМЕРЫ ЗАПИСЕЙ:")
    print("-" * 40)

    samples = client.execute("""
        SELECT 
            mcc,
            gender,
            p2p_flag,
            amount_uzs,
            rday,
            merchant_name
        FROM card_analytics.transactions_simple
        LIMIT 5
    """)

    print("Первые 5 записей:")
    for i, (mcc, gender, p2p, amount, rday, merchant) in enumerate(samples, 1):
        print(f"\n{i}. MCC: '{mcc}' | Gender: '{gender}' | P2P: '{p2p}'")
        print(f"   Amount: '{amount}' | RDay: '{rday}'")
        print(f"   Merchant: '{merchant[:30]}...' " if len(merchant) > 30 else f"   Merchant: '{merchant}'")

    # 7. Рекомендации
    print("\n" + "=" * 60)
    print("💡 РЕКОМЕНДАЦИИ")
    print("=" * 60)

    print("""
1. Если MCC все нулевые или пустые:
   - Данные могут быть анонимизированы
   - Используйте merchant_name для категоризации

2. Если gender пустой:
   - Поле не заполнялось при сборе данных
   - Можно попробовать определить по имени (если есть)

3. Для правильной конвертации дат:
   - Используйте базу 2000-01-01 (даты будут 2024-2025)
   - Или уточните у источника данных формат rday

4. Запустите исправление:
   python fix_table_final.py
    """)


if __name__ == "__main__":
    diagnose_data()