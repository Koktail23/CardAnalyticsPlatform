# run_analysis.py
"""
Выполнение аналитических запросов к данным
"""

from clickhouse_driver import Client
import pandas as pd
from datetime import datetime


def run_analysis():
    """Выполняет серию аналитических запросов"""

    print("\n" + "=" * 60)
    print("📊 КОМПЛЕКСНЫЙ АНАЛИЗ КАРТОЧНЫХ ТРАНЗАКЦИЙ")
    print("=" * 60)

    # Подключение
    client = Client(
        host='localhost',
        port=9000,
        user='analyst',
        password='admin123'
    )

    # Проверка какие таблицы доступны
    tables = client.execute("""
        SELECT name, total_rows
        FROM system.tables
        WHERE database = 'card_analytics'
            AND name NOT LIKE '.%'
        ORDER BY name
    """)

    print("\n📂 Доступные таблицы:")
    table_to_use = None
    for name, rows in tables:
        print(f"  • {name}: {rows:,} записей")
        if rows > 0 and not table_to_use:
            table_to_use = f"card_analytics.{name}"

    if not table_to_use:
        print("\n❌ Нет таблиц с данными!")
        return

    print(f"\n✅ Используем таблицу: {table_to_use}")

    # 1. ОБЩАЯ СТАТИСТИКА
    print("\n" + "-" * 60)
    print("1️⃣ ОБЩАЯ СТАТИСТИКА")
    print("-" * 60)

    general_stats = client.execute(f"""
        SELECT 
            count() as total_transactions,
            sum(toFloat64OrNull(amount_uzs)) as total_volume,
            avg(toFloat64OrNull(amount_uzs)) as avg_amount,
            max(toFloat64OrNull(amount_uzs)) as max_amount,
            min(toFloat64OrNull(amount_uzs)) as min_amount,
            uniq(hpan) as unique_cards,
            uniq(pinfl) as unique_clients,
            uniq(merchant_name) as unique_merchants
        FROM {table_to_use}
    """)[0]

    print(f"📈 Транзакции: {general_stats[0]:,}")
    print(f"💰 Общий объем: {general_stats[1]:,.0f} UZS" if general_stats[1] else "💰 Общий объем: Н/Д")
    print(f"💵 Средний чек: {general_stats[2]:,.0f} UZS" if general_stats[2] else "💵 Средний чек: Н/Д")
    print(f"📊 Макс. транзакция: {general_stats[3]:,.0f} UZS" if general_stats[3] else "📊 Макс. транзакция: Н/Д")
    print(f"👥 Уникальных карт: {general_stats[5]:,}")
    print(f"👤 Уникальных клиентов: {general_stats[6]:,}")
    print(f"🏪 Уникальных мерчантов: {general_stats[7]:,}")

    # 2. ТОП MCC КАТЕГОРИЙ
    print("\n" + "-" * 60)
    print("2️⃣ ТОП-10 MCC КАТЕГОРИЙ")
    print("-" * 60)

    mcc_stats = client.execute(f"""
        SELECT 
            mcc,
            count() as cnt,
            sum(toFloat64OrNull(amount_uzs)) as volume,
            avg(toFloat64OrNull(amount_uzs)) as avg_amount
        FROM {table_to_use}
        WHERE mcc != '' AND toUInt16OrNull(mcc) IS NOT NULL
        GROUP BY mcc
        ORDER BY cnt DESC
        LIMIT 10
    """)

    if mcc_stats:
        for i, (mcc, cnt, volume, avg_amt) in enumerate(mcc_stats, 1):
            volume_str = f"{volume:,.0f}" if volume else "0"
            avg_str = f"{avg_amt:,.0f}" if avg_amt else "0"
            percent = cnt / general_stats[0] * 100
            print(f"{i:2}. MCC {mcc}: {cnt:,} ({percent:.1f}%) | Объем: {volume_str} UZS | Средний: {avg_str} UZS")

    # 3. P2P АНАЛИЗ
    print("\n" + "-" * 60)
    print("3️⃣ P2P АНАЛИЗ")
    print("-" * 60)

    p2p_stats = client.execute(f"""
        SELECT 
            CASE 
                WHEN p2p_flag IN ('1', 'True') OR p2p_flag = '1' THEN 'P2P'
                ELSE 'Покупки'
            END as type,
            count() as cnt,
            sum(toFloat64OrNull(amount_uzs)) as volume,
            avg(toFloat64OrNull(amount_uzs)) as avg_amount
        FROM {table_to_use}
        GROUP BY type
    """)

    for tx_type, cnt, volume, avg_amt in p2p_stats:
        percent = cnt / general_stats[0] * 100
        volume_str = f"{volume:,.0f}" if volume else "0"
        avg_str = f"{avg_amt:,.0f}" if avg_amt else "0"
        emoji = "💸" if tx_type == "P2P" else "🛒"
        print(f"{emoji} {tx_type}: {cnt:,} ({percent:.1f}%) | Объем: {volume_str} UZS | Средний: {avg_str} UZS")

    # 4. ТОП БАНКОВ
    print("\n" + "-" * 60)
    print("4️⃣ ТОП-10 БАНКОВ-ЭМИТЕНТОВ")
    print("-" * 60)

    bank_stats = client.execute(f"""
        SELECT 
            emitent_bank,
            count() as cnt,
            sum(toFloat64OrNull(amount_uzs)) as volume,
            uniq(hpan) as unique_cards
        FROM {table_to_use}
        WHERE emitent_bank != ''
        GROUP BY emitent_bank
        ORDER BY cnt DESC
        LIMIT 10
    """)

    for i, (bank, cnt, volume, cards) in enumerate(bank_stats, 1):
        percent = cnt / general_stats[0] * 100
        volume_str = f"{volume:,.0f}" if volume else "0"
        print(f"{i:2}. {bank}: {cnt:,} ({percent:.1f}%) | Карт: {cards:,} | Объем: {volume_str} UZS")

    # 5. ТОП РЕГИОНОВ
    print("\n" + "-" * 60)
    print("5️⃣ ТОП-10 РЕГИОНОВ")
    print("-" * 60)

    region_stats = client.execute(f"""
        SELECT 
            emitent_region,
            count() as cnt,
            sum(toFloat64OrNull(amount_uzs)) as volume,
            uniq(pinfl) as unique_clients
        FROM {table_to_use}
        WHERE emitent_region != ''
        GROUP BY emitent_region
        ORDER BY cnt DESC
        LIMIT 10
    """)

    for i, (region, cnt, volume, clients) in enumerate(region_stats, 1):
        percent = cnt / general_stats[0] * 100
        volume_str = f"{volume:,.0f}" if volume else "0"
        print(f"{i:2}. {region}: {cnt:,} ({percent:.1f}%) | Клиентов: {clients:,} | Объем: {volume_str} UZS")

    # 6. ТОП МЕРЧАНТОВ
    print("\n" + "-" * 60)
    print("6️⃣ ТОП-20 МЕРЧАНТОВ")
    print("-" * 60)

    merchant_stats = client.execute(f"""
        SELECT 
            merchant_name,
            count() as cnt,
            sum(toFloat64OrNull(amount_uzs)) as volume,
            avg(toFloat64OrNull(amount_uzs)) as avg_check
        FROM {table_to_use}
        WHERE merchant_name != ''
        GROUP BY merchant_name
        ORDER BY volume DESC
        LIMIT 20
    """)

    for i, (merchant, cnt, volume, avg_check) in enumerate(merchant_stats, 1):
        volume_str = f"{volume:,.0f}" if volume else "0"
        avg_str = f"{avg_check:,.0f}" if avg_check else "0"
        print(f"{i:2}. {merchant[:50]}: {cnt:,} транз. | {volume_str} UZS | Ср.чек: {avg_str}")

    # 7. ВРЕМЕННОЙ АНАЛИЗ
    print("\n" + "-" * 60)
    print("7️⃣ АКТИВНОСТЬ ПО ЧАСАМ")
    print("-" * 60)

    hourly_stats = client.execute(f"""
        SELECT 
            toUInt8OrDefault(hour_num, 0) as hour,
            count() as cnt,
            sum(toFloat64OrNull(amount_uzs)) as volume
        FROM {table_to_use}
        WHERE hour_num != ''
        GROUP BY hour
        ORDER BY hour
        LIMIT 24
    """)

    if hourly_stats:
        print("\nЧас | Транзакций | График")
        print("-" * 40)
        max_cnt = max(h[1] for h in hourly_stats) if hourly_stats else 1

        for hour, cnt, volume in hourly_stats:
            bar_length = int(cnt / max_cnt * 30)
            bar = "█" * bar_length
            print(f"{hour:02d}  | {cnt:6,}    | {bar}")

    # 8. РАСПРЕДЕЛЕНИЕ ПО ПОЛУ
    print("\n" + "-" * 60)
    print("8️⃣ РАСПРЕДЕЛЕНИЕ ПО ПОЛУ")
    print("-" * 60)

    gender_stats = client.execute(f"""
        SELECT 
            gender,
            count() as cnt,
            sum(toFloat64OrNull(amount_uzs)) as volume,
            avg(toFloat64OrNull(amount_uzs)) as avg_amount
        FROM {table_to_use}
        WHERE gender IN ('M', 'F', 'Male', 'Female')
        GROUP BY gender
    """)

    for gender, cnt, volume, avg_amt in gender_stats:
        percent = cnt / general_stats[0] * 100
        volume_str = f"{volume:,.0f}" if volume else "0"
        avg_str = f"{avg_amt:,.0f}" if avg_amt else "0"
        emoji = "👨" if gender in ('M', 'Male') else "👩"
        gender_name = "Мужчины" if gender in ('M', 'Male') else "Женщины"
        print(f"{emoji} {gender_name}: {cnt:,} ({percent:.1f}%) | Объем: {volume_str} UZS | Средний: {avg_str} UZS")

    # 9. ТИПЫ КАРТ
    print("\n" + "-" * 60)
    print("9️⃣ ТИПЫ КАРТ")
    print("-" * 60)

    card_stats = client.execute(f"""
        SELECT 
            card_type,
            count() as cnt,
            uniq(hpan) as unique_cards,
            sum(toFloat64OrNull(amount_uzs)) as volume
        FROM {table_to_use}
        WHERE card_type != ''
        GROUP BY card_type
        ORDER BY cnt DESC
        LIMIT 10
    """)

    for card_type, cnt, cards, volume in card_stats:
        percent = cnt / general_stats[0] * 100
        volume_str = f"{volume:,.0f}" if volume else "0"
        print(f"💳 {card_type}: {cnt:,} ({percent:.1f}%) | Карт: {cards:,} | Объем: {volume_str} UZS")

    # 10. КАЧЕСТВО ДАННЫХ
    print("\n" + "-" * 60)
    print("🔟 КАЧЕСТВО ДАННЫХ")
    print("-" * 60)

    quality_stats = client.execute(f"""
        SELECT 
            countIf(hpan = '') as empty_hpan,
            countIf(amount_uzs = '') as empty_amount,
            countIf(mcc = '') as empty_mcc,
            countIf(merchant_name = '') as empty_merchant,
            countIf(pinfl = '') as empty_pinfl
        FROM {table_to_use}
    """)[0]

    total = general_stats[0]
    print(f"Пустые значения:")
    print(f"  • hpan: {quality_stats[0]:,} ({quality_stats[0] / total * 100:.1f}%)")
    print(f"  • amount_uzs: {quality_stats[1]:,} ({quality_stats[1] / total * 100:.1f}%)")
    print(f"  • mcc: {quality_stats[2]:,} ({quality_stats[2] / total * 100:.1f}%)")
    print(f"  • merchant_name: {quality_stats[3]:,} ({quality_stats[3] / total * 100:.1f}%)")
    print(f"  • pinfl: {quality_stats[4]:,} ({quality_stats[4] / total * 100:.1f}%)")

    print("\n" + "=" * 60)
    print("✅ АНАЛИЗ ЗАВЕРШЕН!")
    print("=" * 60)

    print("\n📊 Рекомендации:")
    print("  1. Создайте оптимизированную таблицу: python fix_optimized_table.py")
    print("  2. Запустите дашборд: streamlit run run_app.py")
    print("  3. Используйте ClickHouse UI для SQL запросов: http://localhost:8123/play")

    # Сохраняем результаты
    print("\n💾 Сохранение результатов в analysis_results.txt...")
    with open('analysis_results.txt', 'w', encoding='utf-8') as f:
        f.write(f"Анализ карточных транзакций\n")
        f.write(f"Дата: {datetime.now()}\n")
        f.write(f"Таблица: {table_to_use}\n")
        f.write(f"Всего записей: {general_stats[0]:,}\n")
        f.write(f"Общий объем: {general_stats[1]:,.0f} UZS\n" if general_stats[1] else "")

    print("✅ Результаты сохранены в analysis_results.txt")


if __name__ == "__main__":
    run_analysis()