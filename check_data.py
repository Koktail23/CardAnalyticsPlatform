# check_data.py
"""
Проверка загруженных данных в ClickHouse
"""

from clickhouse_driver import Client
import sys


def check_clickhouse_data():
    """Проверка данных в ClickHouse"""

    print("\n" + "=" * 60)
    print("ПРОВЕРКА ДАННЫХ В CLICKHOUSE")
    print("=" * 60)

    try:
        # Подключение
        client = Client(
            host='localhost',
            port=9000,
            user='analyst',
            password='admin123'
        )

        # Проверка таблиц
        print("\n📊 ТАБЛИЦЫ В card_analytics:")
        print("-" * 40)

        tables = client.execute('''
            SELECT name, total_rows, formatReadableSize(total_bytes) as size
            FROM system.tables
            WHERE database = 'card_analytics'
                AND name NOT LIKE '.%'  -- Исключаем системные таблицы
            ORDER BY name
        ''')

        if tables:
            for name, rows, size in tables:
                print(f"  • {name}: {rows:,} строк ({size})")
        else:
            print("  ❌ Таблицы не найдены")
            return

        # Проверка данных в transactions_main
        count = client.execute('SELECT count() FROM card_analytics.transactions_main')[0][0]

        if count == 0:
            print("\n❌ Таблица transactions_main пустая!")
            print("Запустите загрузку данных:")
            print("  python load_data_fixed.py")
            return

        print(f"\n✅ В таблице transactions_main: {count:,} записей")

        # Детальная статистика
        print("\n📈 СТАТИСТИКА ДАННЫХ:")
        print("-" * 40)

        # Период данных
        date_range = client.execute('''
            SELECT 
                toDate(min(rday)) as min_date,
                toDate(max(rday)) as max_date,
                dateDiff('day', min_date, max_date) as days
            FROM card_analytics.transactions_main
        ''')[0]

        print(f"\n📅 Период данных:")
        print(f"  С {date_range[0]} по {date_range[1]} ({date_range[2]} дней)")

        # Основные метрики
        metrics = client.execute('''
            SELECT
                count() as total,
                sum(amount_uzs) as volume,
                avg(amount_uzs) as avg_amount,
                max(amount_uzs) as max_amount,
                uniq(hpan) as unique_cards,
                uniq(pinfl) as unique_clients,
                uniq(merchant_name) as unique_merchants
            FROM card_analytics.transactions_main
        ''')[0]

        print(f"\n💰 Финансовые показатели:")
        print(f"  • Общий объем: {metrics[1]:,.0f} UZS")
        print(f"  • Средняя сумма: {metrics[2]:,.0f} UZS")
        print(f"  • Максимальная сумма: {metrics[3]:,.0f} UZS")

        print(f"\n👥 Уникальные сущности:")
        print(f"  • Карт: {metrics[4]:,}")
        print(f"  • Клиентов: {metrics[5]:,}")
        print(f"  • Мерчантов: {metrics[6]:,}")

        # MCC статистика
        print(f"\n🏪 Топ-5 MCC категорий:")
        mcc_stats = client.execute('''
            SELECT mcc, count() as cnt, sum(amount_uzs) as volume
            FROM card_analytics.transactions_main
            GROUP BY mcc
            ORDER BY cnt DESC
            LIMIT 5
        ''')

        for mcc, cnt, volume in mcc_stats:
            print(f"  • MCC {mcc}: {cnt:,} транзакций ({volume:,.0f} UZS)")

        # P2P статистика
        p2p_stats = client.execute('''
            SELECT 
                p2p_flag,
                count() as cnt,
                sum(amount_uzs) as volume,
                avg(amount_uzs) as avg_amount
            FROM card_analytics.transactions_main
            GROUP BY p2p_flag
        ''')

        print(f"\n💸 P2P статистика:")
        for flag, cnt, volume, avg_amt in p2p_stats:
            p2p_type = "P2P" if flag == 1 else "Покупки"
            percent = cnt / count * 100
            print(f"  • {p2p_type}: {cnt:,} ({percent:.1f}%) - {volume:,.0f} UZS")

        # Банки
        print(f"\n🏦 Топ-5 банков-эмитентов:")
        bank_stats = client.execute('''
            SELECT emitent_bank, count() as cnt
            FROM card_analytics.transactions_main
            WHERE emitent_bank != ''
            GROUP BY emitent_bank
            ORDER BY cnt DESC
            LIMIT 5
        ''')

        for bank, cnt in bank_stats:
            print(f"  • {bank}: {cnt:,} транзакций")

        # Регионы
        print(f"\n📍 Топ-5 регионов:")
        region_stats = client.execute('''
            SELECT emitent_region, count() as cnt
            FROM card_analytics.transactions_main
            WHERE emitent_region != ''
            GROUP BY emitent_region
            ORDER BY cnt DESC
            LIMIT 5
        ''')

        for region, cnt in region_stats:
            print(f"  • {region}: {cnt:,} транзакций")

        print("\n" + "=" * 60)
        print("✅ ДАННЫЕ УСПЕШНО ЗАГРУЖЕНЫ И ГОТОВЫ К АНАЛИЗУ!")
        print("=" * 60)

        print("\n🚀 Следующие шаги:")
        print("  1. Запустить дашборд: streamlit run run_app.py")
        print("  2. Открыть ClickHouse UI: http://localhost:8123/play")
        print("  3. Начать анализ данных по плану (Неделя 2)")

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        print("\nПроверьте:")
        print("  1. Docker запущен: docker-compose ps")
        print("  2. Данные загружены: python load_data_fixed.py")


if __name__ == "__main__":
    check_clickhouse_data()