#!/usr/bin/env python3
"""
Скрипт для проверки структуры таблиц в ClickHouse
"""

from clickhouse_driver import Client


def check_table_structure():
    """Проверяет структуру таблиц и типы данных"""

    print("\n" + "=" * 60)
    print("ПРОВЕРКА СТРУКТУРЫ ТАБЛИЦ")
    print("=" * 60)

    # Подключение
    client = Client(
        host='localhost',
        port=9000,
        user='analyst',
        password='admin123',
        database='card_analytics'
    )

    tables_to_check = ['transactions_simple', 'transactions_optimized']

    for table_name in tables_to_check:
        print(f"\n📊 Таблица: {table_name}")
        print("-" * 40)

        try:
            # Получаем структуру таблицы
            structure = client.execute(f"DESCRIBE {table_name}")

            # Ключевые поля для проверки
            key_fields = ['amount_uzs', 'p2p_flag', 'hour_num', 'rday', 'transaction_date', 'mcc']

            print("Ключевые поля:")
            for field_name, field_type, *_ in structure:
                if field_name in key_fields:
                    print(f"  • {field_name}: {field_type}")

            # Проверяем несколько записей
            print("\nПример данных (первая запись):")
            sample = client.execute(f"""
                SELECT 
                    amount_uzs,
                    p2p_flag,
                    hour_num,
                    rday,
                    mcc
                FROM {table_name}
                LIMIT 1
            """)

            if sample:
                row = sample[0]
                print(f"  amount_uzs: {row[0]} (тип: {type(row[0]).__name__})")
                print(f"  p2p_flag: {row[1]} (тип: {type(row[1]).__name__})")
                print(f"  hour_num: {row[2]} (тип: {type(row[2]).__name__})")
                print(f"  rday: {row[3]} (тип: {type(row[3]).__name__})")
                print(f"  mcc: {row[4]} (тип: {type(row[4]).__name__})")

            # Статистика
            stats = client.execute(f"""
                SELECT 
                    count() as total,
                    count(amount_uzs) as amount_filled,
                    count(p2p_flag) as p2p_filled
                FROM {table_name}
            """)[0]

            print(f"\nСтатистика:")
            print(f"  Всего записей: {stats[0]:,}")
            print(f"  Заполнено amount_uzs: {stats[1]:,}")
            print(f"  Заполнено p2p_flag: {stats[2]:,}")

        except Exception as e:
            print(f"  ❌ Ошибка: {e}")

    print("\n" + "=" * 60)
    print("✅ ПРОВЕРКА ЗАВЕРШЕНА")
    print("=" * 60)


if __name__ == "__main__":
    check_table_structure()